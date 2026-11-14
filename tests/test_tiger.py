"""CPU checks for the TIGER ablation terms (cycle.models.tiger)."""

import os
import random

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from cycle.config import Config, load_config
from cycle.models.tiger import (GCN, TigerBranches, add_tiger_terms, build_tiger,
                                distinct_loss, gcn_normalized_adj, hsic, hsic_dense,
                                shared_consistency_loss, shared_consistency_loss_dense)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CFG = os.path.join(REPO, "configs", "default.yaml")


def _cycle_loss(cfg, seed=0):
    """CYCLE's fixed-weight loss a*L_e + b*L_f + c*L_r on a fixed random batch."""
    g = torch.Generator().manual_seed(seed)
    ctx = torch.randn(8, 16, generator=g)
    cand = torch.randn(8, 16, generator=g)
    scores = ctx @ cand.t()
    loss_e = F.cross_entropy(scores, torch.arange(8))
    loss_f = torch.rand((), generator=g)
    loss_r = torch.rand((), generator=g)
    w = cfg.loss
    return w.a * loss_e + w.b * loss_f + w.c * loss_r


# --- a. weights 0: loss bit-identical, nothing built, no RNG drawn ----------

def test_default_config_keeps_cycle_weights_and_tiger_off():
    cfg = load_config(DEFAULT_CFG)
    assert (cfg.loss.a, cfg.loss.b, cfg.loss.c) == (1.0, 1.0, 1.0)
    assert cfg.loss.shared_consistency_weight == 0.0 and cfg.loss.hsic_weight == 0.0


def test_weights_zero_loss_is_bit_identical():
    cfg = load_config(DEFAULT_CFG)
    loss = _cycle_loss(cfg)
    reference = _cycle_loss(cfg)                  # "current" loss, computed separately
    branches, inputs = build_tiger(cfg, torch.device("cpu"))
    assert branches is None and inputs is None
    out = add_tiger_terms(loss, cfg.loss, branches, inputs, {})
    assert out is loss
    assert out.dtype == reference.dtype
    assert out.view(torch.int32).item() == reference.view(torch.int32).item()


def test_weights_zero_consumes_no_rng():
    cfg = load_config(DEFAULT_CFG)
    torch.manual_seed(123); np.random.seed(123); random.seed(123)
    before = (torch.get_rng_state().clone(), np.random.get_state()[1].copy(), random.getstate())
    build_tiger(cfg, torch.device("cpu"))
    after = (torch.get_rng_state(), np.random.get_state()[1], random.getstate())
    assert torch.equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])
    assert before[2] == after[2]


def test_missing_required_keys_raise_when_active():
    cfg = load_config(DEFAULT_CFG, ["loss.hsic_weight=0.1"])
    raw = {k: v for k, v in cfg.items() if k != "tiger"}
    with pytest.raises(KeyError):
        build_tiger(Config.wrap(raw), torch.device("cpu"))


# --- b. shared consistency loss (Eq. 9) -------------------------------------

def test_shared_consistency_zero_when_branches_equal():
    z = torch.randn(50, 7, generator=torch.Generator().manual_seed(1))
    assert shared_consistency_loss(z, z.clone()).item() == 0.0
    assert shared_consistency_loss_dense(z, z.clone()).item() == 0.0


def test_shared_consistency_matches_literal_equation():
    g = torch.Generator().manual_seed(2)
    a, b = torch.randn(40, 6, generator=g), torch.randn(40, 6, generator=g)
    fast = shared_consistency_loss(a.double(), b.double())
    assert fast.item() == pytest.approx(shared_consistency_loss_dense(a, b).item(), rel=1e-10)
    assert fast.item() > 0


# --- c. HSIC (Eq. 10-12) -----------------------------------------------------

def test_hsic_non_negative():
    g = torch.Generator().manual_seed(3)
    for _ in range(50):
        a, b = torch.randn(30, 5, generator=g), torch.randn(30, 4, generator=g)
        assert hsic(a, b).item() >= 0.0
        assert hsic_dense(a, b).item() >= -1e-12


def test_hsic_zero_when_one_input_constant():
    g = torch.Generator().manual_seed(4)
    a = torch.randn(30, 5, generator=g)
    const = torch.full((30, 3), 2.0)
    assert hsic(const, a).item() == 0.0              # R Z_a = 0 exactly
    assert hsic(a, const).item() == pytest.approx(0.0, abs=1e-10)
    assert hsic_dense(a, const).item() == pytest.approx(0.0, abs=1e-10)
    assert hsic_dense(const, a).item() == pytest.approx(0.0, abs=1e-10)


def test_hsic_matches_literal_equation():
    g = torch.Generator().manual_seed(5)
    a, b = torch.randn(25, 6, generator=g), torch.randn(25, 3, generator=g)
    assert hsic(a.double(), b.double()).item() == pytest.approx(hsic_dense(a, b).item(), rel=1e-10)


def test_distinct_loss_is_sum_of_two_hsic():
    g = torch.Generator().manual_seed(6)
    zr, zf, zsr, zsf = (torch.randn(20, 4, generator=g) for _ in range(4))
    assert distinct_loss(zr, zf, zsr, zsf).item() == pytest.approx(
        (hsic(zr, zsr) + hsic(zf, zsf)).item())


# --- GCN form (Eq. 6, 7) -----------------------------------------------------

def test_gcn_matches_literal_equation():
    rng = np.random.default_rng(7)
    n, m, h = 12, 9, 5
    a = (rng.random((n, n)) < 0.3).astype(np.float64)
    a = np.triu(a, 1); a = a + a.T
    x = (rng.random((n, m)) < 0.4).astype(np.float32)
    at = a + np.eye(n)
    d = np.diag(at.sum(1) ** -0.5)
    a_hat = torch.tensor(d @ at @ d, dtype=torch.float32)
    torch.manual_seed(0)
    gcn = GCN(m, h, 2)
    xt = torch.tensor(x)
    expected = xt
    for lin in gcn.weights:
        expected = torch.relu(a_hat @ expected @ lin.weight.t())
    got = gcn(gcn_normalized_adj(sp.csr_matrix(a)), xt.to_sparse())
    assert torch.allclose(got, expected, atol=1e-5)


def test_shared_branch_uses_one_weight_set():
    br = TigerBranches(9, 4, 2)
    names = {n.split(".")[0] for n, _ in br.named_parameters()}
    assert names == {"gcn_r", "gcn_f", "gcn_s"}
    n = 6
    adj_r = gcn_normalized_adj(sp.csr_matrix(np.eye(n)[[1, 0, 3, 2, 5, 4]]))
    inputs = {"X": torch.rand(n, 9).to_sparse(), "adj_r": adj_r, "adj_f": adj_r}
    z_r, z_f, z_sr, z_sf = br(inputs)
    assert torch.equal(z_sr, z_sf)                   # same W_s, same graph -> same output


# --- whole-graph run on the real train-year graph (CPU) ---------------------

GRAPH_2019 = os.path.join(REPO, "data", "processed", "graphs", "2019.npz")


@pytest.mark.skipif(not os.path.exists(GRAPH_2019), reason="processed 2019 graph not present")
def test_active_terms_on_full_2019_graph_cpu():
    cfg = load_config(DEFAULT_CFG, ["loss.shared_consistency_weight=0.5",
                                    "loss.hsic_weight=0.5"])
    torch.manual_seed(0)
    branches, inputs = build_tiger(cfg, torch.device("cpu"))
    assert inputs["num_nodes"] == 10373
    assert tuple(inputs["X"].shape) == (10373, 2929)
    base = _cycle_loss(cfg)
    extra = {}
    out = add_tiger_terms(base, cfg.loss, branches, inputs, extra)
    assert torch.isfinite(out)
    assert extra["loss_s"] >= 0 and extra["loss_d"] >= 0
    assert out.item() == pytest.approx(base.item() + 0.5 * extra["loss_s"] + 0.5 * extra["loss_d"],
                                       rel=1e-5)
    out.backward()
    assert all(p.grad is not None for p in branches.parameters())
