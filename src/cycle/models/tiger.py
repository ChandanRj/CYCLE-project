"""TIGER loss terms as an ablation for CYCLE (off by default).

Equations from TIGER (Zhang, Cao, Groth 2024), Sec. 5.2-5.3:

  Eq. 6   distinct GCN, one per graph:
          Z^(l) = ReLU( D̃^-1/2 Ã D̃^-1/2 Z^(l-1) W^(l) ),  Ã = A + I,
          D̃ = diag(row sums of Ã),  Z^(0) = X (feature matrix)
  Eq. 7   shared GCN: the same form, one weight set W_s used for both the
          structure-graph (A_r) and the feature-graph (A_f) inputs
  Eq. 9   L_s = || Z_sr Z_sr^T - Z_sf Z_sf^T ||_F^2
  Eq. 10-11  HSIC(Z_a, Z_b) = (n-1)^-2 tr(R K_a R K_b),
          R = I - (1/n) e e^T,  K = Z Z^T (linear kernel),  n = #entities
  Eq. 12  L_d = HSIC(Z_r, Z_sr) + HSIC(Z_f, Z_sf)
  Eq. 13  L = L_e + a L_s + b L_d

In this repo the two TIGER weights are ``loss.shared_consistency_weight``
(TIGER's a) and ``loss.hsic_weight`` (TIGER's b). CYCLE's own ``loss.a/b/c``
(weights of L_e, L_f, L_r) are unchanged, so the ablation objective is
CYCLE's loss + shared_consistency_weight * L_s + hsic_weight * L_d.
Both weights default to 0; then no TIGER module is built and train.py is
unchanged.

Inputs (repo choice, not TIGER's own construction): the train year's
processed graph, data/processed/graphs/{year}.npz. A_r = relation graph,
A_f = the repo's kNN description graph, X = the repo's binary 2,929-column
token feature matrix. Both adjacencies are symmetric and binary with no
self-loops (checked for 2019), so Ã = A + I is the standard self-loop
augmentation. X is used raw (binary, Z^(0) = X), not row-normalised.

GCN depth and width (``tiger.gcn_layers``, ``tiger.hidden_dim``) are not
given in the TIGER paper or its supplementary (Zenodo 10.5281/zenodo.12790573).
They are required config keys with no code default; configs/default.yaml sets
2 layers x 512 as OUR choice for this ablation. The GCN layers have no bias,
following Eq. 6 (Z W only).

Whole-graph computation and memory: TIGER runs the GCNs on the full entity
graph (n = 10,373 per year here), not per batch, and so does this module.
A dense n x n matrix in float32 is n^2 * 4 B ~= 430 MB for n = 10,373 (and
twice that in float64). Eq. 9-11 involve three kinds of n x n matrices
(Ã, K = Z Z^T and R). This module never materialises any of them:

  * Ã is kept as a torch sparse tensor (nnz = edges + n).
  * HSIC uses the exact identity
        tr(R Z_a Z_a^T R Z_b Z_b^T) = || (R Z_a)^T Z_b ||_F^2,
    which follows from R = R^T and R^2 = R. R Z_a is Z_a with column means
    removed, so only a d x d matrix is formed.
  * L_s uses the exact expansion of the Frobenius norm
        ||A A^T - B B^T||_F^2 = ||A^T A||_F^2 + ||B^T B||_F^2 - 2 ||A^T B||_F^2,
    again d x d only.

These reductions run in float64 to limit cancellation in L_s. The dense
reference implementations (``*_dense``) follow the equations literally and
are used only by the unit tests on small n.
"""

from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def gcn_normalized_adj(adj):
    """D̃^-1/2 (A + I) D̃^-1/2 for a scipy sparse A (Eq. 6), as a torch sparse tensor."""
    a = sp.csr_matrix(adj, dtype=np.float64)
    n = a.shape[0]
    a_tilde = a + sp.identity(n, dtype=np.float64, format="csr")
    d = np.asarray(a_tilde.sum(axis=1)).ravel()          # >= 1 because of I
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(d))
    norm = (d_inv_sqrt @ a_tilde @ d_inv_sqrt).tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((norm.row, norm.col)).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(norm.data), (n, n)).coalesce()


def load_tiger_inputs(data_root, year, device):
    """Raw binary A_r, A_f and X of one year's processed graph, ready for the GCNs."""
    z = np.load(os.path.join(data_root, "processed", "graphs", f"{year}.npz"))
    n = int(z["n"])

    def csr(prefix, shape):
        idx = z[f"{prefix}_indices"]
        return sp.csr_matrix((np.ones(len(idx), np.float32), idx, z[f"{prefix}_indptr"]),
                             shape=shape)

    x = csr("X", tuple(z["X_shape"])).tocoo()
    x_t = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack((x.row, x.col)).astype(np.int64)),
        torch.from_numpy(x.data), x.shape).coalesce()
    return {
        "num_nodes": n,
        "adj_r": gcn_normalized_adj(csr("rel", (n, n))).to(device),
        "adj_f": gcn_normalized_adj(csr("knn", (n, n))).to(device),
        "X": x_t.to(device),
    }


# ---------------------------------------------------------------------------
# GCN branches (Eq. 6, 7)
# ---------------------------------------------------------------------------

def _mm(a, b):
    return torch.sparse.mm(a, b) if a.is_sparse else a @ b


class GCN(nn.Module):
    """Z^(l) = ReLU(Â Z^(l-1) W^(l)), l = 1..num_layers, Z^(0) = X, no bias."""

    def __init__(self, in_dim, hidden_dim, num_layers):
        super().__init__()
        if num_layers < 1:
            raise ValueError("tiger.gcn_layers must be >= 1")
        dims = [in_dim] + [hidden_dim] * num_layers
        self.weights = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(num_layers))

    def forward(self, adj_norm, x):
        z = x
        for lin in self.weights:
            zw = _mm(z, lin.weight.t())          # Z^(l-1) W^(l); X may be sparse
            z = F.relu(_mm(adj_norm, zw))        # Â (Z W)
        return z


class TigerBranches(nn.Module):
    """Distinct GCNs for A_r and A_f, plus one shared GCN (W_s) used for both."""

    def __init__(self, in_dim, hidden_dim, num_layers):
        super().__init__()
        self.gcn_r = GCN(in_dim, hidden_dim, num_layers)    # distinct, structure graph
        self.gcn_f = GCN(in_dim, hidden_dim, num_layers)    # distinct, feature graph
        self.gcn_s = GCN(in_dim, hidden_dim, num_layers)    # shared weights W_s

    def forward(self, inputs):
        x, a_r, a_f = inputs["X"], inputs["adj_r"], inputs["adj_f"]
        return (self.gcn_r(a_r, x), self.gcn_f(a_f, x),
                self.gcn_s(a_r, x), self.gcn_s(a_f, x))      # Z_r, Z_f, Z_sr, Z_sf


# ---------------------------------------------------------------------------
# losses (Eq. 9-12)
# ---------------------------------------------------------------------------

def shared_consistency_loss(z_sr, z_sf):
    """Eq. 9, ||Z_sr Z_sr^T - Z_sf Z_sf^T||_F^2, via the d x d Gram expansion (exact)."""
    a, b = z_sr.double(), z_sf.double()
    ata, btb, atb = a.t() @ a, b.t() @ b, a.t() @ b
    return (ata.pow(2).sum() + btb.pow(2).sum() - 2 * atb.pow(2).sum()).to(z_sr.dtype)


def hsic(z_a, z_b):
    """Eq. 10-11, biased HSIC with linear kernels: (n-1)^-2 ||(R Z_a)^T Z_b||_F^2 (exact)."""
    n = z_a.shape[0]
    if z_b.shape[0] != n or n < 2:
        raise ValueError("HSIC needs two inputs with the same n >= 2 rows")
    a = z_a.double()
    a = a - a.mean(dim=0, keepdim=True)          # R Z_a
    cross = a.t() @ z_b.double()
    return (cross.pow(2).sum() / (n - 1) ** 2).to(z_a.dtype)


def distinct_loss(z_r, z_f, z_sr, z_sf):
    """Eq. 12, L_d = HSIC(Z_r, Z_sr) + HSIC(Z_f, Z_sf)."""
    return hsic(z_r, z_sr) + hsic(z_f, z_sf)


def shared_consistency_loss_dense(z_sr, z_sf):
    """Literal Eq. 9 with n x n matrices. Tests only (O(n^2) memory)."""
    a, b = z_sr.double(), z_sf.double()
    return (a @ a.t() - b @ b.t()).pow(2).sum()


def hsic_dense(z_a, z_b):
    """Literal Eq. 10-11 with n x n matrices. Tests only (O(n^2) memory)."""
    a, b = z_a.double(), z_b.double()
    n = a.shape[0]
    r = torch.eye(n, dtype=torch.float64) - torch.full((n, n), 1.0 / n, dtype=torch.float64)
    ka, kb = a @ a.t(), b @ b.t()
    return torch.trace(r @ ka @ r @ kb) / (n - 1) ** 2


# ---------------------------------------------------------------------------
# train.py hooks
# ---------------------------------------------------------------------------

def tiger_weights(loss_cfg):
    """(shared_consistency_weight, hsic_weight); missing keys mean 0 (off)."""
    w_s = float(loss_cfg.get("shared_consistency_weight", 0.0) or 0.0)
    w_d = float(loss_cfg.get("hsic_weight", 0.0) or 0.0)
    if w_s < 0 or w_d < 0:
        raise ValueError("TIGER loss weights must be >= 0")
    return w_s, w_d


def build_tiger(cfg, device, logger=None):
    """Return (TigerBranches, inputs) if a TIGER weight is > 0, else (None, None).

    When both weights are 0, nothing is built or loaded and no random numbers
    are drawn, so the CYCLE run is unchanged.
    """
    w_s, w_d = tiger_weights(cfg.loss)
    if w_s == 0 and w_d == 0:
        return None, None
    t = cfg.get("tiger")
    if not t or "gcn_layers" not in t or "hidden_dim" not in t:
        raise KeyError("TIGER loss weights > 0 need tiger.gcn_layers and tiger.hidden_dim "
                       "(required: not given by the TIGER paper, no code default)")
    inputs = load_tiger_inputs(cfg.data.root, cfg.data.train_year, device)
    branches = TigerBranches(inputs["X"].shape[1], int(t["hidden_dim"]),
                             int(t["gcn_layers"])).to(device)
    if logger is not None:
        logger.info(f"TIGER ablation: n={inputs['num_nodes']} entities, "
                    f"{t['gcn_layers']} GCN layers x {t['hidden_dim']}, "
                    f"shared_consistency_weight={w_s}, hsic_weight={w_d}")
    return branches, inputs


def add_tiger_terms(loss, loss_cfg, branches, inputs, extra):
    """Return ``loss`` plus the weighted TIGER terms.

    With ``branches is None`` (both weights 0), ``loss`` is returned as is:
    the very same object, so the result is bit-identical to CYCLE's loss.
    """
    if branches is None:
        return loss
    w_s, w_d = tiger_weights(loss_cfg)
    z_r, z_f, z_sr, z_sf = branches(inputs)
    if w_s > 0:
        loss_s = shared_consistency_loss(z_sr, z_sf)
        loss = loss + w_s * loss_s
        extra["loss_s"] = loss_s.item()
    if w_d > 0:
        loss_d = distinct_loss(z_r, z_f, z_sr, z_sf)
        loss = loss + w_d * loss_d
        extra["loss_d"] = loss_d.item()
    return loss
