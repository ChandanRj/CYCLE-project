"""Checks for cycle.stats (NumPy only)."""

import numpy as np
import pytest

from cycle.stats import paired_bootstrap_diff, summarize_seeds


def test_summarize_constant_values_has_zero_spread():
    s = summarize_seeds([0.3, 0.3, 0.3])
    assert s["mean"] == pytest.approx(0.3)
    assert s["sd"] == pytest.approx(0.0)
    assert s["ci_low"] == pytest.approx(0.3) and s["ci_high"] == pytest.approx(0.3)


def test_summarize_reports_mean_sd_and_range():
    s = summarize_seeds([0.2, 0.3, 0.4])
    assert s["mean"] == pytest.approx(0.3)
    assert s["sd"] == pytest.approx(0.1)
    assert s["range"] == pytest.approx(0.2)
    assert s["ci_low"] <= s["mean"] <= s["ci_high"]


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize_seeds([])


def test_paired_identical_systems_no_difference():
    x = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    r = paired_bootstrap_diff(x, x, n_boot=2000)
    assert r["diff"] == 0.0
    assert r["p_value"] == pytest.approx(1.0)


def test_paired_clearly_better_system_has_ci_above_zero():
    rng = np.random.default_rng(1)
    b = rng.random(400) < 0.5
    a = b | (rng.random(400) < 0.3)  # A is right wherever B is, plus extra
    r = paired_bootstrap_diff(a.astype(int), b.astype(int), n_boot=4000)
    assert r["diff"] > 0
    assert r["ci_low"] > 0
    assert r["p_value"] < 0.05


def test_paired_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 0], [1, 0, 1])
