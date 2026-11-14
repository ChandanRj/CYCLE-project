"""Uncertainty for comparisons between runs (NumPy only).

Two levels of uncertainty matter here, and PROGRESS.md showed that the
seed level dominates:

* Seed level: ``summarize_seeds`` gives mean, SD and a bootstrap CI over
  independent training seeds. Report this for every arm.
* Test-mention level: ``paired_bootstrap_diff`` resamples the *same* test
  mentions for two systems, so it captures evaluation noise for one checkpoint
  pair. It does not capture training-seed noise; use it only alongside the
  seed-level summary.
"""

from __future__ import annotations

import numpy as np


def summarize_seeds(values, n_boot=10000, alpha=0.05, seed=0):
    """Mean, sample SD and percentile-bootstrap CI over seeds.

    With few seeds (3-5) the CI is wide and approximate; say so in the report.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("values must be a non-empty 1-D sequence")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot_means = x[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if x.size > 1 else float("nan"),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "range": float(x.max() - x.min()),
    }


def paired_bootstrap_diff(correct_a, correct_b, n_boot=10000, alpha=0.05, seed=0):
    """Paired bootstrap CI for accuracy(A) - accuracy(B) on the same mentions.

    ``correct_a`` and ``correct_b`` are 0/1 arrays aligned by mention.
    Returns the observed difference, the CI and a two-sided bootstrap p-value
    for "no difference" (the share of resamples on the other side of zero,
    doubled, capped at 1).
    """
    a = np.asarray(correct_a, dtype=np.float64)
    b = np.asarray(correct_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("correct_a and correct_b must be aligned, non-empty 1-D arrays")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boot = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    p_one = min(np.mean(boot <= 0), np.mean(boot >= 0))
    return {
        "diff": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(1.0, 2 * p_one)),
        "n_mentions": int(d.size),
    }
