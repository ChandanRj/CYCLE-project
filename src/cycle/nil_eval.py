"""NIL evaluation from top-k score dumps (NumPy only).

The GPU side (scripts/dump_topk.py) saves, for each mention, the top-k pool
scores and pool indices from a trained bi-encoder. Every NIL rule in
cycle.nil only needs the top-1 and top-2 scores, so top-k dumps (k >= 2) are
enough. The full (mentions x pool) matrix never has to be stored.

Dump format (.npz):
    scores   float32 (n, k)  top-k scores, sorted descending
    indices  int64   (n, k)  pool index of each top-k candidate
    gold     int64   (n,)    gold pool index, or NIL (-1) if the gold entity
                             is not in the KB this dump was scored against

Protocol: the NIL threshold is chosen on the validation dump only, then
applied unchanged to the test dump.
"""

from __future__ import annotations

import numpy as np

from cycle.nil import NIL, OBJECTIVES, metrics, nil_score


def load_topk(path):
    data = np.load(path)
    scores = np.asarray(data["scores"], dtype=np.float64)
    indices = np.asarray(data["indices"], dtype=np.int64)
    gold = np.asarray(data["gold"], dtype=np.int64)
    if scores.ndim != 2 or indices.shape != scores.shape or gold.shape[0] != scores.shape[0]:
        raise ValueError(f"{path}: inconsistent shapes")
    if scores.shape[1] < 2:
        raise ValueError(f"{path}: need k >= 2 candidates per mention")
    return scores, indices, gold


def predict_topk(scores, indices, threshold, rule="max"):
    """Top-1 pool index, or NIL where the NIL score exceeds the threshold."""
    nil = nil_score(scores, rule)
    return np.where(nil > threshold, NIL, indices[:, 0])


def calibrate_topk(scores, indices, gold, rule="max",
                   objective="unnormalized_acc", max_candidates=512):
    """Threshold that maximises ``objective`` on a validation dump.

    Candidate thresholds come from observed NIL scores, subsampled to at most
    ``max_candidates`` values, plus +inf (never predict NIL). Ties keep the
    lowest threshold that reaches the best value, so the choice is stable.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}")
    nil = nil_score(scores, rule)
    top = indices[:, 0]

    uniq = np.unique(nil)
    if uniq.size > max_candidates:
        uniq = np.unique(np.quantile(uniq, np.linspace(0.0, 1.0, max_candidates)))
    thresholds = np.concatenate([uniq, [np.inf]])

    best_t, best_m, best_v = np.inf, None, -np.inf
    for t in thresholds:
        pred = np.where(nil > t, NIL, top)
        m = metrics(pred, gold)
        v = m[objective]
        if np.isnan(v):
            continue
        if v > best_v:
            best_t, best_m, best_v = float(t), m, v
    if best_m is None:
        raise ValueError("objective undefined for every threshold")
    return best_t, best_m


def evaluate_nil(val, test, rule="max", objective="unnormalized_acc"):
    """Calibrate on ``val`` (a (scores, indices, gold) tuple), report on ``test``.

    Returns a dict with the chosen threshold, validation metrics, test metrics
    at that threshold, and the always-link baseline on test (no NIL at all).
    """
    vs, vi, vg = val
    ts, ti, tg = test
    threshold, val_m = calibrate_topk(vs, vi, vg, rule=rule, objective=objective)
    val_nil_predictions = int(np.sum(nil_score(vs, rule) > threshold))
    test_pred = predict_topk(ts, ti, threshold, rule)
    return {
        "rule": rule,
        "objective": objective,
        "threshold": threshold,
        "val_nil_predictions": val_nil_predictions,
        "val": val_m,
        "test": metrics(test_pred, tg),
        "test_always_link": metrics(ti[:, 0], tg),
    }
