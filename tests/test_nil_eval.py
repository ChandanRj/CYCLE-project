"""Checks for cycle.nil_eval on synthetic top-k dumps (NumPy only)."""

import numpy as np
import pytest

from cycle.nil import NIL
from cycle.nil_eval import calibrate_topk, evaluate_nil, load_topk, predict_topk


def _dump(rows):
    scores = np.array([r[0] for r in rows], dtype=np.float64)
    indices = np.array([r[1] for r in rows], dtype=np.int64)
    gold = np.array([r[2] for r in rows], dtype=np.int64)
    return scores, indices, gold


# Rows 0-2 are confident, in-KB. Rows 3-5 are weak and their gold is NIL.
VAL = _dump([
    ([0.9, 0.1, 0.0], [5, 1, 2], 5),
    ([0.8, 0.2, 0.1], [3, 0, 1], 3),
    ([0.7, 0.3, 0.2], [7, 8, 9], 7),
    ([0.2, 0.1, 0.05], [4, 6, 2], NIL),
    ([0.25, 0.1, 0.0], [1, 2, 3], NIL),
    ([0.3, 0.2, 0.1], [0, 5, 6], NIL),
])
TEST = _dump([
    ([0.85, 0.2, 0.1], [2, 3, 4], 2),
    ([0.75, 0.2, 0.1], [9, 8, 7], 9),
    ([0.15, 0.1, 0.05], [1, 2, 3], NIL),
    ([0.22, 0.2, 0.1], [4, 5, 6], NIL),
])


def test_predict_topk_marks_weak_rows_nil():
    pred = predict_topk(VAL[0], VAL[1], threshold=-0.5, rule="max")
    assert pred.tolist() == [5, 3, 7, NIL, NIL, NIL]


def test_calibration_recovers_separating_threshold():
    t, m = calibrate_topk(*VAL, rule="max", objective="unnormalized_acc")
    assert m["unnormalized_acc"] == pytest.approx(1.0)
    assert m["nil_f1"] == pytest.approx(1.0)
    assert -0.7 <= t < -0.3


def test_evaluate_applies_val_threshold_to_test():
    r = evaluate_nil(VAL, TEST, rule="max", objective="unnormalized_acc")
    assert r["test"]["unnormalized_acc"] == pytest.approx(1.0)
    assert r["test"]["link_rate"] == pytest.approx(0.5)
    assert r["test_always_link"]["unnormalized_acc"] == pytest.approx(0.5)
    assert r["val_nil_predictions"] == 3


def test_never_nil_is_chosen_when_every_mention_is_in_kb():
    rows = _dump([
        ([0.9, 0.1], [0, 1], 0),
        ([0.2, 0.1], [2, 3], 2),
        ([0.3, 0.2], [4, 5], 4),
    ])
    r = evaluate_nil(rows, rows, rule="max", objective="unnormalized_acc")
    assert r["val_nil_predictions"] == 0
    assert r["test"]["link_rate"] == pytest.approx(1.0)


def test_margin_rule_runs_with_two_candidates():
    rows = _dump([
        ([0.9, 0.1], [0, 1], 0),
        ([0.5, 0.45], [2, 3], NIL),
    ])
    r = evaluate_nil(rows, rows, rule="margin", objective="nil_f1")
    assert r["rule"] == "margin"
    assert r["val"]["nil_precision"] == pytest.approx(1.0)


def test_load_topk_rejects_single_candidate(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, scores=np.ones((2, 1), np.float32),
             indices=np.zeros((2, 1), np.int64), gold=np.zeros(2, np.int64))
    with pytest.raises(ValueError):
        load_topk(path)


def test_load_topk_roundtrip(tmp_path):
    path = tmp_path / "ok.npz"
    np.savez(path, scores=VAL[0].astype(np.float32), indices=VAL[1], gold=VAL[2])
    s, i, g = load_topk(path)
    assert s.shape == (6, 3) and i.shape == (6, 3) and g.shape == (6,)
    assert g.tolist() == VAL[2].tolist()
