"""Hand-computed checks for cycle.nil (NumPy only; runs without a GPU)."""

import math

import numpy as np
import pytest

from cycle.nil import (NIL, calibrate_threshold, hold_out_kb, label_against_kb,
                       metrics, nil_score, predict)

# 5 mentions x 3 candidates. Rows 2 and 3 are weak matches (likely NIL).
SCORES = np.array([
    [0.9, 0.1, 0.0],   # gold 0, confident
    [0.2, 0.8, 0.1],   # gold 1, confident
    [0.1, 0.2, 0.15],  # gold NIL, best score 0.2
    [0.3, 0.1, 0.2],   # gold NIL, best score 0.3
    [0.1, 0.2, 0.7],   # gold 2, confident
])
GOLD = np.array([0, 1, NIL, NIL, 2])


def test_nil_score_max_is_negative_top1():
    assert np.allclose(nil_score(SCORES, "max"), [-0.9, -0.8, -0.2, -0.3, -0.7])


def test_nil_score_margin_is_negative_gap():
    expected = -np.array([0.9 - 0.1, 0.8 - 0.2, 0.2 - 0.15, 0.3 - 0.2, 0.7 - 0.2])
    assert np.allclose(nil_score(SCORES, "margin"), expected)


def test_margin_needs_two_candidates():
    with pytest.raises(ValueError):
        nil_score(np.array([[0.5], [0.6]]), "margin")


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        nil_score(SCORES, "entropy")


def test_predict_threshold_applies_nil():
    pred = predict(SCORES, threshold=-0.25, rule="max")
    assert pred.tolist() == [0, 1, NIL, 0, 2]


def test_metrics_hand_computed():
    pred = np.array([0, 1, NIL, 0, 2])
    m = metrics(pred, GOLD)
    assert m["inkb_acc"] == pytest.approx(1.0)          # rows 0, 1, 4 correct
    assert m["unnormalized_acc"] == pytest.approx(0.8)  # row 3 is a wrong link
    assert m["nil_precision"] == pytest.approx(1.0)     # row 2 correct NIL
    assert m["nil_recall"] == pytest.approx(0.5)        # row 3 missed
    assert m["nil_f1"] == pytest.approx(2 / 3)
    assert m["link_rate"] == pytest.approx(0.8)
    assert m["num_nil_gold"] == 2


def test_metrics_no_predicted_nil_has_zero_precision():
    m = metrics(np.array([0, 1, 0, 0, 2]), GOLD)
    assert m["nil_precision"] == 0.0 and m["nil_recall"] == 0.0 and m["nil_f1"] == 0.0


def test_metrics_no_gold_nil_leaves_nil_f1_undefined():
    m = metrics(np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert math.isnan(m["nil_f1"]) and math.isnan(m["nil_recall"])
    assert m["unnormalized_acc"] == 1.0


def test_calibration_recovers_perfect_threshold():
    t, m = calibrate_threshold(SCORES, GOLD, rule="max",
                               objective="unnormalized_acc")
    assert m["unnormalized_acc"] == pytest.approx(1.0)
    assert predict(SCORES, t, "max").tolist() == GOLD.tolist()


def test_calibration_never_nil_option_is_available():
    # With only in-KB mentions, the best choice is to never predict NIL.
    scores = SCORES[[0, 1, 4]]
    gold = GOLD[[0, 1, 4]]
    t, m = calibrate_threshold(scores, gold, objective="unnormalized_acc")
    assert m["link_rate"] == pytest.approx(1.0)


def test_calibration_rejects_unknown_objective():
    with pytest.raises(ValueError):
        calibrate_threshold(SCORES, GOLD, objective="f2")


def test_label_against_kb_marks_absent_entities_nil():
    kb = {"Q1": 0, "Q2": 1}
    labels = label_against_kb(["Q2", "Q9", "Q1"], kb)
    assert labels.tolist() == [1, NIL, 0]


def test_hold_out_kb_removes_fraction_and_marks_their_mentions():
    pool = ["Q1", "Q2", "Q3", "Q4"]
    mentions = ["Q1", "Q2", "Q3", "Q4", "Q1", "Q4"]
    kb, gold = hold_out_kb(pool, mentions, fraction=0.5, seed=0)
    assert len(kb) == 2
    removed = set(pool) - set(kb)
    expected_nil = sum(q in removed for q in mentions)
    assert int((gold == NIL).sum()) == expected_nil
    for q, g in zip(mentions, gold):
        if q in kb:
            assert g == kb[q]


def test_hold_out_kb_is_deterministic_for_seed():
    pool = [f"Q{i}" for i in range(20)]
    a, _ = hold_out_kb(pool, pool, 0.3, seed=7)
    b, _ = hold_out_kb(pool, pool, 0.3, seed=7)
    assert a == b
