"""NIL (out-of-KB) prediction for entity linking.

A mention is NIL when its gold entity is *not* in the knowledge base (the
candidate pool the linker searches). CYCLE and TIGER never produce NIL: every
gold label is a pool index, so a mention whose entity is absent from the pool
is simply counted as a miss. This module adds the missing decision step.

Everything here is NumPy-only, so it can be unit-tested without a GPU and
called from evaluate.py on bi-encoder score matrices (mentions x candidates).

Conventions
-----------
* ``gold`` / ``pred`` are int arrays of pool indices; ``NIL`` (-1) marks a
  mention with no in-KB entity.
* ``scores`` is a (num_mentions, num_candidates) float matrix of linker scores,
  higher = more likely.
* A decision rule maps scores to a *NIL score* where higher means "more likely
  NIL". A mention is predicted NIL when its NIL score exceeds a threshold.
"""

from __future__ import annotations

import numpy as np

NIL = -1

NIL_RULES = ("max", "margin")
OBJECTIVES = ("unnormalized_acc", "nil_f1", "inkb_acc")


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------

def nil_score(scores, rule="max"):
    """Per-mention NIL score (higher = more likely NIL).

    rule="max":    -top1 score. A mention whose best candidate scores low is
                   likely to have no entity in the KB.
    rule="margin": -(top1 - top2). Small margins are ambiguity, not absence,
                   so this rule is a weaker, control-style baseline. It needs
                   at least two candidates.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] == 0 or s.shape[1] == 0:
        raise ValueError("scores must be a non-empty (mentions x candidates) matrix")
    if rule == "max":
        return -s.max(axis=1)
    if rule == "margin":
        if s.shape[1] < 2:
            raise ValueError("margin rule needs at least two candidates")
        top2 = np.partition(s, -2, axis=1)[:, -2]
        return -(s.max(axis=1) - top2)
    raise ValueError(f"unknown NIL rule {rule!r}; expected one of {NIL_RULES}")


def predict(scores, threshold, rule="max"):
    """Top-1 pool index, or NIL where the NIL score exceeds ``threshold``."""
    s = np.asarray(scores, dtype=np.float64)
    top = s.argmax(axis=1)
    is_nil = nil_score(s, rule) > threshold
    return np.where(is_nil, NIL, top)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(pred, gold):
    """Standard NIL-aware EL metrics.

    inkb_acc:           top-1 accuracy over mentions whose gold is in the KB.
                        A NIL prediction for them counts as wrong.
    unnormalized_acc:   accuracy over all mentions; a NIL prediction is correct
                        when the gold is NIL (the usual "unnormalized" score).
    nil_precision/recall/f1: how well NIL predictions match gold NIL mentions.
                        Undefined counts are returned as NaN.
    link_rate:          fraction of mentions given an in-KB prediction.
    """
    pred = np.asarray(pred)
    gold = np.asarray(gold)
    if pred.shape != gold.shape or pred.ndim != 1 or pred.size == 0:
        raise ValueError("pred and gold must be equal-length, non-empty 1-D arrays")

    in_kb = gold != NIL
    gold_nil = ~in_kb
    pred_nil = pred == NIL
    correct = pred == gold

    tp = int(np.sum(pred_nil & gold_nil))
    fp = int(np.sum(pred_nil & in_kb))
    fn = int(np.sum(~pred_nil & gold_nil))

    # No NIL predictions -> precision is taken as 0 (nothing correct was said).
    # No gold NIL mentions -> recall, and therefore F1, is undefined (NaN).
    nil_precision = tp / (tp + fp) if tp + fp else 0.0
    if gold_nil.any():
        nil_recall = tp / (tp + fn)
        nil_f1 = (
            0.0 if nil_precision + nil_recall == 0
            else 2 * nil_precision * nil_recall / (nil_precision + nil_recall)
        )
    else:
        nil_recall = float("nan")
        nil_f1 = float("nan")

    return {
        "inkb_acc": float(correct[in_kb].mean()) if in_kb.any() else float("nan"),
        "unnormalized_acc": float(correct.mean()),
        "nil_precision": float(nil_precision),
        "nil_recall": float(nil_recall),
        "nil_f1": float(nil_f1),
        "link_rate": float((~pred_nil).mean()),
        "num_mentions": int(gold.size),
        "num_nil_gold": int(gold_nil.sum()),
    }


# ---------------------------------------------------------------------------
# Threshold calibration (fit on validation, apply on test)
# ---------------------------------------------------------------------------

def calibrate_threshold(scores, gold, rule="max", objective="unnormalized_acc",
                        max_candidates=512):
    """Choose the NIL threshold that maximises ``objective`` on validation data.

    Returns (threshold, metrics_at_threshold). Thresholds are taken from the
    observed NIL scores, subsampled to at most ``max_candidates`` values, plus
    +inf (never predict NIL) so "always link" is always an option.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")
    s = np.asarray(scores, dtype=np.float64)
    gold = np.asarray(gold)
    nil = nil_score(s, rule)
    top = s.argmax(axis=1)

    uniq = np.unique(nil)
    if uniq.size > max_candidates:
        uniq = np.quantile(uniq, np.linspace(0.0, 1.0, max_candidates))
        uniq = np.unique(uniq)
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
        raise ValueError("no valid threshold: objective undefined for every threshold")
    return best_t, best_m


# ---------------------------------------------------------------------------
# Building NIL labels
# ---------------------------------------------------------------------------

def label_against_kb(mention_qids, kb_qid_to_index):
    """Map each mention's gold QID to its KB pool index, or NIL if absent.

    This is the *temporal* NIL setting: ``kb_qid_to_index`` is the entity pool
    of an earlier snapshot, and ``mention_qids`` come from a later test year.
    Mentions of entities created after the KB snapshot become NIL.
    """
    return np.array(
        [kb_qid_to_index.get(q, NIL) for q in mention_qids], dtype=np.int64
    )


def hold_out_kb(pool_qids, mention_qids, fraction, seed=0):
    """Synthetic NIL setting: remove a fraction of entities from the KB.

    Returns (kb_qid_to_index, gold) where ``kb_qid_to_index`` covers the
    remaining entities (re-indexed 0..m-1) and ``gold`` marks every mention of
    a removed entity as NIL. Use the *full* pool for scoring, then this KB for
    the held-out-entity analysis, so removed entities cannot be predicted.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    pool_qids = list(pool_qids)
    n_remove = max(1, int(round(fraction * len(pool_qids))))
    removed = set(rng.choice(len(pool_qids), size=n_remove, replace=False).tolist())
    kb_qids = [q for i, q in enumerate(pool_qids) if i not in removed]
    kb_qid_to_index = {q: i for i, q in enumerate(kb_qids)}
    removed_qids = {pool_qids[i] for i in removed}
    gold = np.array(
        [NIL if q in removed_qids else kb_qid_to_index[q] for q in mention_qids],
        dtype=np.int64,
    )
    return kb_qid_to_index, gold
