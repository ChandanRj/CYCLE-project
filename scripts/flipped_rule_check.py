"""Pre-registered flipped-max NIL rule check (CPU only, NumPy). POST HOC rule.

Procedure frozen in outputs/nil_dumps/PREREG_flipped_rule.md:
  * rule: predict NIL if top-1 score >= threshold, otherwise link to top-1
  * threshold fitted on the year's validation dump only, objective nil_f1
  * candidates: unique validation top-1 scores, subsampled to 512 quantiles
    (as in cycle.nil_eval.calibrate_topk), plus +inf (never NIL)
  * ties keep the largest threshold (fewest NIL predictions)
  * applied once to the same year's test dump

Copied unchanged from the scratch script used for 2020; only the year is now
an argument (default 2020) instead of hard-coded.

Usage:
    PYTHONPATH=src python scripts/flipped_rule_check.py --year 2021
"""

import argparse
import json
import sys

import numpy as np
sys.path.insert(0, "src")
from cycle.nil import NIL, metrics
from cycle.nil_eval import load_topk

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--year", type=int, default=2020)
p.add_argument("--dump-dir", default="outputs/nil_dumps")
args = p.parse_args()
Y = args.year

D = args.dump_dir
vs, vi, vg = load_topk(f"{D}/default_kb2019_y{Y}_validation.npz")
ts, ti, tg = load_topk(f"{D}/default_kb2019_y{Y}_test.npz")

def pred(s, i, t):
    return np.where(s[:, 0] >= t, NIL, i[:, 0])

top = vs[:, 0]
uniq = np.unique(top)
if uniq.size > 512:
    uniq = np.unique(np.quantile(uniq, np.linspace(0.0, 1.0, 512)))
cands = np.concatenate([uniq, [np.inf]])

best_t, best_f1, best_m = None, -np.inf, None
for t in cands:                         # ascending; ">=" keeps the largest t on ties
    m = metrics(pred(vs, vi, t), vg)
    if np.isnan(m["nil_f1"]):
        continue
    if m["nil_f1"] >= best_f1:
        best_t, best_f1, best_m = float(t), m["nil_f1"], m
val_npred = int((vs[:, 0] >= best_t).sum())
test_m = metrics(pred(ts, ti, best_t), tg)
test_npred = int((ts[:, 0] >= best_t).sum())
out = {"rule": "NIL if top1 >= threshold (flipped max, post hoc)", "objective": "nil_f1",
       "threshold": best_t, "n_candidates": int(cands.size),
       "val_nil_predictions": val_npred, "val": best_m,
       "test_nil_predictions": test_npred, "test": test_m,
       "test_always_link": metrics(ti[:, 0], tg)}
json.dump(out, open(f"{D}/result_y{Y}_flipped_nil_f1.json", "w"), indent=2, default=float)
print(f"year={Y} threshold={best_t!r}  candidates={cands.size}")
for name, m, n in (("val", best_m, val_npred), ("test", test_m, test_npred)):
    print(f"{name}: NIL_pred={n}/{m['num_mentions']} gold_NIL={m['num_nil_gold']} "
          + " ".join(f"{k}={m[k]:.4f}" for k in ("nil_precision", "nil_recall", "nil_f1", "unnormalized_acc", "inkb_acc", "link_rate")))
a = out["test_always_link"]; print(f"test always-link: unnormalized_acc={a['unnormalized_acc']:.4f}; all-NIL F1={2*0.0301/(1.0301):.4f}")
