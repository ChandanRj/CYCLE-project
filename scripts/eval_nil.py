"""Run NIL evaluation on top-k dumps (CPU only, NumPy).

Threshold is chosen on the validation dump and applied to the test dump.

Usage:
    PYTHONPATH=src python scripts/eval_nil.py \
        --val outputs/nil_dumps/<run>_kb2019_y2019_validation.npz \
        --test outputs/nil_dumps/<run>_kb2019_y2020_test.npz \
        --rule max --objective unnormalized_acc
"""

import argparse
import json

from cycle.nil_eval import evaluate_nil, load_topk


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--rule", default="max", choices=["max", "margin"])
    p.add_argument("--objective", default="unnormalized_acc",
                   choices=["unnormalized_acc", "nil_f1", "inkb_acc"])
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    r = evaluate_nil(load_topk(args.val), load_topk(args.test),
                     rule=args.rule, objective=args.objective)
    t, a = r["test"], r["test_always_link"]
    print(f"rule={r['rule']} objective={r['objective']} threshold={r['threshold']:.4f} "
          f"val_nil_predictions={r['val_nil_predictions']}")
    print("| metric | NIL-aware (test) | always link (test) |")
    print("|---|---|---|")
    for k in ["unnormalized_acc", "inkb_acc", "nil_precision", "nil_recall", "nil_f1", "link_rate"]:
        print(f"| {k} | {t[k]:.4f} | {a[k]:.4f} |")
    print(f"test mentions={t['num_mentions']}, gold NIL={t['num_nil_gold']}")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, default=float)


if __name__ == "__main__":
    main()
