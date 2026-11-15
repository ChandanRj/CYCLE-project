"""Seed-level summary of R@1 for the five-seed protocol (CPU only).

Reads eval_results.jsonl from outputs/new_{train_year}_{config}_seed{S} for each
seed, selects the rows written by scripts/evaluate.sh with the given split and
unmapped adjustment, and calls cycle.stats.summarize_seeds on R@1 for each test
year. Also reports the epoch chosen for best.pt (metrics.jsonl "done" line).

If a run has several identical-setting rows for one year (re-evaluations), they
must agree on R@1; otherwise the script stops.

Usage:
    PYTHONPATH=src python scripts/seed_summary.py \
        --seeds 52313,52314,52315,52316,52317 --out outputs/seed_protocol/SEED_SUMMARY.md
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "src")
from cycle.stats import summarize_seeds

YEARS = (2019, 2020, 2021, 2022)


def load_r1(run_dir, split, unmapped):
    r1 = {}
    with open(os.path.join(run_dir, "eval_results.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("split") != split or d.get("unmapped_adjustment") != unmapped:
                continue
            y, v = d["test_year"], d["recall@1"]
            if y in r1 and r1[y] != v:
                raise SystemExit(f"{run_dir}: conflicting R@1 for {split} {y}: {r1[y]} vs {v}")
            r1[y] = v
    return r1


def best_epoch(run_dir):
    done = None
    with open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") == "done":
                done = d
    if done is None:
        raise SystemExit(f"{run_dir}: no 'done' line in metrics.jsonl")
    return done["best_epoch"], done["best_acc"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="52313,52314,52315,52316,52317")
    p.add_argument("--config", default="default")
    p.add_argument("--entity-set", default="new")
    p.add_argument("--train-year", type=int, default=2019)
    p.add_argument("--unmapped", default="zero", choices=["zero", "mean"])
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    runs = {s: os.path.join(args.outputs, f"{args.entity_set}_{args.train_year}_{args.config}_seed{s}")
            for s in seeds}
    data = {split: {s: load_r1(runs[s], split, args.unmapped) for s in seeds}
            for split in ("test", "validation")}
    epochs = {s: best_epoch(runs[s]) for s in seeds}

    for s in seeds:
        missing = [y for y in YEARS if y not in data["test"][s]]
        if missing:
            raise SystemExit(f"seed {s}: no test R@1 for {missing}")

    lines = [f"Runs: `{args.outputs}/{args.entity_set}_{args.train_year}_{args.config}_seed{{S}}`, "
             f"seeds {', '.join(map(str, seeds))}. evaluate.py `--unmapped-adjustment {args.unmapped}`. "
             f"Statistics: `cycle.stats.summarize_seeds` (n_boot={args.n_boot}, alpha=0.05, "
             f"percentile bootstrap over seeds, seed=0).", ""]
    lines += ["## Test R@1: seed-level summary", "",
              "| Test year | n | mean | SD | 95% bootstrap CI | range |", "|---|---|---|---|---|---|"]
    for y in YEARS:
        r = summarize_seeds([data["test"][s][y] for s in seeds], n_boot=args.n_boot)
        lines.append(f"| {y} | {r['n']} | {r['mean']:.4f} | {r['sd']:.4f} | "
                     f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['range']:.4f} |")
    for split in ("test", "validation"):
        lines += ["", f"## Per-seed R@1 ({split} split)", "",
                  "| Seed | best epoch | valid in-batch acc | " + " | ".join(map(str, YEARS)) + " |",
                  "|---" * (3 + len(YEARS)) + "|"]
        for s in seeds:
            e, a = epochs[s]
            cells = [f"{data[split][s][y]:.4f}" if y in data[split][s] else "n/a" for y in YEARS]
            lines.append(f"| {s} | {e} | {a:.5f} | " + " | ".join(cells) + " |")
    lines += ["", f"Chosen epochs (0-indexed): {[epochs[s][0] for s in seeds]}"]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# Five-seed protocol summary\n\n" + text)


if __name__ == "__main__":
    main()
