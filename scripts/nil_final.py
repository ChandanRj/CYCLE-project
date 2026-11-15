"""Final NIL evaluation: Label A / Label B x max / flipped rule x two seeds (CPU, NumPy).

Procedure pre-registered in outputs/nil_dumps/PREREG_FINAL.md (written before
any data was loaded), with PREREG_FINAL_AMENDMENT_1.md (QID sanity check). Threshold fitted on each year's validation dump only,
objective nil_f1, candidates = unique validation top-1 scores subsampled to 512
quantiles plus a never-NIL value, applied once to the same year's test dump.

  flipped: NIL if top1 >= t; never-NIL = +inf; ties keep largest t
           (identical to scripts/flipped_rule_check.py)
  max:     NIL if top1 <  t; never-NIL = -inf; ties keep smallest t
  Label A: dump gold (NIL = gold entity not in the 2019 KB)
  Label B: Label A with series-like NIL mentions (scripts/nil_subset_check.py
           rule, imported unchanged) removed from the evaluation set

Writes outputs/nil_dumps/final/{label}_{rule}_seed{S}_y{Y}.json and prints one
JSON row per cell. Existing result files are only read.

Usage:
    PYTHONPATH=src python scripts/nil_final.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from cycle.data.graphs import _read_qid_mapping
from cycle.nil import NIL, metrics
from cycle.nil_eval import load_topk
from nil_subset_check import series_like          # rule used unchanged

SEEDS = {52313: "outputs/nil_dumps",
         52314: "/DATA/kmpooja/cycle_outputs/nil_dumps_seed52314"}
YEARS = (2020, 2021, 2022)
OUT = "outputs/nil_dumps/final"


def candidates(top):
    uniq = np.unique(top)
    if uniq.size > 512:
        uniq = np.unique(np.quantile(uniq, np.linspace(0.0, 1.0, 512)))
    return uniq


def is_nil(top, t, rule):
    return top >= t if rule == "flipped" else top < t


def fit(s, i, g, rule):
    top = s[:, 0]
    uniq = candidates(top)
    best_t, best_f1, best_m = None, -np.inf, None
    if rule == "flipped":
        cands = np.concatenate([uniq, [np.inf]])          # ascending; >= keeps largest t
        for t in cands:
            m = metrics(np.where(top >= t, NIL, i[:, 0]), g)
            if np.isnan(m["nil_f1"]):
                continue
            if m["nil_f1"] >= best_f1:
                best_t, best_f1, best_m = float(t), m["nil_f1"], m
    else:
        cands = np.concatenate([[-np.inf], uniq])         # ascending; > keeps smallest t
        for t in cands:
            m = metrics(np.where(top < t, NIL, i[:, 0]), g)
            if np.isnan(m["nil_f1"]):
                continue
            if m["nil_f1"] > best_f1:
                best_t, best_f1, best_m = float(t), m["nil_f1"], m
    return best_t, best_m, int(cands.size)


def load_split(ddir, y, split, pool_title, pool_qid):
    s, i, g = load_topk(f"{ddir}/default_kb2019_y{y}_{split}.npz")
    titles = [json.loads(l)["label_title"]
              for l in open(f"data/raw/mentions/new/{y}/{split}.jsonl", encoding="utf-8")]
    if len(titles) != g.size:
        raise SystemExit(f"STOP: {ddir} {y} {split}: {len(titles)} titles vs {g.size} rows")
    qids = [json.loads(l)["qid"]
            for l in open(f"data/raw/mentions/new/{y}/{split}.jsonl", encoding="utf-8")]
    inkb = np.where(g != NIL)[0]
    # Amendment 1 (PREREG_FINAL_AMENDMENT_1.md): QID check, not title (titles were renamed)
    bad = [k for k in inkb if str(pool_qid[g[k]]) != str(qids[k])]
    if bad:
        raise SystemExit(f"STOP: {ddir} {y} {split}: {len(bad)} in-KB rows with pool QID != qid")
    nil = np.where(g == NIL)[0]
    sl = np.zeros(g.size, bool)
    sl[nil] = [series_like(titles[k], pool_title[i[k, 0]]) for k in nil]
    return s, i, g, sl


def baselines(i, g):
    p = float((g == NIL).mean())
    al = metrics(i[:, 0], g)
    return {"always_link_unnormalized_acc": al["unnormalized_acc"], "always_link_nil_f1": al["nil_f1"],
            "all_nil_precision": p, "all_nil_recall": 1.0, "all_nil_f1": 2 * p / (1 + p),
            "all_nil_unnormalized_acc": p}


def main():
    docs = [json.loads(l) for l in open("data/raw/documents/2019_test.json", encoding="utf-8")]
    kb = _read_qid_mapping("data/raw/relationship/2019/2019_test_qid_id_mapping.txt")
    assert all(kb[d["qid"]] == k for k, d in enumerate(docs)), "pool order != qid mapping"
    pool_title = [d["title"] for d in docs]
    pool_qid = [d["qid"] for d in docs]
    os.makedirs(OUT, exist_ok=True)

    for seed, ddir in SEEDS.items():
        for y in YEARS:
            val = load_split(ddir, y, "validation", pool_title, pool_qid)
            test = load_split(ddir, y, "test", pool_title, pool_qid)
            for label in ("A", "B"):
                def sub(d):
                    s, i, g, sl = d
                    keep = np.ones(g.size, bool) if label == "A" else ~sl
                    return s[keep], i[keep], g[keep], int((~keep).sum())
                vs, vi, vg, v_rm = sub(val)
                ts, ti, tg, t_rm = sub(test)
                for rule in ("max", "flipped"):
                    t, vm, nc = fit(vs, vi, vg, rule)
                    tn = is_nil(ts[:, 0], t, rule)
                    tm = metrics(np.where(tn, NIL, ti[:, 0]), tg)
                    row = {"label": label, "rule": rule, "seed": seed, "year": y, "threshold": t,
                           "n_candidates": nc, "val_removed": v_rm, "test_removed": t_rm,
                           "val_nil_predictions": int(is_nil(vs[:, 0], t, rule).sum()),
                           "val": vm, "test_nil_predictions": int(tn.sum()), "test": tm,
                           "test_baselines": baselines(ti, tg)}
                    if label == "A" and rule == "flipped":     # must reproduce the frozen run
                        ref = json.load(open(f"{ddir}/result_y{y}_flipped_nil_f1.json"))
                        if ref["threshold"] != t or any(
                                abs(ref["test"][k] - tm[k]) > 1e-12 for k in
                                ("nil_precision", "nil_recall", "nil_f1", "unnormalized_acc")):
                            raise SystemExit(f"STOP: Label A flipped does not reproduce {ddir} {y}")
                        row["reproduces_frozen_result"] = True
                    json.dump(row, open(f"{OUT}/{label}_{rule}_seed{seed}_y{y}.json", "w"),
                              indent=2, default=float)
                    print(json.dumps(row, default=float))


if __name__ == "__main__":
    main()
