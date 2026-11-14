"""Diagnostic: flipped-rule NIL recall on "series-like" vs other NIL mentions.

CPU only, NumPy. Not a new result: label definition, rule and thresholds are
unchanged. Thresholds are read from the result JSONs written by
scripts/flipped_rule_check.py (frozen procedure in
outputs/nil_dumps/PREREG_flipped_rule.md: fitted on each year's validation
dump only, nil_f1). They are applied here to the test dump only.

Series-like rule (decided before running):
  * Names are Wikipedia titles. Gold name = `label_title` of the test mention
    (data/raw/mentions/new/{Y}/test.jsonl). The dumps store gold = -1 for NIL,
    so they carry no gold name. Top-1 name = title of the 2019 pool entity at
    dump index `indices[:, 0]`. The pool row order is taken from
    data/raw/documents/2019_test.json and checked against the 2019 qid->index
    mapping (data/raw/relationship/2019/2019_test_qid_id_mapping.txt).
  * Normalise a name: lowercase, split on "_", then
      - drop the FIRST token if it is a year:  ^\\d{4}$
      - drop EVERY season token:               ^\\d{4}[-–]\\d{2,4}$  (e.g. 2018–19, 2019-20)
    Other tokens, including years that are not leading, are kept as they are.
  * A NIL mention is series-like if the normalised gold name equals the
    normalised top-1 name exactly (same token sequence) and is non-empty.
    The same normalisation is applied to both names, otherwise the top-1's own
    year would prevent any match.

Recall per group = share of that group's NIL mentions that the flipped rule
(NIL if top-1 score >= threshold) predicts as NIL.

Usage:
    PYTHONPATH=src python scripts/nil_subset_check.py
"""

import json
import re
import sys

import numpy as np

sys.path.insert(0, "src")
from cycle.data.graphs import _read_qid_mapping
from cycle.nil import NIL
from cycle.nil_eval import load_topk

YEAR = re.compile(r"^\d{4}$")
SEASON = re.compile(r"^\d{4}[-–]\d{2,4}$")
SEEDS = {52313: "outputs/nil_dumps", 52314: "outputs/nil_dumps_seed52314"}


def normalise(title):
    toks = [t for t in title.lower().split("_") if t]
    if toks and YEAR.match(toks[0]):
        toks = toks[1:]
    return tuple(t for t in toks if not SEASON.match(t))


def series_like(gold_title, top1_title):
    g = normalise(gold_title)
    return bool(g) and g == normalise(top1_title)


def main():
    docs = [json.loads(l) for l in open("data/raw/documents/2019_test.json", encoding="utf-8")]
    kb = _read_qid_mapping("data/raw/relationship/2019/2019_test_qid_id_mapping.txt")
    assert all(kb[d["qid"]] == i for i, d in enumerate(docs)), "pool order != qid mapping"
    pool_title = [d["title"] for d in docs]

    rows = []
    for seed, ddir in SEEDS.items():
        for y in (2020, 2021, 2022):
            res = json.load(open(f"{ddir}/result_y{y}_flipped_nil_f1.json"))
            t = res["threshold"]
            s, i, g = load_topk(f"{ddir}/default_kb2019_y{y}_test.npz")
            titles = [json.loads(l)["label_title"]
                      for l in open(f"data/raw/mentions/new/{y}/test.jsonl", encoding="utf-8")]
            assert len(titles) == g.size
            nil = np.where(g == NIL)[0]
            pred_nil = s[:, 0] >= t
            # sanity: overall NIL recall must equal the frozen-procedure JSON
            rec_all = pred_nil[nil].mean()
            assert abs(rec_all - res["test"]["nil_recall"]) < 1e-12, (seed, y)
            sl = np.array([series_like(titles[k], pool_title[i[k, 0]]) for k in nil])
            a, b = nil[sl], nil[~sl]
            rows.append(dict(seed=seed, year=y, threshold=t, n_nil=int(nil.size),
                             n_series=int(a.size), n_rest=int(b.size),
                             recall_series=float(pred_nil[a].mean()) if a.size else float("nan"),
                             recall_rest=float(pred_nil[b].mean()) if b.size else float("nan"),
                             recall_all=float(rec_all),
                             examples=[(titles[k], pool_title[i[k, 0]]) for k in a[:3]]))
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
