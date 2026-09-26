"""Diagnose cross-year negatives (Eq. 11) for every consecutive year pair.

Shows that removed relation edges always touch an entity absent from the
next year's train graph, so the strict N_i^r set is empty for every pair.

Usage (needs prepared data):
    PYTHONPATH=src python scripts/diag_negatives.py
"""
import os
import numpy as np
from cycle.data.graphs import _read_qid_mapping, _read_edges, _undirected_qid_set

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REL = "data/raw/relationship"


def load(y):
    m = _read_qid_mapping(f"{REL}/{y}/{y}_train_qid_id_mapping.txt")
    raw = _read_edges(f"{REL}/{y}/{y}_train_qid_relation.txt")
    return m, raw, _undirected_qid_set(raw)


for y in range(2014, 2023):
    m1, raw1, E1 = load(y - 1)
    m2, raw2, E2 = load(y)
    directed1 = set(raw1)
    rev_dup = sum(1 for a, b in raw1 if (b, a) in directed1 and a != b)
    added, removed = E2 - E1, E1 - E2
    nodes1, nodes2 = set(m1), set(m2)
    # edge endpoints vs the train mapping
    ep2_out = sum(1 for p in E2 if not p <= nodes2)
    # removed edges: why they vanish
    both_in2 = [p for p in removed if p <= nodes2]
    one_out = [p for p in removed if len(p & nodes2) == 1]
    both_out = [p for p in removed if not (p & nodes2)]
    added_kept = [p for p in added if p <= nodes2]
    # id alignment: same QID → same id across years?
    common = nodes1 & nodes2
    same_id = sum(1 for q in common if m1[q] == m2[q])
    # same-year id file consistency with qid file
    ids2 = np.loadtxt(f"{REL}/{y}/{y}_train_id_relation.txt", dtype=np.int64)
    consistent = all(
        (m2.get(a), m2.get(b)) == (int(i), int(j))
        for (a, b), (i, j) in zip(raw2, ids2)
    )
    # processed npz
    z = np.load(f"data/processed/graphs/{y}.npz")
    print(
        f"{y-1}->{y}: |E1|={len(E1)} |E2|={len(E2)} |E1&E2|={len(E1 & E2)} "
        f"added={len(added)} (kept {len(added_kept)}) removed={len(removed)} "
        f"[both endpoints in {y} map: {len(both_in2)}, one: {len(one_out)}, none: {len(both_out)}] | "
        f"nodes {len(nodes1)}/{len(nodes2)} common={len(common)} same_id={same_id} | "
        f"reverse-dup edges in {y-1} file={rev_dup} | E2 edges w/ endpoint outside map={ep2_out} | "
        f"id/qid files consistent={consistent} | npz pos={len(z['pos_flat'])} neg={len(z['neg_flat'])}"
    )
