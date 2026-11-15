"""Dump top-k bi-encoder candidate scores for NIL evaluation (GPU side).

Scores each mention of ``--year``/``--split`` against the entity pool of
``--kb-year``/``--kb-split``, keeps the top-k candidates, and maps each gold
entity to the KB by QID. A gold entity that is not in the KB becomes NIL, so
the same script covers:

  * same-year eval:       --kb-year 2019 --year 2019   (no NIL expected)
  * temporal NIL:         --kb-year 2019 --year 2020   (entities created after 2019 are NIL)

Scoring is bi-encoder only (dot product of mention and entity encodings),
without the candidate-side graph fusion that evaluate.py applies for
candidate-mode runs. Dumps from this script are therefore NOT numerically
identical to evaluate.py recall for candidate-fused runs. Say so in any table.

Usage:
    PYTHONPATH=src python scripts/dump_topk.py --run-dir outputs/<run> \
        --kb-year 2019 --year 2020 --split test --k 64 \
        --out outputs/nil_dumps/<run>_kb2019_y2020_test.npz

Output (.npz): scores float32 (n,k), indices int64 (n,k), gold int64 (n,)
(gold = KB pool index, or -1 for NIL), plus meta fields.
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from cycle.config import Config, apply_overrides
from cycle.data.graphs import _read_qid_mapping
from cycle.data.mentions import load_mention_dataset
from cycle.evaluate import encode_pool
from cycle.data.documents import load_candidate_pool
from cycle.models.biencoder import BiEncoder
from cycle.utils import configure_tf32, resolve_device, set_all_seeds

NIL = -1


def _invert(qid_to_index):
    return {idx: qid for qid, idx in qid_to_index.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--kb-year", required=True, type=int)
    p.add_argument("--kb-split", default="test", choices=["test", "validation"])
    p.add_argument("--year", required=True, type=int)
    p.add_argument("--split", required=True, choices=["test", "validation"])
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.k < 2:
        raise SystemExit("--k must be >= 2 (margin rule needs top-2)")

    with open(os.path.join(args.run_dir, "config_used.json"), encoding="utf-8") as f:
        cfg = Config.wrap(apply_overrides(json.load(f), [], must_exist=True))
    configure_tf32(cfg.run.tf32)
    set_all_seeds(cfg.run.seed)
    device = resolve_device()
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.run.amp)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.bert_model,
                                              do_lower_case=cfg.model.lowercase)
    biencoder = BiEncoder(cfg.model.bert_model).to(device)
    ckpt = torch.load(os.path.join(args.run_dir, args.checkpoint),
                      map_location=device, weights_only=True)
    biencoder.load_state_dict(ckpt["biencoder"])
    biencoder.eval()

    # KB pool (entities the model may output) and its QID -> index map.
    kb_pool_ids, kb_n = load_candidate_pool(cfg, tokenizer, args.kb_year, args.kb_split)
    kb_map = _read_qid_mapping(os.path.join(
        cfg.data.root, "raw", "relationship", str(args.kb_year),
        f"{args.kb_year}_{args.kb_split}_qid_id_mapping.txt"))
    if len(kb_map) != kb_n:
        raise SystemExit(f"KB map has {len(kb_map)} QIDs but pool has {kb_n} entities")
    kb_encs = encode_pool(biencoder, kb_pool_ids, device,
                          cfg.eval.encode_batch_size, amp_dtype)

    # Eval mentions and their gold QIDs (via the eval year's own pool map).
    dataset = load_mention_dataset(cfg, tokenizer, cfg.data.entity_set,
                                   args.year, args.split, args.limit)
    eval_map = _read_qid_mapping(os.path.join(
        cfg.data.root, "raw", "relationship", str(args.year),
        f"{args.year}_{args.split}_qid_id_mapping.txt"))
    eval_id_to_qid = _invert(eval_map)
    label_all = dataset.tensors[2].tolist()
    gold_qids = [eval_id_to_qid[int(i)] for i in label_all]
    gold = np.array([kb_map.get(q, NIL) for q in gold_qids], dtype=np.int64)

    # Score mentions against the KB pool, batch by batch, keep top-k.
    total = dataset.tensors[0].size(0)
    all_scores, all_idx = [], []
    with torch.no_grad():
        for s in tqdm(range(0, total, cfg.eval.batch_size), desc="score mentions"):
            ctx = dataset.tensors[0][s: s + cfg.eval.batch_size].to(device)
            if amp_dtype is not None and device.type == "cuda":
                with torch.autocast("cuda", dtype=amp_dtype):
                    enc = biencoder.encode_context(ctx).float()
            else:
                enc = biencoder.encode_context(ctx).float()
            sc = enc @ kb_encs.t()
            v, i = sc.topk(args.k, dim=1)
            all_scores.append(v.cpu())
            all_idx.append(i.cpu())

    scores = torch.cat(all_scores).numpy().astype(np.float32)
    indices = torch.cat(all_idx).numpy().astype(np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, scores=scores, indices=indices, gold=gold,
             kb_year=args.kb_year, year=args.year, split=args.split,
             run_dir=args.run_dir, k=args.k)
    n_nil = int((gold == NIL).sum())
    print(f"wrote {args.out}: {total} mentions, k={args.k}, "
          f"gold NIL={n_nil} ({n_nil / total:.3f})")


if __name__ == "__main__":
    main()
