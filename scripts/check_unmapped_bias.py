"""Candidate-mode check: how often is the top-1 prediction a pool entity with
no node in the train-year graph (an "unmapped" entity)?

Compares that rate with the fair share (unmapped / pool size) and with a
no-adjustment control (text-only scores from the same checkpoint). Uses
cycle.evaluate.enhance_pool, so it measures exactly what evaluation does
under the chosen --unmapped-adjustment.

Usage (candidate-mode run dir):
    PYTHONPATH=src python scripts/check_unmapped_bias.py \
        --run-dir outputs/new_2019_paper_candidate_seed52313 \
        --test-years 2019,2020 --unmapped-adjustment zero
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from cycle.config import Config
from cycle.data.documents import load_candidate_pool
from cycle.data.graphs import _read_qid_mapping
from cycle.data.mentions import load_mention_dataset
from cycle.evaluate import encode_pool, enhance_pool, load_candidate_enhancer
from cycle.models.biencoder import BiEncoder
from cycle.utils import configure_tf32, resolve_device, set_all_seeds

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", required=True)
parser.add_argument("--checkpoint", default="best.pt")
parser.add_argument("--test-years", required=True)
parser.add_argument("--unmapped-adjustment", default="zero", choices=["zero", "mean"])
args = parser.parse_args()

with open(os.path.join(args.run_dir, "config_used.json"), encoding="utf-8") as f:
    cfg = Config.wrap(json.load(f))
configure_tf32(cfg.run.tf32)
set_all_seeds(cfg.run.seed)
device = resolve_device()
amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.run.amp)
tok = AutoTokenizer.from_pretrained(cfg.model.bert_model, do_lower_case=cfg.model.lowercase)
bi = BiEncoder(cfg.model.bert_model).to(device)
ckpt = torch.load(os.path.join(args.run_dir, args.checkpoint),
                  map_location=device, weights_only=True)
bi.load_state_dict(ckpt["biencoder"])
bi.eval()
enhancer, train_map = load_candidate_enhancer(cfg, ckpt, device)
if enhancer is None:
    raise SystemExit("not a candidate-mode checkpoint (fusion.mode != candidate)")

for year in args.test_years.split(","):
    year = year.strip()
    pool_ids, n_ent = load_candidate_pool(cfg, tok, year, "test")
    raw = encode_pool(bi, pool_ids, device, cfg.eval.encode_batch_size, amp)
    enh, _ = enhance_pool(raw, enhancer, train_map, cfg, year, "test", device,
                          args.unmapped_adjustment)
    test_map = _read_qid_mapping(os.path.join(
        cfg.data.root, "raw", "relationship", year, f"{year}_test_qid_id_mapping.txt"))
    unmapped = torch.zeros(n_ent, dtype=torch.bool, device=device)
    for qid, i in test_map.items():
        if qid not in train_map:
            unmapped[i] = True

    ds = load_mention_dataset(cfg, tok, cfg.data.entity_set, year, "test", None)
    with torch.no_grad(), torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        ctx = torch.cat([bi.encode_context(c.to(device)).float()
                         for c in ds.tensors[0].split(cfg.eval.batch_size)])
    gold = ds.tensors[2].to(device)
    delta = enh - raw                               # per-entity adjustment
    top1, top1_raw, shift = [], [], []
    for s in range(0, ctx.size(0), 1024):
        c = ctx[s:s + 1024]
        top1.append((c @ enh.t()).argmax(1))
        top1_raw.append((c @ raw.t()).argmax(1))
        shift.append((c @ delta[~unmapped].t()).mean(1))
    top1, top1_raw = torch.cat(top1), torch.cat(top1_raw)
    correct = top1 == gold
    gold_un = unmapped[gold]
    rate = lambda m: round(m.float().mean().item(), 4) if m.numel() else None
    print(json.dumps({
        "run": os.path.basename(os.path.normpath(args.run_dir)), "year": int(year),
        "unmapped_adjustment": args.unmapped_adjustment,
        "mentions": int(gold.numel()), "unmapped_entities": int(unmapped.sum()),
        "fair_share": rate(unmapped),
        "top1_unmapped_rate": rate(unmapped[top1]),
        "top1_unmapped_rate_no_adjustment": rate(unmapped[top1_raw]),
        "gold_unmapped_rate": rate(gold_un),
        "wrong_top1_unmapped_rate": rate(unmapped[top1[~correct]]),
        "recall@1": rate(correct),
        "recall@1_gold_mapped": rate(correct[~gold_un]),
        "mean_score_shift_mapped": round(torch.cat(shift).mean().item(), 3),
    }), flush=True)
