"""Measure directional collapse of the paper graph module's embeddings.

For z_r and z_f over all train-year graph nodes: pairwise cosine (200k random
pairs), share of energy in the top singular direction, spread around the mean
embedding (centered Frobenius norm / (sqrt(n) * |mean|)) and norm stats. Also
the first-layer input term |W1 x| vs bias |b1|, and, for L_f anchors, how
positives and negatives differ in norm and in dot product / tau.

A healthy embedding has pairwise cosine well below 1 and a top-direction
share well below 1; the collapsed paper runs sit at ~0.9999 for both.

Usage:
    PYTHONPATH=src python scripts/check_embedding_collapse.py --run-dir outputs/<run>
    PYTHONPATH=src python scripts/check_embedding_collapse.py --run-dir outputs/<run> --init
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from cycle.config import Config
from cycle.data.graphs import load_year_graph
from cycle.models.graph_paper import PaperGraphModule

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", required=True)
parser.add_argument("--checkpoint", default="best.pt")
parser.add_argument("--init", action="store_true",
                    help="measure the untrained module (seeded like the run) instead")
args = parser.parse_args()

with open(os.path.join(args.run_dir, "config_used.json"), encoding="utf-8") as f:
    cfg = Config.wrap(json.load(f))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g = load_year_graph(os.path.join(cfg.data.root, "processed", "graphs",
                                 f"{cfg.data.train_year}.npz"), dev)
n = g["num_nodes"]
gc = cfg.graph
torch.manual_seed(cfg.run.seed)
m = PaperGraphModule(
    feat_dim=g["X"].shape[1], hidden_dim=gc.hidden_dim, feat_drop=gc.feat_drop,
    attn_drop=gc.attn_drop, tau=gc.tau, neighbor_threshold=gc.neighbor_threshold,
    isolated_fallback=gc.isolated_fallback, feature_negatives=gc.feature_negatives,
    relation_extra_negatives=gc.get("relation_extra_negatives", 0),
).to(dev)
if not args.init:
    ckpt = torch.load(os.path.join(args.run_dir, args.checkpoint),
                      map_location=dev, weights_only=True)
    m.load_state_dict(ckpt["graph_module"])

with torch.no_grad():
    m.resample_neighbors(g, dev, generator=torch.Generator().manual_seed(1))
    m.eval()
    z_r, z_f = m(g)
    out = {"run": os.path.basename(os.path.normpath(args.run_dir)),
           "checkpoint": "init" if args.init else args.checkpoint,
           "similarity": gc.get("similarity", "dot"),
           "negative_mining": gc.get("negative_mining", "random")}
    i, j = torch.randint(n, (2, 200000), generator=torch.Generator().manual_seed(0))
    for name, z in (("z_r", z_r), ("z_f", z_f)):
        zn = F.normalize(z, dim=1)
        pc = (zn[i.to(dev)] * zn[j.to(dev)]).sum(1)
        s = torch.linalg.svdvals(z)
        sc = torch.linalg.svdvals(z - z.mean(0))
        out[name] = {
            "pair_cos_mean": pc.mean().item(), "pair_cos_min": pc.min().item(),
            "top1_energy": (s[0] ** 2 / (s ** 2).sum()).item(),
            "spread": (sc ** 2).sum().sqrt().item() / (math.sqrt(n) * z.mean(0).norm().item()),
            "norm_mean": z.norm(dim=1).mean().item(), "norm_std": z.norm(dim=1).std().item(),
        }
    X = g["X"].to_dense()
    out["W1x_mean"] = (X @ m.fc_in.weight.t()).norm(dim=1).mean().item()
    out["b1_norm"] = m.fc_in.bias.norm().item()
    if m._pos_idx is not None:
        rows = m._pos_mask.any(1).nonzero().squeeze(1)
        pos, pm = m._pos_idx[rows], m._pos_mask[rows]
        neg = m._neg_idx[rows][:, -m.relation_extra_negatives:] if m.relation_extra_negatives else None
        a = z_f[rows]
        out["Lf_pos_norm"] = z_r[pos][pm].norm(dim=-1).mean().item()
        out["Lf_pos_cos"] = F.cosine_similarity(a.unsqueeze(1), z_r[pos], dim=-1)[pm].mean().item()
        if neg is not None:
            out["Lf_neg_norm"] = z_r[neg].norm(dim=-1).mean().item()
            out["Lf_neg_cos"] = F.cosine_similarity(a.unsqueeze(1), z_r[neg], dim=-1).mean().item()
print(json.dumps(out, default=lambda x: round(x, 5)))
