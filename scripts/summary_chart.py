"""R@1 by test year for the final 2019 runs; values read from eval_results.jsonl.

Candidate-mode runs use the rows evaluated with --unmapped-adjustment mean
(PROGRESS.md finding 5); the other runs use their single evaluation.

Usage (needs matplotlib, which requirements.txt does not include):
    python scripts/summary_chart.py   # writes results/summary_chart.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUTPUTS = os.path.join(REPO, "outputs")
OUT_PNG = os.path.join(REPO, "results", "summary_chart.png")
RUNS = [  # (label, run dir, unmapped_adjustment filter, color)
    # text-only control in neutral gray; runs keep the reference categorical
    # slots 1-3 (light mode). Gray #5f5e5a checked for >= 14.5 OKLab separation
    # from all three hues under normal, deutan, protan and tritan vision.
    ("text-only control", "new_2019_textonly_seed52313", None, "#5f5e5a"),
    ("paper-strict (candidate mode)", "new_2019_paper_candidate_seed52313", "mean", "#2a78d6"),
    ("paper + extra negatives (candidate mode)",
     "new_2019_paper_candidate_extranneg_seed52313", "mean", "#eb6834"),
    ("continual entities: paper + extra negatives (gold_mention)",
     "continual_2019_paper_extranneg_seed52313", None, "#1baf7a"),
]
YEARS = [2019, 2020, 2021, 2022]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def r1(run_dir, adjustment):
    rec = {}
    with open(f"{OUTPUTS}/{run_dir}/eval_results.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("unmapped_adjustment") == adjustment:
                rec[r["test_year"]] = r["recall@1"]
    return [rec[y] for y in YEARS]


fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

x = np.arange(len(YEARS))
width, gap = 0.2, 0.012  # small surface gap between adjacent bars
for i, (label, run, adj, color) in enumerate(RUNS):
    vals = r1(run, adj)
    print(label, [round(v, 4) for v in vals])
    offs = x + (i - (len(RUNS) - 1) / 2) * width
    bars = ax.bar(offs, vals, width - 2 * gap, color=color, label=label, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5, color=INK2)

ax.set_xticks(x, [str(y) for y in YEARS])
ax.set_xlabel("Test year", color=INK)
ax.set_ylabel("Recall@1", color=INK)
ax.set_ylim(0, 0.45)
ax.set_title("CYCLE reproduction: R@1 across year gaps (train year 2019, debug_n=1024, seed 52313)",
             color=INK, fontsize=12, pad=22)
ax.text(0.5, 1.015, "First three: new-entity test set. Candidate mode uses "
        "--unmapped-adjustment mean. Continual is a different test set.",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color=INK2)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(INK2)
ax.tick_params(colors=INK2, length=0)
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
for t in leg.get_texts():
    t.set_color(INK)

fig.tight_layout()
fig.savefig(OUT_PNG, facecolor=SURFACE)
print("saved", OUT_PNG)
