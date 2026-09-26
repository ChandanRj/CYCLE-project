"""R@1 by test year for the three 2019 runs; values read from eval_results.jsonl.

Usage (needs matplotlib, which requirements.txt does not include):
    python scripts/summary_chart.py   # writes outputs/summary_chart.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
RUNS = [  # (label, run dir, color: reference categorical slots 1-3, light mode)
    ("paper-strict v2", "new_2019_paper_seed52313_v2", "#2a78d6"),
    ("paper + extra negatives", "new_2019_paper_extranneg_seed52313", "#eb6834"),
    ("continual + extra negatives", "continual_2019_paper_extranneg_seed52313", "#1baf7a"),
]
YEARS = [2019, 2020, 2021, 2022]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def r1(run_dir):
    rec = {}
    with open(f"{ROOT}/{run_dir}/eval_results.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rec[r["test_year"]] = r["recall@1"]
    return [rec[y] for y in YEARS]


fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

x = np.arange(len(YEARS))
width, gap = 0.26, 0.012  # small surface gap between adjacent bars
for i, (label, run, color) in enumerate(RUNS):
    vals = r1(run)
    print(label, [round(v, 4) for v in vals])
    offs = x + (i - 1) * width
    bars = ax.bar(offs, vals, width - 2 * gap, color=color, label=label, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, color=INK2)

ax.set_xticks(x, [str(y) for y in YEARS])
ax.set_xlabel("Test year", color=INK)
ax.set_ylabel("Recall@1", color=INK)
ax.set_ylim(0, 0.45)
ax.set_title("CYCLE reproduction: R@1 across year gaps (train year 2019, debug_n=1024)",
             color=INK, fontsize=12, pad=12)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(INK2)
ax.tick_params(colors=INK2, length=0)
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
for t in leg.get_texts():
    t.set_color(INK)

fig.tight_layout()
fig.savefig(f"{ROOT}/summary_chart.png", facecolor=SURFACE)
print("saved", f"{ROOT}/summary_chart.png")
