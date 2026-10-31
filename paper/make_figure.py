#!/usr/bin/env python3
"""Rebuild Fig. 1 and every neural-world number in the paper from data/.

    python3 paper/make_figure.py

data/neural/<arm>-s<seed>.jsonl      one record per generation (world_net.py log)
data/neural/<arm>-s<seed>.arch.jsonl every organism that lived, with its shape
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "neural"
SEEDS = [7, 8, 9, 10, 11]
ARMS = {"evolve": ("#1f4e9c", "-", "evolve"),
        "drift": ("#c0782f", "--", "drift (random survival)"),
        "train": ("#3b8a5a", ":", "train (one brain, no evolution)")}


def jsonl(path):
    return [json.loads(line) for line in open(path)]


curves = {}
for arm in ARMS:
    runs = [jsonl(DATA / ("%s-s%d.jsonl" % (arm, s))) for s in SEEDS]
    shapes = [jsonl(DATA / ("%s-s%d.arch.jsonl" % (arm, s))) for s in SEEDS]
    acc = np.array([[r["mean_acc"] for r in rows] for rows in runs])
    curves[arm] = acc
    first = [next(g for g, a in enumerate(row) if a >= 0.95) for row in acc]
    print("== %s, seeds %s" % (arm, SEEDS))
    print("  first generation at >= 0.95: %s (mean %.1f)" % (first, np.mean(first)))
    print("  generation 3: %.2f +- %.2f" % (acc[:, 3].mean(), acc[:, 3].std(ddof=1)))
    print("  hunts per run: %s" % [sum(r["hunts"] for r in rows) for rows in runs])
    print("  organisms with 3 layers per run: %s" % [sum(o["layers"] == 3 for o in sh) for sh in shapes])
    print("  organisms with 1 layer per run:  %s" % [sum(o["layers"] == 1 for o in sh) for sh in shapes])

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.3, 1.75))
for arm, (color, ls, label) in ARMS.items():
    a = curves[arm]
    g = np.arange(a.shape[1])
    m, sd = a.mean(0), a.std(0, ddof=1)
    ax.fill_between(g, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1), color=color, alpha=0.15, lw=0)
    ax.plot(g, m, ls, color=color, lw=1.3, label=label)
ax.axhline(1 / 7, color="0.5", lw=0.6, ls="-.")
ax.text(10.9, 1 / 7 + 0.025, "one-layer ceiling (1/7)", ha="right", fontsize=6.5, color="0.4")
ax.set_xlabel("generation (800 SGD steps each)")
ax.set_ylabel("held-out accuracy")
ax.set_ylim(0, 1.04)
ax.set_xlim(0, 11)
ax.set_xticks(range(0, 12, 2))
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.52))
fig.tight_layout(pad=0.2)
fig.savefig(ROOT / "paper" / "fig_arms.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(ROOT / "docs" / "img" / "arms.png", dpi=220, bbox_inches="tight", pad_inches=0.04)
print("wrote paper/fig_arms.pdf and docs/img/arms.png")
