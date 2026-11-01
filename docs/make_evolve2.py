#!/usr/bin/env python3
"""Rebuild the evolve-2 figure and numbers from data/evolve2/.

    python3 docs/make_evolve2.py

data/evolve2/<arm>-s<seed>.jsonl is the world_net.py log of one run (one
record per generation). Arms: evolve, drift (random survival), train2 and
train4 (one brain of fixed depth, no evolution).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "evolve2"
ARMS = {"evolve": ("#1f4e9c", "-", "evolve"),
        "drift": ("#c0782f", "--", "drift (random survival)"),
        "train2": ("#3b8a5a", ":", "one brain, 2 layers"),
        "train4": ("#8a3b7a", "-.", "one brain, 4 layers")}


def jsonl(path):
    return [json.loads(line) for line in open(path)]


runs = {arm: [jsonl(p) for p in sorted(DATA.glob("%s-s*.jsonl" % arm),
                                       key=lambda p: int(p.stem.split("-s")[1]))]
        for arm in ARMS}
runs = {arm: rs for arm, rs in runs.items() if rs}
gens = min(len(r) for rs in runs.values() for r in rs)
hops = [rec["hops"] for rec in next(iter(runs.values()))[0][:gens]]
eras = [g for g in range(1, gens) if hops[g] != hops[g - 1]]


def series(rs, key):
    return np.array([[rec[key] for rec in r[:gens]] for r in rs], dtype=float)


for arm, rs in runs.items():
    acc, lay = series(rs, "mean_acc"), series(rs, "mean_layers")
    print("== %s (%d seeds)" % (arm, len(rs)))
    starts = [0] + eras
    ends = eras + [gens]
    for s, e in zip(starts, ends):
        k = hops[s]
        rec = []
        for row in acc:
            hit = next((g - s for g in range(s, e) if row[g] >= 0.9), None)
            rec.append("-" if hit is None else str(hit))
        deep = [sum(n for L, n in r[e - 1]["layer_hist"].items() if int(L) >= k + 1) / r[e - 1]["pop"]
                for r in rs]
        print("  era %d hop(s), gens %d-%d: generations to 0.9 per seed [%s]; end accuracy %.2f +- %.2f;"
              " end layers %.2f; share with >= %d layers %.2f"
              % (k, s, e - 1, " ".join(rec), acc[:, e - 1].mean(), acc[:, e - 1].std(ddof=1) if len(acc) > 1 else 0,
                 lay[:, e - 1].mean(), k + 1, np.mean(deep)))
    if arm in ("evolve", "drift"):
        print("  hunts per run: %s" % [sum(rec["hunts"] for rec in r[:gens]) for r in rs])
        print("  deepest brain per run: %s" % [max(rec["max_layers"] for rec in r[:gens]) for r in rs])

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.2, 3.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
g = np.arange(gens)
for arm, (color, ls, label) in ARMS.items():
    if arm not in runs:
        continue
    for axis, key in ((ax, "mean_acc"), (bx, "mean_layers")):
        a = series(runs[arm], key)
        m = a.mean(0)
        if len(a) > 1:
            sd = a.std(0, ddof=1)
            lo, hi = (np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1)) if key == "mean_acc" else (m - sd, m + sd)
            axis.fill_between(g, lo, hi, color=color, alpha=0.13, lw=0)
        axis.plot(g, m, ls, color=color, lw=1.3, label=label)
for axis in (ax, bx):
    for e in eras:
        axis.axvline(e - 0.5, color="0.6", lw=0.6, ls=":")
    axis.spines[["top", "right"]].set_visible(False)
for s, e in zip([0] + eras, eras + [gens]):
    ax.text((s + e - 1) / 2, 1.07, "%d hop%s" % (hops[s], "" if hops[s] == 1 else "s"),
            ha="center", fontsize=7, color="0.35")
ax.set_ylim(0, 1.12)
ax.set_ylabel("accuracy, this era")
ax.legend(frameon=False, fontsize=6.5, loc="lower left", ncol=2)
bx.set_ylabel("mean layers")
bx.set_xlabel("generation (800 SGD steps each)")
bx.set_xlim(0, gens - 1)
fig.tight_layout(pad=0.3)
out = ROOT / "docs" / "img" / "evolve2.png"
fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.04)
print("wrote", out.relative_to(ROOT))

# one population, generation by generation: what it can do, and how deep it is
SHOW = DATA / "evolve-s8.jsonl"
if SHOW.exists():
    r = jsonl(SHOW)[:gens]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.2, 3.4), sharex=True,
                                 gridspec_kw={"height_ratios": [1.3, 1]})
    hop_style = (("#9aa7b8", ":", "1 hop"), ("#c0782f", "--", "2 hops"), ("#1f4e9c", "-", "3 hops"))
    for k, (color, ls, label) in enumerate(hop_style):
        ax.plot(g, [rec["acc_by_hop"][k] for rec in r], ls, color=color, lw=1.4, label=label)
    for e in eras:
        ax.axvline(e - 0.5, color="0.6", lw=0.6, ls=":")
        bx.axvline(e - 0.5, color="0.6", lw=0.6, ls=":")
    for s, e in zip([0] + eras, eras + [gens]):
        ax.text((s + e - 1) / 2, 1.07, "world asks %d hop%s" % (hops[s], "" if hops[s] == 1 else "s"),
                ha="center", fontsize=7, color="0.35")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("population accuracy")
    ax.legend(frameon=False, fontsize=6.5, loc="center left", title="asked:", title_fontsize=6.5)
    shades = {1: "#e8e1d4", 2: "#c9d3e3", 3: "#8ea6cc", 4: "#4f6fa8", 5: "#26407a", 6: "#111f40"}
    bottom = np.zeros(gens)
    for L, color in shades.items():
        n = np.array([rec["layer_hist"].get(str(L), 0) / rec["pop"] for rec in r])
        if n.any():
            bx.bar(g, n, bottom=bottom, color=color, width=0.92, label="%d layer%s" % (L, "" if L == 1 else "s"))
            bottom += n
    bx.set_ylim(0, 1)
    bx.set_ylabel("share of brains")
    bx.set_xlabel("generation (seed 8)")
    bx.set_xlim(-0.5, gens - 0.5)
    bx.legend(frameon=False, fontsize=6, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    for axis in (ax, bx):
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    out = ROOT / "docs" / "img" / "evolve2_s8.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.04)
    print("wrote", out.relative_to(ROOT))
