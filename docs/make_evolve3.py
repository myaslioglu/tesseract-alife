#!/usr/bin/env python3
"""The evolve-3 analysis, as planned in docs/EVOLVE3.md, from data/evolve3/.

    python3 docs/make_evolve3.py

data/evolve3/<arm>-s<seed>.jsonl            one record per generation
data/evolve3/<arm>-s<seed>.organisms.jsonl  one record per organism per generation
Arms: evolve (--rank score), evolveE (--rank energy; seeds 7-11 are the
evolve-2 runs in data/evolve2/, which have no organism logs), drift, train2,
train4. Seeds 7 to 26.
"""
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "evolve3"
OLD = ROOT / "data" / "evolve2"
SEEDS = list(range(7, 27))
ARMS = {"evolve": ("#1f4e9c", "-", "evolve (survival ranks score)"),
        "evolveE": ("#7a8fb8", "--", "evolve, energy rule (evolve-2)"),
        "drift": ("#c0782f", "--", "drift (random survival)"),
        "train2": ("#3b8a5a", ":", "one brain, 2 layers"),
        "train4": ("#8a3b7a", "-.", "one brain, 4 layers")}
SOLVER = 0.95


def jsonl(path):
    return [json.loads(line) for line in open(path)]


def gens_of(arm, s):
    p = DATA / ("%s-s%d.jsonl" % (arm, s))
    if not p.exists() and arm == "evolveE":
        p = OLD / ("evolve-s%d.jsonl" % s)
    return jsonl(p) if p.exists() else None


def orgs_of(arm, s):
    p = DATA / ("%s-s%d.organisms.jsonl" % (arm, s))
    return jsonl(p) if p.exists() else None


runs = {arm: {s: r for s in SEEDS if (r := gens_of(arm, s)) is not None} for arm in ARMS}
G = 30
hops = [rec["hops"] for rec in runs["train2"][SEEDS[0]][:G]]
eras = [(g0, g1) for g0, g1 in zip([0, 3, 15], [3, 15, 30])]


def recovered(r, era):
    g0, g1 = eras[era]
    return any(rec["mean_acc"] >= 0.9 for rec in r[g0:g1])


def last3(r, era):
    g1 = eras[era][1]
    return float(np.mean([rec["mean_acc"] for rec in r[g1 - 3:g1]]))


def fisher_greater(a, n1, b, n2):
    """One-sided Fisher exact p: P(first group has >= a successes | margins)."""
    k, n = a + b, n1 + n2
    return sum(comb(k, i) * comb(n - k, n1 - i) for i in range(a, min(k, n1) + 1)) / comb(n, n1)


def perm_greater(x, y, reps=100_000, seed=0):
    """One-sided permutation p for mean(x) - mean(y), random splits."""
    rng = np.random.default_rng(seed)
    pool, n = np.array(x + y), len(x)
    obs = np.mean(x) - np.mean(y)
    hits = 0
    for _ in range(reps):
        p = rng.permutation(pool)
        hits += p[:n].mean() - p[n:].mean() >= obs - 1e-12
    return (hits + 1) / (reps + 1)


print("seeds per arm:", {arm: len(rs) for arm, rs in runs.items()})
table = {}
for arm, rs in runs.items():
    table[arm] = {era: sum(recovered(r, era) for r in rs.values()) for era in (1, 2)}
    print("%-8s recovered: 2 hops %2d of %d | 3 hops %2d of %d | last-3 accuracy 2 hops %.2f, 3 hops %.2f"
          % (arm, table[arm][1], len(rs), table[arm][2], len(rs),
             np.mean([last3(r, 1) for r in rs.values()]), np.mean([last3(r, 2) for r in rs.values()])))

ev, dr, ee, t4 = runs["evolve"], runs["drift"], runs["evolveE"], runs["train4"]
print("\nPRIMARY  recovery from the step to two hops, evolve vs drift: %d/%d vs %d/%d, one-sided Fisher p = %.4f"
      % (table["evolve"][1], len(ev), table["drift"][1], len(dr),
         fisher_greater(table["evolve"][1], len(ev), table["drift"][1], len(dr))))
print("secondary 1  three hops, evolve vs drift: %d/%d vs %d/%d, Fisher p = %.4f"
      % (table["evolve"][2], len(ev), table["drift"][2], len(dr),
         fisher_greater(table["evolve"][2], len(ev), table["drift"][2], len(dr))))
for era, name in ((1, "two hops"), (2, "three hops")):
    x, y = [last3(r, era) for r in ev.values()], [last3(r, era) for r in dr.values()]
    print("secondary 2  last-3 accuracy, %s: evolve %.3f vs drift %.3f, permutation p = %.4f"
          % (name, np.mean(x), np.mean(y), perm_greater(x, y)))
for era, name in ((1, "two hops"), (2, "three hops")):
    print("secondary 3  recovery, %s, evolve vs energy rule: %d/%d vs %d/%d, Fisher p = %.4f"
          % (name, table["evolve"][era], len(ev), table["evolveE"][era], len(ee),
             fisher_greater(table["evolve"][era], len(ev), table["evolveE"][era], len(ee))))

print("secondary 4  solver retention (brains at >= %.2f that survive to reproduce):" % SOLVER)
for arm in ("evolve", "evolveE", "drift"):
    solvers = [o for s in SEEDS if (os_ := orgs_of(arm, s)) for o in os_ if o["acc"] >= SOLVER]
    kept = sum(o["survived"] for o in solvers)
    print("    %-8s %4d of %4d (%.2f)%s" % (arm, kept, len(solvers), kept / max(1, len(solvers)),
          "  (seeds with organism logs only)" if arm == "evolveE" else ""))

print("secondary 5  generations from the first solver to recovery (organism logs):")
for arm in ("evolve", "evolveE", "drift"):
    out = []
    for s in SEEDS:
        os_, r = orgs_of(arm, s), runs[arm].get(s)
        if os_ is None or r is None:
            continue
        for era in (1, 2):
            g0, g1 = eras[era]
            first = next((g for g in range(g0, g1) if any(o["acc"] >= SOLVER for o in os_ if o["gen"] == g)), None)
            if first is None:
                continue
            rec = next((g for g in range(first, g1) if r[g]["mean_acc"] >= 0.9), None)
            out.append("-" if rec is None else str(rec - first))
    print("    %-8s %s" % (arm, " ".join(out) if out else "no solvers"))

lives = {arm: np.mean([sum(rec["pop"] for rec in r[:G]) for r in rs.values()]) for arm, rs in runs.items()}
print("secondary 6  evolve vs one brain with 4 layers: recovered 2 hops %d/%d vs %d/%d, 3 hops %d/%d vs %d/%d;"
      " brain lives per run %.0f vs %.0f"
      % (table["evolve"][1], len(ev), table["train4"][1], len(t4), table["evolve"][2], len(ev),
         table["train4"][2], len(t4), lives["evolve"], lives["train4"]))

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.2, 3.6), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
g = np.arange(G)
for arm, (color, ls, label) in ARMS.items():
    rs = list(runs[arm].values())
    for axis, key in ((ax, "mean_acc"), (bx, "mean_layers")):
        a = np.array([[rec[key] for rec in r[:G]] for r in rs], dtype=float)
        axis.plot(g, a.mean(0), ls, color=color, lw=1.4, label="%s, %d seeds" % (label, len(rs)))
for axis in (ax, bx):
    for g0, _ in eras[1:]:
        axis.axvline(g0 - 0.5, color="0.6", lw=0.6, ls=":")
    axis.spines[["top", "right"]].set_visible(False)
for (g0, g1) in eras:
    ax.text((g0 + g1 - 1) / 2, 1.07, "%d hop%s" % (hops[g0], "" if hops[g0] == 1 else "s"),
            ha="center", fontsize=7, color="0.35")
ax.set_ylim(0, 1.12)
ax.set_ylabel("mean accuracy, this era")
bx.set_ylabel("mean layers")
bx.set_xlabel("generation (800 SGD steps each)")
bx.set_xlim(0, G - 1)
fig.tight_layout(pad=0.3)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.0))
out = ROOT / "docs" / "img" / "evolve3.png"
fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.04)
print("wrote", out.relative_to(ROOT))
