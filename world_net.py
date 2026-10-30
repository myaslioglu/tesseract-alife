#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""world_net.py — organisms that carry a trained transformer in their genome.

Each organism's genome now contains:
  * arch     — d, mlp multiplier, layers, learning rate  (evolvable)
  * weights  — the trained attention parameters, float32 (evolvable)
  * memory   — a replay buffer of past (sequence, answer) pairs (retention)
  * the usual economics: eff, brood, diet, mu, and a habitat coordinate

Life cycle, one generation after another:
  1. load brain, train `steps` on fresh batches AND on its own stored memory
  2. the world measures held-out accuracy (new data) and retention accuracy
     (the data it kept from earlier generations)
  3. energy buys reproduction; the child inherits the weights, mutated
  4. a hunter may steal a neighbour's brain wholesale — foreign weights only

Import is allowed (numpy), which the old world forbade. The network is used by
the environment, not by the organisms: word statistics are fetched once and the
task is sampled from them. Organisms never get a socket — a mutated child could
otherwise send your files anywhere, and nothing about this experiment needs it.

Usage: python3 world_net.py --gens 25 --K 12 --net
"""
import base64
import io
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import pill

ROOT = Path(__file__).resolve().parent
POP = ROOT / "net_pop"
DEAD = ROOT / "net_dead"
LOG = ROOT / "net_world.jsonl"
REPORT = ROOT / "net_report.md"
CORPUS = ROOT / "corpus.txt"

COST = 5
UPKEEP = 6.0
EFF_EXP = 0.5
HUNT_COST = 4
HUNT_SHARE = 0.45
MAXITER_0 = 60
MAXITER_STEP = 3

SEED = {
    "id": "n000-root",
    "gen": 0,
    "parent": None,
    "energy": 18,
    "mu": 0.16,
    "eff": 1.0,
    "brood": 1,
    "diet": 0.05,
    "cr": -0.75,
    "ci": 0.11,
    "arch": {"d": 32, "mlp": 2, "layers": 2, "lr": 0.05},
    "weights": None,
    "memory": [],
    "lineage": ["n000-root"],
}


# ---------------------------------------------------------------- encoding
def encode_weights(p):
    buf = io.BytesIO()
    np.savez_compressed(buf, **p)
    return base64.b64encode(buf.getvalue()).decode()


def decode_weights(blob):
    buf = io.BytesIO(base64.b64decode(blob))
    z = np.load(buf)
    return {k: z[k].astype(np.float64) for k in z.files}


# ---------------------------------------------------------------- network
def fetch_corpus(url="https://www.gutenberg.org/files/1342/1342-0.txt", limit=400_000):
    """The environment may use the network. Organisms may not."""
    if CORPUS.exists():
        return CORPUS.read_text(encoding="utf-8", errors="ignore")[:limit]
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=20) as fh:
            text = fh.read(limit).decode("utf-8", "ignore")
        CORPUS.write_text(text, encoding="utf-8")
        return text
    except Exception as exc:
        pass
    try:                                  # this machine's python has a broken cert chain
        import subprocess
        raw = subprocess.run(["curl", "-sL", "--max-time", "25", url],
                             capture_output=True).stdout
        if raw:
            text = raw.decode("utf-8", "ignore")[:limit]
            CORPUS.write_text(text, encoding="utf-8")
            return text
    except Exception as exc:
        print("  corpus fetch failed (%s) — synthetic" % exc)
    return None


def make_task(text, n, rng, seqlen=9):
    """Induction sequences whose token statistics come from real text, if given."""
    if text:
        chars = [c for c in sorted(set(text)) if c.isalpha()][:12]
        if len(chars) >= 8:
            idx = {c: i for i, c in enumerate(chars)}
            arr = np.array([idx[c] for c in text if c in idx], dtype=np.int64)
            if arr.size > 4096:
                x = np.zeros((n, seqlen), dtype=np.int64)
                y = np.zeros(n, dtype=np.int64)
                for i in range(n):
                    j = int(rng.integers(0, arr.size - seqlen - 1))
                    win = arr[j:j + seqlen - 1].copy()
                    k = int(rng.integers(0, seqlen - 2))
                    x[i, :seqlen - 1] = win
                    x[i, seqlen - 1] = win[k]
                    y[i] = win[k + 1]
                return x, y, len(chars)
    x, y = pill.make_batch(n, rng)
    return x, y, pill.VOCAB


# ---------------------------------------------------------------- habitat
def escape(cr, ci, maxiter):
    zr = zi = 0.0
    for _ in range(maxiter):
        zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
        if zr * zr + zi * zi > 4.0:
            return _
    return maxiter


def fertility(cr, ci, maxiter):
    n = escape(cr, ci, maxiter)
    if n >= maxiter:
        return 0.30
    return round(0.35 + 2.40 * (n / maxiter), 4)


# ---------------------------------------------------------------- mutation
def mutate(g, rng):
    """Brains drift: weights get noise, architecture rarely changes."""
    p = decode_weights(g["weights"])
    arch = dict(g["arch"])

    # architecture mutation (rare, and costly: shapes no longer line up)
    if rng.random() < 0.02:
        which = rng.choice(["layers", "d", "mlp"])
        if which == "layers":
            arch["layers"] = max(1, min(3, arch["layers"] + rng.choice([-1, 1])))
        elif which == "d":
            arch["d"] = int(rng.choice([16, 24, 32, 48]))
        else:
            arch["mlp"] = max(1, min(3, arch["mlp"] + rng.choice([-1, 1])))
        arch = {"d": int(arch["d"]), "mlp": int(arch["mlp"]),
                "layers": int(arch["layers"]), "lr": float(arch["lr"])}
        if not pill.arch_ok(p, arch):
            p = pill.init_params(arch, rng)          # new brain, from scratch
    arch["lr"] = float(np.clip(arch["lr"] * np.exp(rng.normal(0, 0.10)), 0.005, 0.12))

    sigma = 0.006 * (1.0 + g["mu"])
    for k in p:
        p[k] = p[k] + rng.normal(0, sigma, p[k].shape)
        if rng.random() < 0.01:                     # occasional hard reset
            p[k] = pill.init_params(arch, rng)[k]

    g["arch"] = arch
    g["weights"] = encode_weights(p)
    g["mu"] = round(float(np.clip(g["mu"] + rng.normal(0, 0.02), 0.02, 0.60)), 4)
    g["eff"] = round(float(np.clip(g["eff"] * np.exp(rng.normal(0, 0.07)), 0.02, 8.0)), 4)
    g["brood"] = int(max(1, min(3, g["brood"] + rng.choice([0, 0, 0, 1, -1]))))
    g["diet"] = round(float(np.clip(g["diet"] + rng.normal(0, 0.07), 0.0, 1.0)), 3)
    step = 0.004 if rng.random() > 0.08 else 0.05
    g["cr"] = round(g["cr"] + rng.normal(0, step), 7)
    g["ci"] = round(g["ci"] + rng.normal(0, step), 7)

    # retention: the memory buffer is inherited, and it leaks
    mem = [m for m in g.get("memory", []) if rng.random() > 0.08]
    g["memory"] = mem[-32:]
    return g


def write_child(parent, energy, outdir, rng):
    child = mutate({k: (v if not isinstance(v, list) else list(v)) for k, v in parent.items()}, rng)
    child["gen"] = parent["gen"] + 1
    child["parent"] = parent["id"]
    child["energy"] = max(2, energy)
    child["id"] = "%03d-%04x" % (child["gen"], rng.integers(0, 0xFFFF))
    child["lineage"] = (parent["lineage"] + [child["id"]])[-24:]
    path = Path(outdir) / (child["id"] + ".json")
    path.write_text(json.dumps(child), encoding="utf-8")
    return child["id"]


# ---------------------------------------------------------------- training
def live(g, rng, text, steps, vocab):
    """One organism's generation: train, then be measured."""
    arch = g["arch"]
    p = decode_weights(g["weights"]) if g["weights"] else pill.init_params(arch, rng)
    if not pill.arch_ok(p, arch):
        p = pill.init_params(arch, rng)

    for _ in range(steps):
        xf, yf, _ = make_task(text, 32, rng)
        mem = g.get("memory") or []
        if mem:
            xm = np.array([m[0] for m in mem], dtype=np.int64)
            ym = np.array([m[1] for m in mem], dtype=np.int64)
            x = np.concatenate([xf, xm])[:64]
            y = np.concatenate([yf, ym])[:64]
        else:
            x, y = xf, yf
        logits, cache = pill.forward(p, x, arch["layers"])
        loss, dlogits, _ = pill.softmax_xent(logits, y)
        grad = pill.backward(p, cache, dlogits, arch["layers"])
        norm = np.sqrt(sum(float((v * v).sum()) for v in grad.values()))
        scale = min(1.0, 1.0 / (norm + 1e-12))
        for k in p:
            p[k] = p[k] - arch["lr"] * scale * grad[k]

    xh, yh, _ = make_task(text, 128, rng)
    acc = pill.accuracy(p, arch, xh, yh)
    ret = 0.0
    mem = g.get("memory") or []
    if len(mem) >= 4:
        xm = np.array([m[0] for m in mem], dtype=np.int64)
        ym = np.array([m[1] for m in mem], dtype=np.int64)
        ret = pill.accuracy(p, arch, xm, ym)

    # remember what it just saw, up to 32 exemplars
    mem = (g.get("memory") or [])[-24:] + [[xh[i].tolist(), int(yh[i])] for i in range(8)]
    g["memory"] = mem[-32:]
    g["weights"] = encode_weights(p)
    return g, acc, ret


def main():
    gens, K, pop0, steps = 25, 12, 10, 800
    text = None
    if "--gens" in sys.argv:
        gens = int(sys.argv[sys.argv.index("--gens") + 1])
    if "--K" in sys.argv:
        K = int(sys.argv[sys.argv.index("--K") + 1])
    if "--steps" in sys.argv:
        steps = int(sys.argv[sys.argv.index("--steps") + 1])
    if "--net" in sys.argv:
        text = fetch_corpus()
        print("corpus: %s" % ("fetched %d chars" % len(text) if text else "synthetic"))

    rng = np.random.default_rng(7)
    for d in (POP, DEAD):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    LOG.unlink(missing_ok=True)

    cur = POP / "gen000"
    cur.mkdir(parents=True)
    root = dict(SEED)
    root["weights"] = encode_weights(pill.init_params(root["arch"], rng))
    (cur / "n000-root.json").write_text(json.dumps(root), encoding="utf-8")
    for _ in range(pop0 - 1):
        write_child(root, 12, cur, rng)
    print("primordial soup: %d brains, %d layers each"
          % (pop0, root["arch"]["layers"]))

    hist, log = [], open(LOG, "w", encoding="utf-8")
    t0 = time.time()
    vocab = pill.VOCAB

    for gen in range(gens):
        maxiter = MAXITER_0 + gen * MAXITER_STEP
        files = sorted(cur.glob("*.json"))
        if not files:
            print("EXTINCTION at gen %d" % gen)
            break

        # predation: steal a foreign brain
        gmap = {f: json.loads(f.read_text(encoding="utf-8")) for f in files}
        energy = {f: gmap[f].get("energy", 0) for f in files}
        hunts = refused = 0
        hunted = set()
        for f in files:
            g = gmap[f]
            if len(files) < 2 or rng.random() >= g.get("diet", 0.0):
                continue
            mine = set(g.get("lineage", []))
            prey = [x for x in files if x != f
                    and not (mine & set(gmap[x].get("lineage", [])))
                    and gmap[x]["arch"] != g["arch"]]
            if not prey:
                refused += 1
                continue
            victim = prey[int(rng.integers(0, len(prey)))]
            share = int(energy[victim] * HUNT_SHARE)
            if share < 2:
                continue
            energy[f] += share - HUNT_COST
            energy[victim] -= share
            g["weights"] = gmap[victim]["weights"]     # the brain is eaten
            g["arch"] = dict(gmap[victim]["arch"])
            gmap[f] = g
            hunted.add(f)
            hunts += 1

        rows = []
        for f in files:
            g = gmap[f]
            g, acc, ret = live(g, rng, text, steps, vocab)
            gmap[f] = g
            fert = fertility(g["cr"], g["ci"], maxiter)
            pts = round(acc * 24) + round(ret * 4)
            if f in hunted:
                pts = round(pts * 0.6)          # hunting costs foraging time
            gain = max(0.0, pts * fert * (g["eff"] ** EFF_EXP))
            cost = UPKEEP * (g["eff"] ** 2)
            rows.append({"file": f, "g": g, "acc": acc, "ret": ret,
                         "energy": energy[f] + round(gain) - round(cost)})

        rows.sort(key=lambda r: (-r["energy"], -r["acc"]))
        mean_brood = sum(r["g"]["brood"] for r in rows) / len(rows)
        n_surv = max(3, min(len(rows), int(round(K / max(1.0, mean_brood)))))
        survivors = [r for r in rows if r["energy"] >= COST][:n_surv]
        for r in [x for x in rows if x not in survivors]:
            shutil.move(str(r["file"]), str(DEAD / f"{gen:03d}-{r['file'].name}"))

        nxt = POP / ("gen%03d" % (gen + 1))
        nxt.mkdir(exist_ok=True)
        born = 0
        for r in survivors:
            for _ in range(r["g"]["brood"]):
                write_child(r["g"], max(2, r["energy"] // (1 + r["g"]["brood"])), nxt, rng)
                born += 1
            shutil.move(str(r["file"]), str(DEAD / f"{gen:03d}-{r['file'].name}"))

        best = rows[0]
        layers = [r["g"]["arch"]["layers"] for r in rows]
        rec = {"gen": gen, "maxiter": maxiter, "pop": len(rows),
               "surv": len(survivors), "born": born, "hunts": hunts,
               "refused": refused,
               "best_acc": round(best["acc"], 3), "best_ret": round(best["ret"], 3),
               "mean_acc": round(sum(r["acc"] for r in rows) / len(rows), 3),
               "mean_ret": round(sum(r["ret"] for r in rows) / len(rows), 3),
               "mean_layers": round(sum(layers) / len(layers), 2),
               "two_plus": sum(1 for L in layers if L >= 2),
               "mean_eff": round(sum(r["g"]["eff"] for r in rows) / len(rows), 3),
               "best_arch": best["g"]["arch"]}
        hist.append(rec)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print("gen %02d pop %2d surv %2d born %2d | acc best %.2f mean %.2f | "
              "ret %.2f | layers %.1f (>=2: %d/%d) | eff %.2f"
              % (gen, rec["pop"], rec["surv"], born, rec["best_acc"],
                 rec["mean_acc"], rec["best_ret"], rec["mean_layers"],
                 rec["two_plus"], rec["pop"], rec["mean_eff"]))
        cur = nxt

    log.close()
    print("\n--- %d generations, %.0fs ---" % (len(hist), time.time() - t0))
    if hist:
        first, last = hist[0], hist[-1]
        print("accuracy %0.2f -> %0.2f | layers %0.1f -> %0.1f"
              % (first["mean_acc"], last["mean_acc"],
                 first["mean_layers"], last["mean_layers"]))

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("# HCS net world — evolving attention, generation by generation\n\n")
        fh.write("| gen | pop | surv | born | hunts | best acc | mean acc | retention |"
                 " layers | 2+ layers | eff |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for h in hist:
            fh.write("| %s | %s | %s | %s | %s | %.2f | %.2f | %.2f | %.1f | %d | %.2f |\n" % (
                h["gen"], h["pop"], h["surv"], h["born"], h["hunts"],
                h["best_acc"], h["mean_acc"], h["mean_ret"], h["mean_layers"],
                h["two_plus"], h["mean_eff"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
