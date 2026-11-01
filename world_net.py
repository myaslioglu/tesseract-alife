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
       python3 world_net.py --control train    # one brain, no evolution: the baseline
       python3 world_net.py --control drift    # same world, survivors drawn by lottery
"""
import base64
import io
import json
import random
import re
import shutil
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

import pill

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus.txt"

COST = 5
UPKEEP = 6.0
EFF_EXP = 0.5
HUNT_COST = 4
HUNT_SHARE = 0.45
MAXITER_0 = 60
MAXITER_STEP = 3
KIN_DEPTH = 2      # kin = share a parent or grandparent; the whole tree is one family
FOREIGN = 0.10     # relative weight distance; siblings ~0.06, second cousins ~0.12
ARCH_MU = 0.02     # chance per birth that the architecture mutates (--arch-mu)
MAX_LAYERS = 3     # cap on attention blocks (--max-layers)
MAX_HOPS = pill.MAXHOPS   # deepest era in a hop schedule; accuracy is logged at each depth
MOMENTUM = 0.0     # SGD momentum inside a life (--momentum); 0 is the paper's learner

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


@lru_cache(maxsize=4)
def letter_stream(text, size=pill.VOCAB):
    """The text's `size` most frequent letters, lowercased, and where each occurs.

    The text is reduced to a stream of just those letters. Returns the letters
    (most frequent first), the stream length, and for each letter the sorted
    positions at which it occurs in the stream. Cached: the corpus is the same
    string on every call, and recounting 400k characters per batch is slow.
    """
    low = text.lower()
    counts = Counter(c for c in low if c.isalpha())
    chars = tuple(sorted(counts, key=lambda c: (-counts[c], c))[:size])
    idx = {c: i for i, c in enumerate(chars)}
    stream = np.array([idx[c] for c in low if c in idx], dtype=np.int64)
    where = tuple(np.flatnonzero(stream == v) for v in range(len(chars)))
    return chars, stream.size, where


def make_task(text, n, rng, seqlen=9):
    """Induction sequences whose token statistics come from real text, if given.

    Vocabulary: the `pill.VOCAB` (12) most frequent letters of the text,
    lowercased. Each sequence starts at a random point in the text and reads
    forward, keeping a letter only the first time it appears, until it holds
    `seqlen-1` distinct letters. The prefix follows the text's order, but
    nothing repeats. As in `pill.make_batch`, the query is one of those
    letters, so it occurs exactly once earlier and the letter after it is the
    only right answer.
    """
    if text:
        chars, size, where = letter_stream(text)
        m = seqlen - 1
        if len(chars) >= m and size > 4096:
            # starts at or before the m-th latest final occurrence still have m
            # distinct letters ahead of them; later starts do not
            last = sorted((int(w[-1]) for w in where), reverse=True)[m - 1]
            j = rng.integers(0, last + 1, size=n)
            nxt = np.full((len(chars), n), size, dtype=np.int64)
            for v, w in enumerate(where):           # first occurrence of v at or after j
                at = np.searchsorted(w, j)
                ok = at < w.size
                nxt[v, ok] = w[at[ok]]
            prefix = np.argsort(nxt, axis=0, kind="stable")[:m].T   # by first appearance
            k = rng.integers(0, m - 1, size=n)
            rows = np.arange(n)
            x = np.zeros((n, seqlen), dtype=np.int64)
            x[:, :m] = prefix
            x[:, m] = prefix[rows, k]
            y = prefix[rows, k + 1]
            return x, y, len(chars)
    x, y = pill.make_batch(n, rng)
    return x, y, pill.VOCAB


def task(text, n, rng, hops):
    """The environment's task this generation: `hops`-hop induction in a hop
    era (--hops-every), otherwise the one-hop task of the paper. Each era asks
    only its own depth; old eras survive only in what organisms remember."""
    if hops:
        x, y = pill.make_hops(n, rng, hops)
        return x, y, pill.HOPVOCAB
    return make_task(text, n, rng)


def seqlen_for(hops):
    return pill.HOPLEN if hops else pill.SEQLEN


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

    # architecture mutation: rare, and reshaped rather than wiped
    if rng.random() < ARCH_MU:
        which = rng.choice(["layers", "d", "mlp"])
        if which == "layers":
            arch["layers"] = max(1, min(MAX_LAYERS, arch["layers"] + rng.choice([-1, 1])))
        elif which == "d":
            arch["d"] = int(rng.choice([16, 24, 32, 48]))
        else:
            arch["mlp"] = max(1, min(3, arch["mlp"] + rng.choice([-1, 1])))
        arch = {"d": int(arch["d"]), "mlp": int(arch["mlp"]),
                "layers": int(arch["layers"]), "lr": float(arch["lr"])}
        if not pill.arch_ok(p, arch):
            p = pill.morph(p, g["arch"], arch, rng)  # reshaped, not wiped
    arch["lr"] = float(np.clip(arch["lr"] * np.exp(rng.normal(0, 0.10)), 0.005, 0.12))

    sigma = 0.006 * (1.0 + g["mu"])
    for k in p:
        p[k] = p[k] + rng.normal(0, sigma, p[k].shape)
        if rng.random() < 0.01:                     # occasional hard reset
            p[k] = pill.init_params(arch, rng, p["tok"].shape[0], p["pos"].shape[0])[k]

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


def kin(a, b):
    """Share an ancestor within KIN_DEPTH generations, self included."""
    n = KIN_DEPTH + 1
    return bool(set(a.get("lineage", [])[-n:]) & set(b.get("lineage", [])[-n:]))


def brain_distance(pa, pb):
    """Relative L2 distance between two brains; different shapes are infinitely far."""
    if set(pa) != set(pb) or any(pa[k].shape != pb[k].shape for k in pa):
        return float("inf")
    diff = np.sqrt(sum(float(((pa[k] - pb[k]) ** 2).sum()) for k in pa))
    na = np.sqrt(sum(float((v * v).sum()) for v in pa.values()))
    nb = np.sqrt(sum(float((v * v).sum()) for v in pb.values()))
    return diff / (0.5 * (na + nb) + 1e-12)


def write_child(parent, energy, outdir, rng, mutant=True):
    child = {k: (v if not isinstance(v, list) else list(v)) for k, v in parent.items()}
    if mutant:
        child = mutate(child, rng)
    child["gen"] = parent["gen"] + 1
    child["parent"] = parent["id"]
    child["energy"] = max(2, energy)
    child["id"] = "%03d-%04x" % (child["gen"], rng.integers(0, 0xFFFF))
    child["lineage"] = (parent["lineage"] + [child["id"]])[-24:]
    path = Path(outdir) / (child["id"] + ".json")
    path.write_text(json.dumps(child), encoding="utf-8")
    return child["id"]


# ---------------------------------------------------------------- training
def live(g, rng, text, steps, vocab, hops=0):
    """One organism's generation: train, then be measured.

    With a hop schedule, `hops` is this generation's task depth. Fitness is
    held-out accuracy at that depth; accuracy at every depth up to MAX_HOPS is
    measured too, so the log shows what a brain can still do and cannot yet do.
    """
    arch = g["arch"]
    L = seqlen_for(hops)
    p = decode_weights(g["weights"]) if g["weights"] else pill.init_params(arch, rng, vocab, L)
    if not pill.arch_ok(p, arch):
        p = pill.init_params(arch, rng, vocab, L)

    vel = {k: np.zeros_like(v) for k, v in p.items()} if MOMENTUM else None
    for _ in range(steps):
        xf, yf, _ = task(text, 32, rng, hops)
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
            if vel is None:
                p[k] = p[k] - arch["lr"] * scale * grad[k]
            else:                                   # velocity starts at zero each life
                vel[k] = MOMENTUM * vel[k] - arch["lr"] * scale * grad[k]
                p[k] = p[k] + vel[k]

    xh, yh, _ = task(text, 128, rng, hops)
    acc = pill.accuracy(p, arch, xh, yh)
    ret = 0.0
    mem = g.get("memory") or []
    if len(mem) >= 4:
        xm = np.array([m[0] for m in mem], dtype=np.int64)
        ym = np.array([m[1] for m in mem], dtype=np.int64)
        ret = pill.accuracy(p, arch, xm, ym)
    by_hop = []
    if hops:
        for k in range(1, MAX_HOPS + 1):
            xk, yk = pill.make_hops(128, rng, k)
            by_hop.append(pill.accuracy(p, arch, xk, yk))

    # remember what it just saw, up to 32 exemplars
    mem = (g.get("memory") or [])[-24:] + [[xh[i].tolist(), int(yh[i])] for i in range(8)]
    g["memory"] = mem[-32:]
    g["weights"] = encode_weights(p)
    return g, acc, ret, by_hop


def main():
    global KIN_DEPTH, FOREIGN, ARCH_MU, MAX_LAYERS, MOMENTUM
    gens, K, pop0, steps, seed, control = 25, 12, 10, 800, 7, None
    hops_every, layers0 = 0, None
    text = None
    if "--hops-every" in sys.argv:        # eras: 1 hop, then 2 after N gens, 3 after 2N
        hops_every = int(sys.argv[sys.argv.index("--hops-every") + 1])
    if "--momentum" in sys.argv:
        MOMENTUM = float(sys.argv[sys.argv.index("--momentum") + 1])
    if "--arch-mu" in sys.argv:
        ARCH_MU = float(sys.argv[sys.argv.index("--arch-mu") + 1])
    if "--max-layers" in sys.argv:
        MAX_LAYERS = int(sys.argv[sys.argv.index("--max-layers") + 1])
    if "--layers" in sys.argv:            # the founding brain's depth
        layers0 = int(sys.argv[sys.argv.index("--layers") + 1])
    if "--gens" in sys.argv:
        gens = int(sys.argv[sys.argv.index("--gens") + 1])
    if "--K" in sys.argv:
        K = int(sys.argv[sys.argv.index("--K") + 1])
    if "--steps" in sys.argv:
        steps = int(sys.argv[sys.argv.index("--steps") + 1])
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
    if "--kin" in sys.argv:
        KIN_DEPTH = int(sys.argv[sys.argv.index("--kin") + 1])
    if "--foreign" in sys.argv:
        FOREIGN = float(sys.argv[sys.argv.index("--foreign") + 1])
    if "--control" in sys.argv:
        control = sys.argv[sys.argv.index("--control") + 1]
        if control not in ("train", "drift"):
            sys.exit("--control is 'train' (one brain, no evolution) or 'drift' (random survival)")
    if "--net" in sys.argv:
        text = fetch_corpus()
        print("corpus: %s" % ("fetched %d chars" % len(text) if text else "synthetic"))
        if text:
            print("task letters: %s" % "".join(letter_stream(text)[0]))

    tag = "-" + control if control else ""
    POP, DEAD = ROOT / ("net_pop" + tag), ROOT / ("net_dead" + tag)
    LOG, REPORT = ROOT / ("net_world%s.jsonl" % tag), ROOT / ("net_report%s.md" % tag)
    if control == "train":
        pop0 = 1                     # one brain: never mutated, never culled, never hunted

    rng = np.random.default_rng(seed)
    for d in (POP, DEAD):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    LOG.unlink(missing_ok=True)

    cur = POP / "gen000"
    cur.mkdir(parents=True)
    vocab = pill.HOPVOCAB if hops_every else pill.VOCAB
    root = dict(SEED)
    if layers0:
        root["arch"] = dict(SEED["arch"], layers=layers0)
    root["weights"] = encode_weights(
        pill.init_params(root["arch"], rng, vocab, seqlen_for(1 if hops_every else 0)))
    (cur / "n000-root.json").write_text(json.dumps(root), encoding="utf-8")
    for _ in range(pop0 - 1):
        write_child(root, 12, cur, rng)
    print("primordial soup: %d brains, %d layers each%s%s"
          % (pop0, root["arch"]["layers"],
             " | control arm: %s" % control if control else "",
             " | task deepens by one hop every %d generations" % hops_every if hops_every else ""))

    hist, log = [], open(LOG, "w", encoding="utf-8")
    t0 = time.time()

    for gen in range(gens):
        maxiter = MAXITER_0 + gen * MAXITER_STEP
        hops = min(MAX_HOPS, 1 + gen // hops_every) if hops_every else 0
        files = sorted(cur.glob("*.json"))
        if not files:
            print("EXTINCTION at gen %d" % gen)
            break

        # predation: steal a foreign brain
        gmap = {f: json.loads(f.read_text(encoding="utf-8")) for f in files}
        brains = {f: decode_weights(gmap[f]["weights"]) for f in files}
        energy = {f: gmap[f].get("energy", 0) for f in files}
        hunts = refused_kin = refused_clone = 0
        hunted = set()
        for f in files:
            g = gmap[f]
            if len(files) < 2 or rng.random() >= g.get("diet", 0.0):
                continue
            strangers = [x for x in files if x != f and not kin(g, gmap[x])]
            prey = [x for x in strangers if brain_distance(brains[f], brains[x]) >= FOREIGN]
            if not prey:
                if strangers:
                    refused_clone += 1        # unrelated by descent, same brain: nothing to gain
                else:
                    refused_kin += 1
                continue
            victim = prey[int(rng.integers(0, len(prey)))]
            share = int(energy[victim] * HUNT_SHARE)
            if share < 2:
                continue
            energy[f] += share - HUNT_COST
            energy[victim] -= share
            g["weights"] = gmap[victim]["weights"]     # the brain is eaten
            g["arch"] = dict(gmap[victim]["arch"])
            brains[f] = brains[victim]
            hunted.add(f)
            hunts += 1

        rows = []
        for f in files:
            g = gmap[f]
            g, acc, ret, by_hop = live(g, rng, text, steps, vocab, hops)
            gmap[f] = g
            fert = fertility(g["cr"], g["ci"], maxiter)
            pts = round(acc * 24) + round(ret * 4)
            if f in hunted:
                pts = round(pts * 0.6)          # hunting costs foraging time
            gain = max(0.0, pts * fert * (g["eff"] ** EFF_EXP))
            cost = UPKEEP * (g["eff"] ** 2)
            rows.append({"file": f, "g": g, "acc": acc, "ret": ret, "by_hop": by_hop,
                         "energy": energy[f] + round(gain) - round(cost)})

        rows.sort(key=lambda r: (-r["energy"], -r["acc"]))
        mean_brood = sum(r["g"]["brood"] for r in rows) / len(rows)
        n_surv = max(3, min(len(rows), int(round(K / max(1.0, mean_brood)))))
        if control == "train":
            survivors = rows
        elif control == "drift":                    # same world, survival by lottery
            survivors = [rows[i] for i in rng.permutation(len(rows))][:n_surv]
        else:
            survivors = [r for r in rows if r["energy"] >= COST][:n_surv]
        for r in [x for x in rows if x not in survivors]:
            shutil.move(str(r["file"]), str(DEAD / f"{gen:03d}-{r['file'].name}"))

        nxt = POP / ("gen%03d" % (gen + 1))
        nxt.mkdir(exist_ok=True)
        born = 0
        for r in survivors:
            kids = 1 if control == "train" else r["g"]["brood"]
            for _ in range(kids):
                write_child(r["g"], max(2, r["energy"] // (1 + kids)), nxt, rng,
                            mutant=control != "train")
                born += 1
            shutil.move(str(r["file"]), str(DEAD / f"{gen:03d}-{r['file'].name}"))

        fittest = rows[0]
        layers = [r["g"]["arch"]["layers"] for r in rows]
        layer_hist = {L: layers.count(L) for L in sorted(set(layers))}
        by_hop = [round(sum(r["by_hop"][k] for r in rows) / len(rows), 3)
                  for k in range(MAX_HOPS)] if hops else []
        rec = {"gen": gen, "arm": control or "evolve", "seed": seed, "maxiter": maxiter,
               "hops": hops,
               "pop": len(rows), "surv": len(survivors), "born": born, "hunts": hunts,
               "refused_kin": refused_kin, "refused_clone": refused_clone,
               "top_acc": round(max(r["acc"] for r in rows), 3),
               "fittest_acc": round(fittest["acc"], 3), "fittest_ret": round(fittest["ret"], 3),
               "mean_acc": round(sum(r["acc"] for r in rows) / len(rows), 3),
               "mean_ret": round(sum(r["ret"] for r in rows) / len(rows), 3),
               "acc_by_hop": by_hop,
               "mean_layers": round(sum(layers) / len(layers), 2),
               "max_layers": max(layers),
               "layer_hist": {str(L): n for L, n in layer_hist.items()},
               "two_plus": sum(1 for L in layers if L >= 2),
               "mean_eff": round(sum(r["g"]["eff"] for r in rows) / len(rows), 3),
               "fittest_arch": fittest["g"]["arch"]}
        hist.append(rec)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        depth = ("by hop " + " ".join("%.2f" % a for a in by_hop) + " | ") if hops else ""
        print("gen %02d%s pop %2d surv %2d born %2d hunt %d (kin %d clone %d) | "
              "acc top %.2f fittest %.2f mean %.2f | %slayers %.1f %s | eff %.2f"
              % (gen, " h%d" % hops if hops else "", rec["pop"], rec["surv"], born, hunts,
                 refused_kin, refused_clone, rec["top_acc"], rec["fittest_acc"], rec["mean_acc"],
                 depth, rec["mean_layers"],
                 " ".join("%dx%d" % (n, L) for L, n in layer_hist.items()), rec["mean_eff"]))
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
        fh.write("arm: **%s**, seed %d, kin depth %d, foreign distance %.2f, "
                 "architecture mutation %.2f, max layers %d, momentum %.2f%s\n\n"
                 % (control or "evolve", seed, KIN_DEPTH, FOREIGN, ARCH_MU, MAX_LAYERS, MOMENTUM,
                    ", one more hop every %d generations" % hops_every if hops_every else ""))
        hop_cols = "".join(" acc@%dhop |" % (k + 1) for k in range(MAX_HOPS)) if hops_every else ""
        fh.write("| gen | hops | pop | surv | born | hunts | refused kin/clone | top acc | fittest acc |"
                 " mean acc |%s retention | layers | max layers | eff |\n" % hop_cols)
        fh.write("|---" * (14 + (MAX_HOPS if hops_every else 0)) + "|\n")
        for h in hist:
            hop_vals = "".join(" %.2f |" % a for a in h["acc_by_hop"]) if hops_every else ""
            fh.write("| %s | %s | %s | %s | %s | %s | %d/%d | %.2f | %.2f | %.2f |%s %.2f | %.1f | %d | %.2f |\n" % (
                h["gen"], h["hops"] or "-", h["pop"], h["surv"], h["born"], h["hunts"],
                h["refused_kin"], h["refused_clone"], h["top_acc"], h["fittest_acc"],
                h["mean_acc"], hop_vals, h["mean_ret"], h["mean_layers"], h["max_layers"],
                h["mean_eff"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
