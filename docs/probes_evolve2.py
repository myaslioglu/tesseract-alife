#!/usr/bin/env python3
"""The two probes behind the evolve-2 design.

    python3 docs/probes_evolve2.py mix     # mixed depths in one batch stall learning
    python3 docs/probes_evolve2.py clicks  # when does a grown brain learn two hops?

mix:    a 3-layer brain trained from scratch on batches that mix 1-hop and
        2-hop questions (hop marker in the input) for 8000 steps, two seeds.
        On pure 2-hop batches the same brain solves two hops (pill.py --hops).
clicks: a 2-layer brain learns one hop (1600 steps), grows a third layer by
        identity insertion, then trains on two hops in 800-step lives, up to
        15. Between lives: nothing, the world's weight noise, or weight noise
        plus the world's 1% chance per tensor of a hard reset.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pill                                            # noqa: E402


def mixed_hops(n, rng, upto):
    """make_hops with each example's depth drawn from 1..upto."""
    k = rng.integers(1, upto + 1, size=n)
    L = pill.HOPLEN
    x = np.zeros((n, L), dtype=np.int64)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        chain = rng.permutation(pill.VOCAB)[:pill.PAIRS + 1]
        for slot, edge in enumerate(rng.permutation(pill.PAIRS)):
            x[i, 2 * slot] = chain[edge]
            x[i, 2 * slot + 1] = chain[edge + 1]
        j = int(rng.integers(0, pill.PAIRS - k[i] + 1))
        x[i, L - 2] = pill.VOCAB + k[i] - 1
        x[i, L - 1] = chain[j]
        y[i] = chain[j + k[i]]
    return x, y


def acc(p, arch, hops, rng):
    x, y = pill.make_hops(512, rng, hops)
    return pill.accuracy(p, arch, x, y)


if sys.argv[1:] == ["mix"]:
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        arch = {"d": 32, "mlp": 2, "layers": 3}
        p = pill.init_params(arch, rng, vocab=pill.HOPVOCAB, seqlen=pill.HOPLEN)
        for step in range(2000, 8001, 2000):
            p, _ = pill.train_online(p, arch, 2000, 0.05, rng, batch=lambda n, r: mixed_hops(n, r, 2))
            a1, a2, a3 = (acc(p, arch, k, rng) for k in (1, 2, 3))
            print("mix  seed %d step %5d  acc@1hop %.2f  acc@2hop %.2f  acc@3hop %.2f"
                  % (seed, step, a1, a2, a3), flush=True)

elif sys.argv[1:] == ["clicks"]:
    for cond in ("none", "noise", "reset"):
        for seed in (0, 1):
            rng = np.random.default_rng(seed)
            a2 = {"d": 32, "mlp": 2, "layers": 2}
            p = pill.init_params(a2, rng, vocab=pill.HOPVOCAB, seqlen=pill.HOPLEN)
            p, _ = pill.train_online(p, a2, 1600, 0.05, rng, batch=lambda n, r: pill.make_hops(n, r, 1))
            a3 = dict(a2, layers=3)
            p = pill.morph(p, a2, a3, rng)
            curve = []
            for life in range(1, 16):
                if cond != "none":                         # birth: the world's mutation
                    for k in p:
                        p[k] = p[k] + rng.normal(0, 0.006 * 1.16, p[k].shape)
                        if cond == "reset" and rng.random() < 0.01:
                            p[k] = pill.init_params(a3, rng, pill.HOPVOCAB, pill.HOPLEN)[k]
                p, _ = pill.train_online(p, a3, 800, 0.05, rng, batch=lambda n, r: pill.make_hops(n, r, 2))
                curve.append(acc(p, a3, 2, rng))
                if curve[-1] > 0.95:
                    break
            print("clicks %-5s seed %d  acc@2hop per life: %s" % (cond, seed, " ".join("%.2f" % a for a in curve)),
                  flush=True)
else:
    sys.exit(__doc__)
