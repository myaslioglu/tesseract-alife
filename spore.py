#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""spore.py — one file, no dependencies, runs anywhere Python 3 exists.

The whole organism in a single file: genome, attention, backprop, memory, and
the ability to write a mutated copy of itself. Standard library only. No numpy,
no pip, no build step. Copy this file to any machine with python3 and it lives.

The genome is the `_G` line below: id, lineage, and the trained brain. Training
rewrites that line in place; breeding copies this whole file with a mutated
genome. An empty `_G` is a fresh spore and grows a random brain on first use.

    python3 spore.py                 # who am I, and how good am I
    python3 spore.py --selftest      # verify gradients, learn, breed, read a child back
    python3 spore.py --train 400     # train this individual, save itself
    python3 spore.py --spawn 3 out/  # write three mutated children

Trade-off, stated up front: pure Python is roughly 50x slower than numpy. The
model is therefore smaller (d=8). Reach costs speed; that is the deal.
"""
import base64
import json
import math
import os
import random
import re
import sys

_G = ""

V, L, D, MLP, LAYERS = 8, 8, 8, 1, 2     # vocab, seqlen, width, mlp mult, layers
LR = 0.05
CLIP = 1.0


# ---------------------------------------------------------------- task
def make_batch(n, rng):
    """Induction, unambiguous: `L-1` distinct tokens, query repeats one of them."""
    xs, ys = [], []
    for _ in range(n):
        toks = rng.sample(range(V), L - 1)
        j = rng.randrange(0, L - 2)
        row = toks + [toks[j]]
        xs.append(row)
        ys.append(toks[j + 1])
    return xs, ys


# ---------------------------------------------------------------- params
def init_params(rng):
    s = 1.0 / math.sqrt(D)
    h = D * MLP
    p = {"tok": [[rng.gauss(0, 0.3) for _ in range(D)] for _ in range(V)],
         "pos": [[rng.gauss(0, 0.1) for _ in range(D)] for _ in range(L)]}
    for l in range(LAYERS):
        for name, scale in (("Wq", s), ("Wk", s), ("Wv", s), ("Wo", s)):
            p["%d.%s" % (l, name)] = [[rng.gauss(0, scale) for _ in range(D)] for _ in range(D)]
        p["%d.W1" % l] = [[rng.gauss(0, math.sqrt(2.0 / D)) for _ in range(h)] for _ in range(D)]
        p["%d.b1" % l] = [0.0] * h
        p["%d.W2" % l] = [[rng.gauss(0, math.sqrt(2.0 / h)) for _ in range(D)] for _ in range(h)]
        p["%d.b2" % l] = [0.0] * D
    p["Wu"] = [[rng.gauss(0, s) for _ in range(V)] for _ in range(D)]
    p["bu"] = [0.0] * V
    return p


def zeros_like(p):
    out = {}
    for k, v in p.items():
        out[k] = [0.0] * len(v) if not isinstance(v[0], list) else [[0.0] * len(v[0]) for _ in v]
    return out


# ---------------------------------------------------------------- math
def vmat(v, W):
    """row-vector times matrix."""
    m, k = len(v), len(W[0])
    return [sum(v[i] * W[i][j] for i in range(m)) for j in range(k)]


def acc_outer(G, v, g):
    for i, vi in enumerate(v):
        if vi == 0.0:
            continue
        Gi = G[i]
        for j, gj in enumerate(g):
            Gi[j] += vi * gj


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------- forward
def forward(p, x):
    B = len(x)
    X = [[[p["tok"][x[b][i]][d] + p["pos"][i][d] for d in range(D)] for i in range(L)]
         for b in range(B)]
    caches = []
    for l in range(LAYERS):
        Wq, Wk, Wv, Wo = p["%d.Wq" % l], p["%d.Wk" % l], p["%d.Wv" % l], p["%d.Wo" % l]
        Q = [[vmat(X[b][i], Wq) for i in range(L)] for b in range(B)]
        K = [[vmat(X[b][i], Wk) for i in range(L)] for b in range(B)]
        Vv = [[vmat(X[b][i], Wv) for i in range(L)] for b in range(B)]

        A = []
        for b in range(B):
            rows = []
            for i in range(L):
                scores = []
                for j in range(L):
                    if j > i:
                        scores.append(-1e9)
                    else:
                        scores.append(dot(Q[b][i], K[b][j]) / math.sqrt(D))
                m = max(scores)
                e = [math.exp(s - m) for s in scores]
                tot = sum(e)
                rows.append([v / tot for v in e])
            A.append(rows)

        C = []
        for b in range(B):
            cb = []
            for i in range(L):
                acc = [0.0] * D
                Ai = A[b][i]
                for j in range(L):
                    a = Ai[j]
                    if a == 0.0:
                        continue
                    Vj = Vv[b][j]
                    for t in range(D):
                        acc[t] += a * Vj[t]
                cb.append(acc)
            C.append(cb)

        H = [[[c + xv for c, xv in zip(vmat(C[b][i], Wo), X[b][i])] for i in range(L)]
             for b in range(B)]
        Z1 = [[[z for z in vmat(H[b][i], p["%d.W1" % l])] for i in range(L)] for b in range(B)]
        for b in range(B):
            for i in range(L):
                bb = p["%d.b1" % l]
                Z1[b][i] = [Z1[b][i][t] + bb[t] for t in range(len(bb))]
        M = [[[max(0.0, z) for z in Z1[b][i]] for i in range(L)] for b in range(B)]
        Z2 = [[vmat(M[b][i], p["%d.W2" % l]) for i in range(L)] for b in range(B)]
        for b in range(B):
            for i in range(L):
                bb = p["%d.b2" % l]
                Z2[b][i] = [Z2[b][i][t] + bb[t] for t in range(D)]
        Xn = [[[Z2[b][i][t] + H[b][i][t] for t in range(D)] for i in range(L)] for b in range(B)]
        caches.append((X, Q, K, Vv, A, C, H, Z1, M))
        X = Xn

    last = [X[b][L - 1] for b in range(B)]
    logits = [vmat(last[b], p["Wu"]) for b in range(B)]
    for b in range(B):
        for t in range(V):
            logits[b][t] += p["bu"][t]
    return logits, (x, X, last, caches)


def softmax_xent(logits, y):
    B = len(y)
    loss = 0.0
    dl = []
    for b in range(B):
        m = max(logits[b])
        e = [math.exp(v - m) for v in logits[b]]
        tot = sum(e)
        pr = [v / tot for v in e]
        loss -= math.log(max(pr[y[b]], 1e-12))
        g = [p / B for p in pr]
        g[y[b]] -= 1.0 / B
        dl.append(g)
    return loss / B, dl


def backward(p, cache, dlogits):
    x, Xend, last, caches = cache
    B = len(x)
    g = zeros_like(p)
    for b in range(B):
        acc_outer(g["Wu"], last[b], dlogits[b])
        for t in range(V):
            g["bu"][t] += dlogits[b][t]

    dX = [[[0.0] * D for _ in range(L)] for _ in range(B)]
    for b in range(B):
        for t in range(D):
            dX[b][L - 1][t] += sum(p["Wu"][t][o] * dlogits[b][o] for o in range(V))

    for l in range(LAYERS - 1, -1, -1):
        X, Q, K, Vv, A, C, H, Z1, M = caches[l]
        dH = [[list(dX[b][i]) for i in range(L)] for b in range(B)]

        dZ2 = [[list(dX[b][i]) for i in range(L)] for b in range(B)]
        for b in range(B):
            for i in range(L):
                acc_outer(g["%d.W2" % l], M[b][i], dZ2[b][i])
                for t in range(D):
                    g["%d.b2" % l][t] += dZ2[b][i][t]

        dM = [[vmat(dZ2[b][i], transpose(p["%d.W2" % l])) for i in range(L)] for b in range(B)]
        dZ1 = [[[dM[b][i][t] if Z1[b][i][t] > 0 else 0.0 for t in range(len(dM[b][i]))]
                for i in range(L)] for b in range(B)]
        for b in range(B):
            for i in range(L):
                acc_outer(g["%d.W1" % l], H[b][i], dZ1[b][i])
                for t in range(len(dZ1[b][i])):
                    g["%d.b1" % l][t] += dZ1[b][i][t]

        W1T = transpose(p["%d.W1" % l])
        for b in range(B):
            for i in range(L):
                add = vmat(dZ1[b][i], W1T)
                for t in range(D):
                    dH[b][i][t] += add[t]

        # dH must be complete before Wo is charged: the MLP branch feeds back into H
        for b in range(B):
            for i in range(L):
                acc_outer(g["%d.Wo" % l], C[b][i], dH[b][i])

        dC = [[vmat(dH[b][i], transpose(p["%d.Wo" % l])) for i in range(L)] for b in range(B)]
        dA = [[[0.0] * L for _ in range(L)] for _ in range(B)]
        dV = [[[0.0] * D for _ in range(L)] for _ in range(B)]
        for b in range(B):
            for i in range(L):
                for j in range(L):
                    if j <= i:
                        dA[b][i][j] = dot(dC[b][i], Vv[b][j])
                for j in range(L):
                    a = A[b][i][j]
                    if a == 0.0:
                        continue
                    for t in range(D):
                        dV[b][j][t] += a * dC[b][i][t]

        # V is a weight projection too: it needs its own gradient
        for b in range(B):
            for j in range(L):
                acc_outer(g["%d.Wv" % l], X[b][j], dV[b][j])

        dS = [[[0.0] * L for _ in range(L)] for _ in range(B)]
        for b in range(B):
            for i in range(L):
                row = A[b][i]
                dotv = sum(dA[b][i][j] * row[j] for j in range(L))
                for j in range(L):
                    if j <= i:
                        dS[b][i][j] = row[j] * (dA[b][i][j] - dotv) / math.sqrt(D)

        for b in range(B):
            for i in range(L):
                for j in range(L):
                    if j > i or dS[b][i][j] == 0.0:
                        continue
                    ds = dS[b][i][j]
                    acc_outer(g["%d.Wq" % l], X[b][i], [ds * t for t in K[b][j]])
                    acc_outer(g["%d.Wk" % l], X[b][j], [ds * t for t in Q[b][i]])

        WqT = transpose(p["%d.Wq" % l])
        WkT = transpose(p["%d.Wk" % l])
        WvT = transpose(p["%d.Wv" % l])
        for b in range(B):
            for i in range(L):
                dq = [0.0] * D
                for j in range(i + 1):
                    ds = dS[b][i][j]
                    if ds:
                        for t in range(D):
                            dq[t] += ds * K[b][j][t]
                dk = [0.0] * D
                for h in range(i, L):
                    ds = dS[b][h][i]
                    if ds:
                        for t in range(D):
                            dk[t] += ds * Q[b][h][t]
                addq = vmat(dq, WqT)
                addk = vmat(dk, WkT)
                addv = vmat(dV[b][i], WvT)
                for t in range(D):
                    dX[b][i][t] = dH[b][i][t] + addq[t] + addk[t] + addv[t]

    for b in range(B):
        for i in range(L):
            tokrow = g["tok"][x[b][i]]
            for t in range(D):
                tokrow[t] += dX[b][i][t]
                g["pos"][i][t] += dX[b][i][t]
    return g


def transpose(M):
    return [list(r) for r in zip(*M)]


def accuracy(p, x, y):
    logits, _ = forward(p, x)
    hit = 0
    for b in range(len(y)):
        if logits[b].index(max(logits[b])) == y[b]:
            hit += 1
    return hit / len(y)


def clip_step(p, g, lr):
    norm = math.sqrt(sum(v * v for k in g for row in
                         (g[k] if isinstance(g[k][0], list) else [g[k]])
                         for v in (row if isinstance(row, list) else [row])))
    scale = min(1.0, CLIP / (norm + 1e-12))
    for k in p:
        if isinstance(p[k][0], list):
            for i in range(len(p[k])):
                for j in range(len(p[k][i])):
                    p[k][i][j] -= lr * scale * g[k][i][j]
        else:
            for i in range(len(p[k])):
                p[k][i] -= lr * scale * g[k][i]


def gradcheck(rng, eps=1e-5):
    p = init_params(rng)
    x, y = make_batch(2, rng)
    logits, cache = forward(p, x)
    loss, dl = softmax_xent(logits, y)
    g = backward(p, cache, dl)
    worst = (0.0, "")
    for k in p:
        matrix = isinstance(p[k][0], list)
        rows, grows = (p[k], g[k]) if matrix else ([p[k]], [g[k]])   # a bias is one row
        for (i, j) in [(i, j) for i in range(len(rows)) for j in range(len(rows[0]))][:3]:
            old = rows[i][j]
            rows[i][j] = old + eps
            li, _ = softmax_xent(forward(p, x)[0], y)
            rows[i][j] = old - eps
            lm, _ = softmax_xent(forward(p, x)[0], y)
            rows[i][j] = old
            num = (li - lm) / (2 * eps)
            ana = grows[i][j]
            err = abs(num - ana) / max(1.0, abs(num) + abs(ana))
            if err > worst[0]:
                worst = (err, "%s[%d][%d]" % (k, i, j) if matrix else "%s[%d]" % (k, j))
    return worst


# ---------------------------------------------------------------- life
def train(p, steps, rng, lr=LR):
    losses = []
    for _ in range(steps):
        x, y = make_batch(4, rng)
        logits, cache = forward(p, x)
        loss, dl = softmax_xent(logits, y)
        g = backward(p, cache, dl)
        clip_step(p, g, lr)
        losses.append(loss)
    return p, losses


# ---------------------------------------------------------------- self
GENE = re.compile(r'^_G = "(.*)"$', re.M)


def read_self(path=None):
    """This file's source, and the genome written inside it (None for a fresh spore)."""
    with open(path or __file__, encoding="utf-8") as fh:
        src = fh.read()
    blob = GENE.search(src).group(1)
    return src, (json.loads(base64.b64decode(blob)) if blob else None)


def write_self(path, src, g):
    """Write `src` to `path` with `g` as its genome: a save, or a birth."""
    blob = base64.b64encode(json.dumps(g).encode()).decode()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(GENE.sub('_G = "%s"' % blob, src, count=1))


def newborn(rng):
    return {"id": "s000-root", "gen": 0, "parent": None, "lineage": ["s000-root"],
            "steps": 0, "params": init_params(rng)}


def spawn(g, src, rng, outdir, n=1):
    """Write n children: this very file, carrying a mutated copy of my genome."""
    os.makedirs(outdir, exist_ok=True)
    made = []
    for _ in range(n):
        child = json.loads(json.dumps(g))
        p = child["params"]
        for k in p:
            if isinstance(p[k][0], list):
                for i in range(len(p[k])):
                    for j in range(len(p[k][i])):
                        if rng.random() < 0.15:
                            p[k][i][j] += rng.gauss(0, 0.02)
            else:
                for i in range(len(p[k])):
                    if rng.random() < 0.15:
                        p[k][i] += rng.gauss(0, 0.02)
        child["gen"] = g["gen"] + 1
        child["parent"] = g["id"]
        child["id"] = "s%03d-%06x" % (child["gen"], rng.getrandbits(24))
        child["lineage"] = (g["lineage"] + [child["id"]])[-24:]
        name = os.path.join(outdir, child["id"] + ".py")
        write_self(name, src, child)
        made.append(name)
    return made


if __name__ == "__main__":
    src, g = read_self()
    if "--selftest" in sys.argv:
        rng = random.Random(7)
        err, where = gradcheck(rng)
        print("gradcheck worst relative error: %.2e at %s" % (err, where))
        xte, yte = make_batch(200, rng)
        p = init_params(rng)
        print("start accuracy %.2f" % accuracy(p, xte, yte))
        import time
        t = time.time()
        p, losses = train(p, 300, rng)
        dt = time.time() - t
        print("after 300 steps: accuracy %.2f  loss %.2f  (%.1f ms/step)"
              % (accuracy(p, xte, yte), losses[-1], 1000 * dt / 300))
        kids = spawn(dict(newborn(rng), params=p, steps=300), src, rng, "spores", 2)
        _, back = read_self(kids[0])
        print("spawned:", kids)
        print("child reads its own genome: %s, gen %d, parent %s, accuracy %.2f"
              % (back["id"], back["gen"], back["parent"], accuracy(back["params"], xte, yte)))
        sys.exit(0)

    rng = random.Random()
    g = g or newborn(rng)
    if "--train" in sys.argv:
        n = int(sys.argv[sys.argv.index("--train") + 1])
        g["params"], losses = train(g["params"], n, rng)
        g["steps"] += n
        write_self(__file__, src, g)
        print("%s trained %d steps (%d in total), loss %.3f, saved into itself"
              % (g["id"], n, g["steps"], losses[-1]))
        sys.exit(0)

    if "--spawn" in sys.argv:
        i = sys.argv.index("--spawn")
        print(json.dumps({"children": spawn(g, src, rng, sys.argv[i + 2], int(sys.argv[i + 1]))}))
        sys.exit(0)

    xte, yte = make_batch(200, rng)
    print("%s, generation %d, %d steps trained, held-out accuracy %.2f"
          % (g["id"], g["gen"], g["steps"], accuracy(g["params"], xte, yte)))
