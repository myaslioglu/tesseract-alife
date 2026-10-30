#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pill.py — the AI pill: a tiny transformer with hand-written backprop.

Why hand-written: the organism must carry its brain inside its own genome and
train it with nothing but numpy. No torch, no autograd, no network.

Architecture (n_layers residual blocks):
    X = tok[x] + pos
    repeat n_layers:
        A = causal_softmax(QKᵀ/√d)          # self-attention
        H = A·V·Wo + X                       # residual
        X = relu(H·W1 + b1)·W2 + b2 + H      # relu MLP + residual
    logits = X[-1]·Wu + bu

Two layers is the minimum that can hold an induction circuit: the first head
learns "what came before this token", the second uses that to copy. One layer
can only match tokens, not their successors.

Task: induction. Sequence whose last token already appeared earlier; predict the
token that followed it. Gradients are verified against finite differences by
`gradcheck()` before anything is trained.
"""
import numpy as np

VOCAB = 12
SEQLEN = 9


# --------------------------------------------------------------------------
# task
# --------------------------------------------------------------------------
def make_batch(n, rng, vocab=VOCAB, seqlen=SEQLEN, distract=0):
    """Induction task, unambiguous by construction.

    The prefix uses `seqlen-1` DISTINCT tokens, then the query is one of them
    placed at the end. Because nothing repeats, exactly one earlier position can
    match the query, so the answer is well defined and a perfect score is
    reachable. With repeats the task has an irreducible error floor and no model
    can exceed ~0.78 — that ceiling was ambiguity, not inability.
    """
    L = seqlen
    x = np.zeros((n, L), dtype=np.int64)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        toks = rng.permutation(vocab)[:L - 1]      # distinct by construction
        x[i, :L - 1] = toks
        j = int(rng.integers(0, L - 2))
        x[i, L - 1] = toks[j]                     # query appears once, earlier
        y[i] = toks[j + 1]                        # what followed it
    return x, y


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
def init_params(arch, rng, vocab=VOCAB):
    d, mult, L = arch["d"], arch["mlp"], arch["layers"]
    h = d * mult
    s = 1.0 / np.sqrt(d)
    p = {"tok": rng.normal(0, 0.3, (vocab, d)),
         "pos": rng.normal(0, 0.1, (SEQLEN, d))}
    for l in range(L):
        p["%d.Wq" % l] = rng.normal(0, s, (d, d))
        p["%d.Wk" % l] = rng.normal(0, s, (d, d))
        p["%d.Wv" % l] = rng.normal(0, s, (d, d))
        p["%d.Wo" % l] = rng.normal(0, s, (d, d))
        p["%d.W1" % l] = rng.normal(0, np.sqrt(2.0 / d), (d, h))
        p["%d.b1" % l] = np.zeros(h)
        p["%d.W2" % l] = rng.normal(0, np.sqrt(2.0 / h), (h, d))
        p["%d.b2" % l] = np.zeros(d)
    p["Wu"] = rng.normal(0, s, (d, vocab))
    p["bu"] = np.zeros(vocab)
    return p


def arch_ok(p, arch):
    d = arch["d"]
    try:
        if p["tok"].shape[1] != d:
            return False
        if ("%d.Wq" % (arch["layers"] - 1)) not in p:
            return False
        if p["0.W1"].shape != (d, d * arch["mlp"]):
            return False
    except Exception:
        return False
    return True


# --------------------------------------------------------------------------
# forward / backward
# --------------------------------------------------------------------------
def forward(p, x, n_layers, keep=True):
    B, L = x.shape
    d = p["tok"].shape[1]
    X = p["tok"][x] + p["pos"][:L]
    mask = np.triu(np.ones((L, L), dtype=bool), k=1)
    caches = []
    for l in range(n_layers):
        Q = X @ p["%d.Wq" % l]
        K = X @ p["%d.Wk" % l]
        V = X @ p["%d.Wv" % l]
        S = Q @ K.transpose(0, 2, 1) / np.sqrt(d)
        S = np.where(mask, -1e9, S)
        S = S - S.max(axis=-1, keepdims=True)
        A = np.exp(S)
        A /= A.sum(axis=-1, keepdims=True)
        C = A @ V
        H = C @ p["%d.Wo" % l] + X
        Z1 = H @ p["%d.W1" % l] + p["%d.b1" % l]
        M = np.maximum(Z1, 0.0)
        Z2 = M @ p["%d.W2" % l] + p["%d.b2" % l]
        Xn = Z2 + H
        if keep:
            caches.append((X, Q, K, V, A, C, H, Z1, M))
        X = Xn
    last = X[:, -1, :]
    logits = last @ p["Wu"] + p["bu"]
    return logits, (x, X, last, caches)


def softmax_xent(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    pr = e / e.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.log(np.maximum(pr[np.arange(n), y], 1e-12)).mean()
    dlogits = pr.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n
    return loss, dlogits, pr


def backward(p, cache, dlogits, n_layers):
    x, Xend, last, caches = cache
    B, L = x.shape
    d = p["tok"].shape[1]
    g = {}
    g["Wu"] = last.T @ dlogits
    g["bu"] = dlogits.sum(axis=0)
    dX = np.zeros_like(Xend)
    dX[:, -1, :] += dlogits @ p["Wu"].T

    for l in range(n_layers - 1, -1, -1):
        X, Q, K, V, A, C, H, Z1, M = caches[l]
        dH = dX.copy()                # residual branch into H
        dZ2 = dX
        g["%d.W2" % l] = M.reshape(-1, M.shape[-1]).T @ dZ2.reshape(-1, d)
        g["%d.b2" % l] = dZ2.sum(axis=(0, 1))
        dM = dZ2 @ p["%d.W2" % l].T
        dZ1 = dM * (Z1 > 0)
        g["%d.W1" % l] = H.reshape(-1, d).T @ dZ1.reshape(-1, Z1.shape[-1])
        g["%d.b1" % l] = dZ1.sum(axis=(0, 1))
        dH = dH + dZ1 @ p["%d.W1" % l].T   # full gradient wrt H
        g["%d.Wo" % l] = C.reshape(-1, d).T @ dH.reshape(-1, d)

        dC = dH @ p["%d.Wo" % l].T
        dA = dC @ V.transpose(0, 2, 1)
        dV = A.transpose(0, 2, 1) @ dC
        dS = A * (dA - (dA * A).sum(axis=-1, keepdims=True))
        dS /= np.sqrt(d)
        dQ = dS @ K
        dK = dS.transpose(0, 2, 1) @ Q
        g["%d.Wq" % l] = X.reshape(-1, d).T @ dQ.reshape(-1, d)
        g["%d.Wk" % l] = X.reshape(-1, d).T @ dK.reshape(-1, d)
        g["%d.Wv" % l] = X.reshape(-1, d).T @ dV.reshape(-1, d)
        dX = (dH                      # residual into the block input
              + dQ @ p["%d.Wq" % l].T
              + dK @ p["%d.Wk" % l].T
              + dV @ p["%d.Wv" % l].T)

    dtok = np.zeros_like(p["tok"])
    np.add.at(dtok, x.reshape(-1), dX.reshape(-1, d))
    g["tok"] = dtok
    dpos = np.zeros_like(p["pos"])
    dpos[:L] = dX.sum(axis=0)
    g["pos"] = dpos
    return g


def gradcheck(rng, eps=1e-6):
    """Numeric vs analytic gradients. If this fails, nothing else is worth running."""
    arch = {"d": 4, "mlp": 1, "layers": 2}
    p = init_params(arch, rng, vocab=5)
    x, y = make_batch(3, rng, vocab=5, seqlen=4, distract=0)
    logits, cache = forward(p, x, arch["layers"])
    loss, dlogits, _ = softmax_xent(logits, y)
    g = backward(p, cache, dlogits, arch["layers"])
    worst = (0.0, "")
    for k, v in p.items():
        flat = v.reshape(-1)
        for i in range(min(flat.size, 5)):
            old = flat[i]
            flat[i] = old + eps
            li, _, _ = softmax_xent(forward(p, x, arch["layers"])[0], y)
            flat[i] = old - eps
            lm, _, _ = softmax_xent(forward(p, x, arch["layers"])[0], y)
            flat[i] = old
            num = (li - lm) / (2 * eps)
            ana = g[k].reshape(-1)[i]
            err = abs(num - ana) / max(1.0, abs(num) + abs(ana))
            if err > worst[0]:
                worst = (err, "%s[%d]" % (k, i))
    return worst


def accuracy(p, arch, x, y):
    logits, _ = forward(p, x, arch["layers"], keep=False)
    return float((logits.argmax(axis=1) == y).mean())


def train_online(p, arch, steps, lr, rng, bs=32, momentum=0.9, clip=1.0):
    """Fresh batch every step: learning, not memorising. Gradients are clipped
    by global norm, without which a 2-layer model diverges to NaN."""
    vel = {k: np.zeros_like(v) for k, v in p.items()}
    hist = []
    for _ in range(steps):
        x, y = make_batch(bs, rng)
        logits, cache = forward(p, x, arch["layers"])
        loss, dlogits, _ = softmax_xent(logits, y)
        g = backward(p, cache, dlogits, arch["layers"])
        norm = np.sqrt(sum(float((v * v).sum()) for v in g.values()))
        scale = min(1.0, clip / (norm + 1e-12))
        for k in p:
            vel[k] = momentum * vel[k] - lr * scale * g[k]
            p[k] += vel[k]
        hist.append(float(loss))
    return p, hist


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    err, where = gradcheck(rng)
    print("gradcheck worst relative error: %.3e at %s" % (err, where))
    assert err < 1e-5, "gradients are wrong; fix before training"

    xte, yte = make_batch(512, rng)
    for layers in (1, 2):
        for d in (16, 32):
            arch = {"d": d, "mlp": 2, "layers": layers}
            p = init_params(arch, rng)
            acc0 = accuracy(p, arch, xte, yte)
            p, h = train_online(p, arch, 400, 0.05, rng)
            print("layers %d d %2d | start %.2f -> held-out %.2f | loss %.3f"
                  % (layers, d, acc0, accuracy(p, arch, xte, yte), h[-1]))
