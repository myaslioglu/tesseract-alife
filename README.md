# Tesseract

**Self-replicating code organisms that evolve their own attention layers.**

An organism is a file. It reads its own source, decodes the genome written
inside itself, trains the neural network it carries, earns energy by solving the
environment, and — if it can afford it — rewrites itself with mutations and
writes a child to disk. Generation after generation, the population evolves not
just behaviour but the architecture itself: how many attention layers, how wide,
how fast it learns.

Named after the tesseract — the four-dimensional cube, because this is a system
whose shape changes along an axis you can't see from the inside.

> **What this is not.** Not malware, not a worm, not a botnet. Every organism
> lives inside one directory, mutates nothing but its own genome, and has no
> network access at all. This is artificial-life research in the lineage of
> Tierra and Avida. See *Containment* below — it is by construction, not by
> promise.

---

## Two levels

**1. The expression world** (`world.py`) — organisms carry a single arithmetic
expression and evolve it by character-level mutation. The habitat is the
Mandelbrot set: fertility at a coordinate is its escape time, so the only rich
ground is the boundary, a line of measure zero that is infinitely long. `fractal.py`
renders the colony on that boundary.

**2. The neural world** (`world_net.py`) — organisms carry a tiny transformer:
token + positional embeddings, causal self-attention blocks, residual MLP, and a
replay memory. Weights live in the genome as compressed float arrays. Each
generation each organism trains with hand-written backprop, then is measured on
held-out data. Children inherit the trained weights, mutated. Here the genome is
a file the world trains and breeds; `spore.py` is the version that does the
whole loop inside its own source.

`pill.py` is the brain itself — attention and backprop in ~250 lines of numpy,
gradients verified against finite differences at **4e-10** relative error. It
also carries `morph()`, which changes a brain's shape without wiping what it
knows (see *Evolving the architecture*).

`spore.py` is the same organism with **zero dependencies**: one file, standard
library only. Its genome — id, lineage and trained weights — sits on its own
`_G` line. `--train` rewrites that line in place; `--spawn` writes children
that are this same file carrying a mutated genome. Copy it to any machine with
`python3` and it lives. Smaller brain, because pure Python is ~50x slower than
numpy at the same size (13 ms vs 0.25 ms per step, measured).

## The task

Induction. A sequence of distinct tokens whose final token appeared earlier;
predict what followed it. This cannot be solved by looking at one position — it
needs a head that reports *what came before this token* and a second head that
copies. It is the smallest task where attention depth is causal, not decorative.

Measured ablation, identical budget and data (`python3 pill.py`: 800 steps of
momentum SGD, 512 held-out sequences):

| model | held-out accuracy |
|---|---|
| 2 layers, d=32 | **1.00** |
| 1 layer, d=32 | 0.14 |

One layer cannot learn it: 0.14 is what *copy a random earlier token* scores
(one of seven). That is the point of evolving layers.

## Genome

| gene | what it does |
|---|---|
| `arch` | d, mlp multiplier, layers, learning rate |
| `weights` | the trained attention parameters, float32 |
| `memory` | replay buffer of past (sequence, answer) pairs — retention |
| `eff` | metabolic efficiency: `gain ∝ eff^0.5`, `upkeep = 6·eff²` |
| `brood` | children per reproduction |
| `diet` | chance of hunting a neighbour |
| `mu` | copy fidelity |
| `cr`, `ci` | habitat coordinate (expression world) |

Optimal metabolism falls out of the curve, not a bound:
`eff* = (pts · fertility / 24)^(2/3)`.

## Evolving the architecture

Architecture mutates rarely (2% of births): ±1 layer, mlp multiplier ±1, or a
new width. The brain is **reshaped, not wiped** (`pill.morph`):

- an inserted block starts as the identity (`Wo = W2 = b2 = 0`), so the mutant
  behaves exactly like its parent and the extra layer is free capacity it can
  learn into;
- new width reads from the residual stream but writes zeros into it, and `Wq`
  is rescaled so attention keeps its temperature;
- growth leaves every logit unchanged (Δ ≤ 1e-15); removing a layer or width is
  real loss.

Without this, a mutant restarts from random weights and races a parent with
thousands of training steps behind it. It loses whatever its shape is worth.

## Predation

A hunter pays 4 energy, takes 45% of a neighbour's energy, and **adopts the
victim's brain** — whole weights, architecture and all. Two rules keep it honest:

- prey must not be **kin**: no shared parent or grandparent (`--kin 2`);
- prey must carry a **foreign brain**: relative weight distance ≥ 0.10
  (`--foreign`). Measured: siblings sit ~0.06 apart after a generation of
  training, second cousins ~0.12, unrelated brains ~1.4. The rule bites after a
  hunt — the lineage says stranger, but the brain is a copy.

And a hunter forages at 60% in the generation it hunts. Refusals are logged as
`kin` (no stranger around) or `clone` (strangers, but the same brain).

## Containment

- every organism is written into one population directory and never leaves it
- the evolved code is checked before it runs: arithmetic, comparison, boolean,
  conditional, comprehension, call, name, literal — nothing else
- no attribute access, no imports inside evolved expressions, no assignment
- each organism process runs under a 6 s timeout and a 512 MB address-space
  limit: `c*10**9` passes the whitelist, dies of `MemoryError`, scores zero and
  starves. macOS does not enforce the memory limit; there only the timeout holds
- no sockets anywhere in the organism code; the environment may fetch a corpus,
  the life never does
- a child that mutates itself into nonsense dies instead of running

## Run it

```bash
python3 pill.py                                                     # verify gradients, then learn
python3 world_net.py --gens 12 --K 12 --steps 800                   # evolve attention
python3 world_net.py --gens 12 --K 12 --steps 800 --control train   # one brain, no evolution
python3 world_net.py --gens 12 --K 12 --steps 800 --control drift   # survivors drawn by lottery
python3 world.py --gens 80 --K 60                                   # evolve expressions on the boundary
python3 fractal.py                                                  # render the colony into output/
python3 spore.py --selftest                                         # zero-dependency organism
```

`--seed` sets the neural world's seed for replicates. A spore's life, by hand —
copy it first, because `--train` rewrites the file it runs from:

```bash
cp spore.py adam.py
python3 adam.py --train 400        # learns, saves into itself
python3 adam.py --spawn 3 kids/    # three mutated copies of itself
python3 kids/s001-*.py             # each child reports who it is and how good
```

## What happened (neural world, 12 generations, seed 7)

Population 9–14, each organism training 800 steps per generation, measured on
128 fresh sequences it has never seen. Three arms, same seed, same budget.
Mean held-out accuracy:

| gen | evolve | drift (survival by lottery) | train (one brain, no evolution) |
|---|---|---|---|
| 0 | 0.17 | 0.17 | 0.17 |
| 1 | 0.30 | 0.29 | 0.24 |
| 2 | 0.51 | 0.57 | 0.57 |
| 3 | 0.76 | 0.84 | 0.90 |
| 4 | 0.97 | 0.99 | 1.00 |
| 5–11 | 1.00 (0.91 at gen 9) | 1.00 | 1.00 |

**Selection adds nothing measurable on this task.** All three arms solve
induction in about four generations — 3,200 steps, which is how long SGD takes —
and a single brain trained alone gets there as fast as the population. What
carries the learning down the lineage is inherited trained weights; selection
rides along. For evolution to earn its keep the task has to be one SGD in a
single lineage cannot crack alone: harder, shifting, or split into niches.

**Predation fires.** 11 hunts in 12 generations (zero before evolve-1); 4
hunters went hungry for lack of non-kin, none for lack of a foreign brain.

**A third layer appeared and held.** A 3-layer mutant was born at generation 6
and its line lived to the end: 3 of the 12 newborns in the last generation carry
three layers (mean 2.2). Two layers are necessary — the ablation shows it — but
on a task two layers already solve, a third is close to neutral, and that is
what the population shows. No 1-layer brain was born in this run.

## Honest limits

Written down rather than hidden, because a benchmark that lies is worse than no
benchmark:

1. **Pure Python trades reach for brain size.** The dependency-free organism runs
   anywhere but its model is too small to crack induction (0.14 held-out after
   300 steps at d=8 — the copy-a-random-token level). Portability and capability
   currently fight.
2. **Retention flatters itself.** The replay buffer's own examples score high
   because they were trained on. Held-out accuracy is the number that counts.
3. **Metabolism does not track habitat** in the expression world
   (Pearson = −0.04). Theory says rich ground selects heavy `eff`; the data
   disagrees. Three candidate explanations, all untested: migration erases local
   adaptation; the surplus is so large the optimum is flat across the range; or
   the payoff is flat near its optimum anyway — at a perfect score on fertility
   1.5, ±20% `eff` costs ~0.36 energy, less than the ±0.5 rounding of gain and
   upkeep and far less than the spread of inherited energy.
4. **Selection is not yet doing the work** — see the control arms above.
5. **One run per arm.** Seeds are fixed and replicates are one flag away, but not
   yet run. 128 held-out sequences give each organism's accuracy a standard
   error of about ±0.04.
6. **What was wrong before evolve-1**, kept on the record:
   - Predation could not fire. Every organism descends from one root and the
     kin rule looked 24 ancestors deep, so for ~24 generations everyone was kin.
     The earlier diagnosis here — "the population converges on one
     architecture" — was wrong: `arch` includes `lr`, which mutates at every
     birth, so architectures always differed. Comparing weights alone would not
     have helped.
   - A 2→1 layer mutation crashed the world (`KeyError: '1.Wq'`, about one birth
     in 300). "One-layer mutants are born and starve" was never observed: mean
     layers was exactly 2.0 in every generation.
   - Any architecture change restarted the brain from random weights, so
     "layers never dropped to 1" measured that handicap, not necessity.
   - The earlier results table does not reproduce from the code it shipped with
     (generation 0: 0.46 claimed, 0.17 at its own seed), and its "best" was the
     highest-energy organism rather than the maximum — hence best < mean. The
     log now records both (`top_acc`, `fittest_acc`).
   - The 1-layer vs 2-layer ablation needed 800 steps; `pill.py` ran 400
     (2 layers: 0.65).

MIT licensed.
