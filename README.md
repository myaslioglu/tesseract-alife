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
held-out data. Children inherit the trained weights, mutated.

`pill.py` is the brain itself — attention and backprop in ~200 lines of numpy,
gradients verified against finite differences at **3.5e-10** relative error.

`spore.py` is the same organism with **zero dependencies**: one file, standard
library only. Copy it to any machine with `python3` and it lives. Smaller brain,
because pure Python is ~7x slower.

## The task

Induction. A sequence of distinct tokens whose final token appeared earlier;
predict what followed it. This cannot be solved by looking at one position — it
needs a head that reports *what came before this token* and a second head that
copies. It is the smallest task where attention depth is causal, not decorative.

Measured ablation, identical budget and data:

| model | held-out accuracy |
|---|---|
| 2 layers, d=32 | **1.000** |
| 1 layer, d=32 | 0.141 |

One layer cannot learn it. That is the point of evolving layers.

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

## Predation

A hunter pays 4 energy, takes 45% of a neighbour's energy, and **adopts the
victim's brain** — whole weights, architecture and all. Two rules keep it honest:
prey must be foreign (different architecture, no shared ancestor), and a hunter
forages at 60% in the generation it hunts.

In practice hunters starve: once a population converges on one lineage there is
nothing foreign left to eat. Predation is a tax on diversity, and success destroys
diversity.

## Containment

- every organism is written into one population directory and never leaves it
- the evolved code is checked before it runs: arithmetic, comparison, boolean,
  conditional, comprehension, call, name, literal — nothing else
- no attribute access, no imports inside evolved expressions, no assignment
- no sockets anywhere in the organism code; the environment may fetch a corpus,
  the life never does
- a child that mutates itself into nonsense dies instead of running

## Run it

```bash
python3 pill.py                                  # verify gradients, then learn
python3 world_net.py --gens 12 --K 12 --steps 800 # evolve attention
python3 world.py --gens 80 --K 60                # evolve expressions on the boundary
python3 fractal.py                               # render the colony
python3 spore.py --selftest                      # zero-dependency organism
```

## What happened (neural world, 10 generations)

Population 10-14, each organism training 800 steps per generation, measured on
128 fresh sequences it has never seen.

| gen | best | mean | retention | layers |
|---|---|---|---|---|
| 0 | 0.45 | 0.46 | — | 2.0 |
| 2 | 0.52 | 0.49 | 1.00 | 2.0 |
| 4 | 0.62 | 0.59 | 1.00 | 2.0 |
| 6 | 0.68 | 0.68 | 1.00 | 2.0 |
| 8 | 0.81 | 0.72 | 1.00 | 2.0 |
| 9 | 0.70 | 0.72 | 1.00 | 2.0 |

Mean accuracy 0.46 → 0.72 across ten generations, best individual 0.81. Each
generation inherits the trained weights and keeps going — learning accumulates
down the lineage instead of restarting from random.

**Layers never dropped to 1.** Every surviving organism carried two or more
attention layers in every generation. One-layer mutants are born, score near
chance, and starve. The architecture is not decoration; the environment audits
it every generation.

## Honest limits


Written down rather than hidden, because a benchmark that lies is worse than no
benchmark:

1. **Pure Python trades reach for brain size.** The dependency-free organism runs
   anywhere but its model is too small to crack induction (0.20 held-out at d=8).
   Portability and capability currently fight.
2. **Retention flatters itself.** The replay buffer's own examples score high
   because they were trained on. Held-out accuracy is the number that counts.
3. **Metabolism does not track habitat** in the expression world
   (Pearson = −0.04). Theory says rich ground selects heavy `eff`; the data
   disagrees and the two candidate explanations are untested.
4. **Horizontal gene transfer is dead on arrival.** Zero hunts in ten
   generations, and it is my bug: prey is defined as an organism with a
   *different architecture*. The population converges on one architecture within
   a generation, so no legal prey ever exists and the predator rule never fires.
   In the expression world the comparison was against the *code*, which is
   fine-grained. Fixing this means comparing weights, not architecture shapes.
5. **The run was cut at generation 10 of 12** by its own process timeout, not by
   extinction. The trend is real; the last two generations are missing.

MIT licensed.
