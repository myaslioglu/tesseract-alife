# HCS world — self-replicating code organisms

A small artificial-life system. An organism is a Python file that reads its own
source, decodes the genome written inside itself, solves the environment's
challenge to earn energy, and — if it can afford it — **rewrites itself with
mutations and writes a child to disk**. The copying happens inside the organism,
not in the harness.

Nothing here can leave its own folder. The only thing mutation is allowed to
touch is one arithmetic expression, and that expression is checked against an
AST whitelist before it is ever compiled. A child that edits itself into
nonsense dies on the spot. That is not a safety wrapper bolted on top — it is
the selection pressure.

---

## Three files

| file | role |
|---|---|
| `world.py` | the petri dish: physics, predation, selection, logging, report |
| `fractal.py` | renders the habitat and overlays where the population lived |
| `pop/genNNN/*.py` | the organisms themselves, one file per individual |

## The genome

Every organism carries a base64 JSON blob on a single line, `_G = "..."`.
Mutating an organism means rewriting that line — the file rewrites its own
source text.

| gene | range | what it does |
|---|---|---|
| `strategy` | code ≤ 240 chars | the expression that solves the challenge |
| `cr`, `ci` | complex plane | the habitat coordinate; fertility is read from it |
| `mu` | 0.02 – 0.60 | copy fidelity — how often a child is edited |
| `eff` | 0.02 – unbounded | metabolic efficiency: boosts gain, costs upkeep |
| `brood` | 1 – 3 | children per reproduction, each with a smaller share of energy |
| `diet` | 0.0 – 1.0 | chance per generation of hunting a neighbour |

## Physics

**Habitat.** Fertility at `(cr, ci)` is the escape time of the Mandelbrot
iteration, scaled into `[0.30, 2.75]`:

```
fertility = 0.35 + 2.40 * (escape_iterations / maxiter)
interior (never escapes) = 0.30        # stable, but nothing happens
```

`maxiter` grows by 3 every generation (60 → 297). Points that were frozen
interior at generation 0 become coastline later, so new territory opens mid-run.

**Energy.**

```
gain = pts * fertility * eff ** 0.5     # hard diminishing returns
cost = 6.0 * eff ** 2                   # upkeep is quadratic
energy_next = energy + round(gain) - round(cost)
```

`eff` has no upper bound. Its optimum falls out of the curve:

```
eff* = (pts * fertility / 24) ** (2/3)
```

so a strong solver on rich ground is pushed to a heavy metabolism, and a weak
solver on poor ground to a lean one. Nothing clamps the gene; the curve does.

`pts` is 2 per correct case plus a 4-point bonus for a perfect run. The optimum
of `eff` therefore depends on how good a solver you are and how fertile your
patch is: in poor ground a lean metabolism wins, in rich ground a heavier one
does. There is no ramp to a ceiling — but see "Known lies" below.

**Predation.** A hunter pays 4 energy, takes 45% of a neighbour's energy, and
adopts the victim's code with a digestion error half the time. That is
horizontal gene transfer: good code can jump between lineages without descent.

Two rules keep it honest. A hunter only eats **foreign** code — the prey must
carry a different strategy and share no ancestor in its lineage record. If
everyone around is kin, the hunter goes hungry rather than inbreed. And a hunter
forages badly in the generation it hunts: its case score is cut to 60%.

**Selection.** Population is held at an optimum `K` (default 30) by logistic
selection, not a hard cap: the number of survivors is `K / mean brood`, so the
next generation lands near `K`. Organisms below the reproduction cost starve
and are moved to `dead/`.

## Running it

```bash
python3 world.py --gens 80 --K 30 --soup 20     # the simulation
python3 fractal.py                              # poster + animation from the log
```

Outputs: `world.jsonl` (per-generation record), `positions.jsonl` (every
organism's coordinate, every generation), `report.md` (the table),
`dead/` (corpses), and from the renderer
`output/mandelbrot_colony.png` + `.gif`.

## Mutation operators

Character-level edits on the strategy string only: span cut, adjacent swap,
insert, delete, one to three per event. Quantitative genes drift by Gaussian
noise; the habitat coordinate crawls by 0.004 per birth, with a 0.05 leap about
one time in twelve.

## The AST whitelist

Before anything runs:

* node types restricted to arithmetic, comparison, boolean, conditional,
  comprehension, call, name, literal
* names restricted to `c`, `x`, `y` plus `sum, len, min, max, abs, sorted,
  range, all, any`
* no attribute access, no imports, no lambdas, no assignment

Anything else raises, the organism scores zero, and it dies of starvation.

## What actually happened (80 generations, v6)

* population grew and held: 24 → 59 against `K = 60`, range 24–63
* **the task was solved**: first perfect 6/6 at generation 10, and 70 of the 80
  generations carried a perfect solver. Under bounded `eff` the same system
  never got past 5/6 in 160 generations of trying
* `eff` stopped pinning: final distribution 0.36 – 1.09, mean 0.65, continuous
  and interior to the range. No wall, either end
* predation collapsed on its own: 44 successful hunts against **754 refusals**
  — hunters found nothing but kin. The colony went clonal (58 of 62 organisms
  carry the identical strategy), and a predator that refuses to eat its own
  kind starves in a clone field

## Known lies

1. **Metabolism does not track habitat.** Theory says rich ground selects heavy
   `eff`. Data says otherwise: Pearson(fertility, eff) = **−0.04**. No spatial
   sorting at all. Either migration is fast enough to erase local adaptation, or
   the energy surplus is so large that the optimum is flat across the whole
   occupied range. I have not separated those two.
2. **Diversity dies early.** `mu` falls to 0.068 and the population becomes a
   clone army by mid-run. Everything after generation 10 is one codebase
   competing with itself, which is why predation had nothing left to eat.
3. **No proof of horizontal transfer mattering.** 44 hunts moved code; none of
   them demonstrably changed an outcome.
4. **`diet` still climbs** to 0.24 with no measured cost beyond the 60% foraging
   cut. The penalty exists now, but I never showed it binds.
