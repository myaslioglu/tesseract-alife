# evolve-3: survival that can see a better brain

evolve-2 ([EVOLVE2.md](EVOLVE2.md)) found that selection could lose its best
brains. evolve-3 finds out why, changes what survival ranks, and repeats the
comparison at twenty seeds per arm.

## Why the best brains were lost

Seed 9 of evolve-2 was rerun with every organism logged
(`net_organisms.jsonl`; the rerun reproduces the original run exactly over 25
generations). In the three-hop era, near-perfect brains appeared three times
and were lost each time. Their energy shows why:

| generation | solver's accuracy | its habitat's fertility | its energy | rank | survivors' accuracy | survivors' fertility |
|---|---|---|---|---|---|---|
| 20 | 0.99 | 0.30 | 3 | 8 of 14 | 0.36 to 0.62 | 0.85 to 1.21 |
| 23 | 1.00, 1.00 | 0.63, 0.63 | 16, 15 | 6 and 7 of 14 | 0.54 to 0.70 | 1.47 to 2.40 |

(In generation 22 a solver at 0.99 did survive, ranked third; its two children
are the generation-23 pair.) Inherited energy was small for everyone (2 to 11)
and did not decide anything. The habitat did: energy gained in a life is
points times fertility, and fertility runs from 0.30 to 2.75, so a brain on
rich ground with half the accuracy outranks a perfect brain on poor ground.

## The change

`--rank score`: survival ranks the points a brain earned this life (accuracy
and retention), with energy only breaking ties. The habitat still feeds each
brain's energy, energy is still split among its children, and a brain whose
energy falls below the cost of reproduction still starves. With the default
(`--rank energy`) the world is unchanged; seed 7 still reproduces the paper's
run exactly.

## Plan, fixed before the twenty-seed runs finished

Arms, seeds 7 to 26 (20 per arm), every other setting as in evolve-2
(30 generations, eras 3/12/15, K = 12, 800 steps per life, momentum 0.9,
architecture mutation 0.3, up to 6 layers), every organism logged:

- **evolve**: `--rank score`;
- **evolve, energy rule**: `--rank energy`, the evolve-2 rule (seeds 7 to 11
  are the evolve-2 runs, which this code reproduces);
- **drift**: survivors drawn by lottery;
- **one brain, 2 layers** and **one brain, 4 layers**.

A run *recovers* from an era change when its population's mean accuracy on the
era's task reaches 0.9 at any generation of that era.

**Primary test.** Recovery from the step to two hops, evolve against drift:
one-sided Fisher exact test on the counts of recovered runs, at the 0.05
level.

**Secondary, reported as such.**
1. The same for three hops.
2. Mean accuracy over the era's last three generations, evolve against drift,
   one-sided permutation test on the difference in means (100,000 random
   splits), for each era.
3. Recovery counts, evolve against the energy rule (does ranking by score help?).
4. Solver retention: the share of brains at 0.95 accuracy or more that survive
   to reproduce, in each arm.
5. Generations from the first solver to recovery.
6. Evolve against one brain born with four layers: recovery counts and brain
   lives spent.

The result counts against selection if evolve does not recover from the step
to two hops more often than drift.
