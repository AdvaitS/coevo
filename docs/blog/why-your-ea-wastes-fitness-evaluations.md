# Why your evolutionary algorithm is wasting fitness evaluations

*And how surrogate-assisted evolution spends the same budget ~3.5× smarter.*

---

Most people run evolutionary algorithms the way textbooks from 1995 taught them:
generate a population, **evaluate every individual**, select, repeat. That is
perfectly fine when fitness is a cheap function like `sum(x**2)`. But the moment
fitness costs real money — a computational-fluid-dynamics simulation, a
wet-lab assay, a full neural-network training run — the textbook approach stops
making sense, because you spend the entire budget evaluating candidates that
will never survive selection.

The fix has been known since Schmidt & Lipson's
[*"Coevolution of Fitness Predictors"* (IEEE TEVC 2008)](https://ieeexplore.ieee.org/document/4475399):
**stop paying for fitness you don't need.** Learn a cheap *predictor* of fitness,
evaluate only the promising candidates for real, and — here's the key insight —
let that predictor *co-evolve* with the population, specializing to the region
being searched rather than trying to model the whole landscape.

This is the idea behind [`coevo`](https://github.com/AdvaitS/coevo), a small
dependency-light library I'm building to make surrogate-assisted evolutionary
computation easy to *measure and study*.

## The experiment

Three classic algorithms (differential evolution, a genetic algorithm, particle
swarm) × three problems (sphere, ackley, noisy ackley), each run two ways:

1. **exact** — evaluate every candidate against the true objective;
2. **surrogate** — predict fitness with a model, truly evaluate only the most
   promising 25%, and re-fit the predictor on the archive of true evaluations.

The headline result, across the board:

> **Surrogate-assisted runs reach comparable optima while spending ~72% fewer
> true evaluations.**

Here's the reproducible table (median over 3 seeds, `pop_size=50`,
`generations=200`):

| algorithm | problem | evaluator | best fitness | true evals |
|---|---|---|---|---|
| DE | sphere(5) | exact | 4.8e-09 | 10050 |
| DE | sphere(5) | surrogate (1-NN) | 4.2e-07 | 2798 |
| PSO | sphere(5) | exact | 1.0e-15 | 10050 |
| PSO | sphere(5) | surrogate (1-NN) | 2.5e-16 | 2798 |

But "fewer evaluations at the same generation count" understates the win,
because on an *expensive* problem you don't fix the generation count — you fix
the **budget**. So I added a budget-normalized comparison: run every config
until it has spent the same number of true evaluations, then compare fitness.

| algorithm | problem | exact @ 5000 evals | surrogate @ 5000 evals |
|---|---|---|---|
| DE | ackley(5) | 0.393 | **5.6e-06** |
| PSO | sphere(5) | 1.1e-08 | **1.2e-28** |

The surrogate-assisted optimiser turns the same budget into many more search
generations, and lands orders of magnitude closer to the optimum. That is the
whole business case for surrogate assistance in one table.

## The part nobody warns you about: surrogate exploitation

If I'd stopped there, this post would be an ad. The interesting result is a
failure mode.

The thin-plate-spline (RBF) surrogate is an exact interpolator, which sounds
great — but it extrapolates *unboundedly* away from its training data. An
aggressive optimizer notices this and does the rational thing: it drives the
population toward the regions where the surrogate claims fitness is
`-6.7 × 10⁷`.

Those regions are fictional. The optimizer has learned to exploit the surrogate,
not the problem:

| algorithm | problem | exact | surrogate (RBF) |
|---|---|---|---|
| GA | sphere(5) | 5.4e-07 | **0.49** |

This is a well-documented failure of surrogate-assisted evolution, and it's
exactly what the "model management" literature exists to fix. The cheapest fix
is embarrassingly simple: **clip predictions to the observed fitness range**.
Bound the predictor and it can't invent `-∞` for you:

| algorithm | problem | exact | surrogate (RBF) | surrogate (RBF, clipped) |
|---|---|---|---|---|
| GA | sphere(5) | 5.4e-07 | 0.49 | **5.1e-06** |

One `np.clip` later, the exploited run recovers exact-quality results while
keeping the ~72% evaluation savings.

## Takeaways

1. If your fitness is expensive, **your optimizer is the bottleneck**, not the
   search operator — stop paying for every evaluation.
2. A bounded predictor (even a dumb 1-NN) is a *robust* surrogate; a fancy
   unbounded interpolator (RBF/GP) needs **model management** or it will be
   exploited.
3. The interesting research questions live in that gap — trust regions,
   uncertainty-aware acquisition, and *evolving the predictor itself*.

`coevo` is a work in progress built to make these experiments one-liners. If
surrogate-assisted evolution, neuroevolution, or interpretable symbolic
discovery interests you, the repo is
[here](https://github.com/AdvaitS/coevo) — issues and PRs welcome.

---

*Cross-post me to dev.to / Hashnode; run the numbers yourself with
`python benchmarks/run_benchmark.py --budget 5000`.*
