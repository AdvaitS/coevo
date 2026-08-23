# coevo

**Surrogate-assisted & coevolutionary evolutionary computation for expensive black-box problems.**

`coevo` is a small, dependency-light research library of evolutionary algorithms
(GA, DE, PSO) whose distinctive feature is a family of *surrogate-assisted*
strategies that learn to predict fitness — so the optimizer can spend its
evaluation budget where it matters, not on every candidate in every generation.

The idea is rooted in the coevolved fitness predictors of Schmidt & Lipson
([*"Coevolution of Fitness Predictors"*, IEEE TEVC 2008](https://ieeexplore.ieee.org/document/4475399)):
instead of approximating the whole fitness landscape, a predictor *co-evolves*
with the population and specializes to the region it is currently exploring.

## Why surrogate-assisted evolution?

Most evolutionary-algorithm libraries assume fitness is cheap — evaluate everyone,
every generation. But in the problems that matter (a wet-lab simulation, a
protein-fitness oracle, a full model-training run), **the fitness evaluation *is*
the bottleneck**. `coevo` replaces most of those expensive evaluations with cheap
predictions, re-fitting a predictor that tracks the population as it moves.

## Install

```bash
pip install -e ".[dev]"   # from a local checkout
```

Requires only `numpy` and `scipy` (Python ≥ 3.10).

## Quickstart

```python
from coevo import (
    CoevolvedPredictor,
    DifferentialEvolution,
    NearestNeighborSurrogate,
    ParticleSwarmOptimization,
    SurrogateEvaluator,
    benchmarks,
)

# Exact evaluation (the classic way).
problem = benchmarks.rastrigin(dim=10)
result = DifferentialEvolution(pop_size=50, generations=300).optimize(problem)
print(result.summary())          # rastrigin: best=... | gap-to-optimum ... | true_evals=...

# Surrogate-assisted: predict fitness, truly evaluate only the most promising.
sa = SurrogateEvaluator(
    benchmarks.rastrigin(dim=10),
    CoevolvedPredictor(NearestNeighborSurrogate()),
    eval_fraction=0.25,
)
sa_result = DifferentialEvolution(pop_size=50, generations=300).optimize(
    benchmarks.rastrigin(dim=10), sa
)
print(sa_result.summary())       # far fewer true_evals, comparable best
```

## What's inside

| Component | Description |
|---|---|
| `DifferentialEvolution` | DE/rand/1/bin with synchronous (batch) updates and budget limiting |
| `GeneticAlgorithm` | real-coded GA: tournament selection, arithmetic crossover, Gaussian mutation, elitism |
| `ParticleSwarmOptimization` | canonical PSO with Clerc constriction coefficients |
| `NSGA2` | non-dominated sorting GA-II (SBX + polynomial mutation) for multi-objective problems |
| `TrueEvaluator` | exact objective evaluation, counts `n_true` |
| `SurrogateEvaluator` | model management: `individual` (pre-selection) or `generation` strategies; retrains a coevolved predictor on the true-eval archive |
| `SurrogateMultiObjectiveEvaluator` | surrogate-assisted pre-selection for multi-objective problems (one surrogate per objective) |
| `MultiObjectiveProblem` / `MultiObjectiveResult` | multi-objective problem/result types with Pareto-front reporting |
| `metrics` | `nondominated_mask`, `fast_non_dominated_sort`, `crowding_distance`, `igd`, `hypervolume` |
| `NearestNeighborSurrogate` | dependency-free 1-NN baseline predictor (bounded) |
| `RBFSurrogate` | thin-plate RBF interpolator (scipy) |
| `GaussianProcessSurrogate` | GP regression (scikit-learn, optional) |
| `RandomForestSurrogate` | RF regression (scikit-learn, optional) |
| `ClippedPredictor` | bounds any surrogate's predictions to the observed fitness range |
| `CoevolvedPredictor` | wraps a surrogate and records how well it tracks the population (`error_trace`) |
| `SymbolicRegressor` / `EvolvedPredictor` | evolves a compact, *interpretable* fitness-prediction expression via genetic programming (Schmidt & Lipson, 2008) |
| `benchmarks` | sphere, rastrigin, rosenbrock, ackley, griewank, shifted/noisy variants |

## Benchmarks

Run the reproducible benchmark (median over seeds):

```bash
python benchmarks/run_benchmark.py --generations 200 --seeds 3       # generations mode
python benchmarks/run_benchmark.py --budget 5000 --seeds 3          # budget-normalized mode
```

Sample output (`pop_size=50`, `generations=200`, `seeds=3`):

| algorithm | problem | evaluator | best fitness (median) | true evals |
|---|---|---|---|---|
| DE | sphere(5) | exact | 4.841e-09 | 10050 |
| DE | sphere(5) | surrogate (1-NN) | 4.182e-07 | 2798 |
| DE | sphere(5) | surrogate (RBF) | 0.001417 | 2798 |
| DE | sphere(5) | surrogate (RBF, clipped) | 7.11e-06 | 2798 |
| GA | sphere(5) | exact | 5.438e-07 | 9850 |
| GA | sphere(5) | surrogate (1-NN) | 4.233e-08 | 2794 |
| GA | sphere(5) | surrogate (RBF) | 0.4959 | 2794 |
| GA | sphere(5) | surrogate (RBF, clipped) | 5.146e-06 | 2794 |
| PSO | sphere(5) | exact | 1.021e-15 | 10050 |
| PSO | sphere(5) | surrogate (1-NN) | 2.521e-16 | 2798 |
| PSO | sphere(5) | surrogate (RBF) | 2.384e-05 | 2798 |
| PSO | sphere(5) | surrogate (RBF, clipped) | 8.892e-15 | 2798 |
| DE | ackley(5) | exact | 0.0007715 | 10050 |
| DE | ackley(5) | surrogate (1-NN) | 0.01304 | 2798 |
| DE | ackley(5) | surrogate (RBF) | 0.00149 | 2798 |
| DE | ackley(5) | surrogate (RBF, clipped) | 0.00105 | 2798 |
| GA | ackley(5) | exact | 8.535 | 9850 |
| GA | ackley(5) | surrogate (1-NN) | 8.535 | 2794 |
| GA | ackley(5) | surrogate (RBF) | 9.296 | 2794 |
| GA | ackley(5) | surrogate (RBF, clipped) | 8.535 | 2794 |
| PSO | ackley(5) | exact | 4.853e-07 | 10050 |
| PSO | ackley(5) | surrogate (1-NN) | 4.449e-07 | 2798 |
| PSO | ackley(5) | surrogate (RBF) | 2.857e-07 | 2798 |
| PSO | ackley(5) | surrogate (RBF, clipped) | 2.951e-07 | 2798 |
| DE | noisy_ackley(5) | exact | 0.04113 | 10050 |
| DE | noisy_ackley(5) | surrogate (1-NN) | 0.2225 | 2798 |
| DE | noisy_ackley(5) | surrogate (RBF) | 0.2567 | 2798 |
| DE | noisy_ackley(5) | surrogate (RBF, clipped) | 0.1811 | 2798 |
| GA | noisy_ackley(5) | exact | 10.72 | 9850 |
| GA | noisy_ackley(5) | surrogate (1-NN) | 10.84 | 2794 |
| GA | noisy_ackley(5) | surrogate (RBF) | 12.35 | 2794 |
| GA | noisy_ackley(5) | surrogate (RBF, clipped) | 10.65 | 2794 |
| PSO | noisy_ackley(5) | exact | 0.008059 | 10050 |
| PSO | noisy_ackley(5) | surrogate (1-NN) | 0.1755 | 2798 |
| PSO | noisy_ackley(5) | surrogate (RBF) | 0.1853 | 2798 |
| PSO | noisy_ackley(5) | surrogate (RBF, clipped) | 0.1437 | 2798 |

### Budget-normalized comparison

At a matched evaluation budget, the surrogate-assisted runs convert that budget
into *many more* search generations. For example, with `--budget 5000`:

| algorithm | problem | exact | surrogate (1-NN) | surrogate (RBF, clipped) |
|---|---|---|---|---|
| DE | ackley(5) | 0.393 | 5.64e-06 | 1.18e-06 |
| PSO | sphere(5) | 1.1e-08 | 1.19e-28 | 3.15e-22 |

The surrogate-assisted optimiser reaches solutions orders of magnitude better
than exact evaluation at the *same* true-evaluation cost.

### What this shows

1. **Surrogate-assisted runs spend ~72% fewer true evaluations** — on expensive
   objectives, that is a ~3.5× cost saving.
2. **The bounded 1-NN predictor is a robust baseline** — it matches (and often
   beats) exact evaluation on smooth functions.
3. **Unbounded surrogates can be *exploited*** — the thin-plate RBF extrapolates
   wildly, and an aggressive GA chases its imaginary minima (`best=0.49` instead
   of `5e-7` on sphere). This is a well-known surrogate-assisted-evolution
   failure mode.
4. **`ClippedPredictor` is a one-line fix** — bounding predictions to the observed
   fitness range recovers exact-quality results (GA sphere `0.4959 → 5.1e-06`)
   while keeping the evaluation savings.
5. **Noisy objectives are harder** — surrogates still save evaluations, but noise
   degrades predictions, so exact evaluation retains an edge in final fitness.

These caveats are the interesting part: `coevo` is built to make them easy to
*measure and study*, not to hide them.

## Multi-objective optimization

`NSGA2` (SBX + polynomial mutation) optimizes multi-objective problems, and
`SurrogateMultiObjectiveEvaluator` applies the same surrogate-assisted
pre-selection (one predictor per objective). Run the MO benchmark with
`python benchmarks/run_benchmark.py --mo`:

| problem | evaluator | IGD (median) | true evals |
|---|---|---|---|
| zdt1 | exact | 0.0051 | 25100 |
| zdt1 | surrogate (1-NN) | 0.0057 | 7880 |
| zdt2 | exact | 0.0049 | 25100 |
| zdt2 | surrogate (1-NN) | 0.0053 | 7880 |
| zdt3 | exact | 0.19 | 25100 |
| zdt3 | surrogate (1-NN) | 0.19 | 7880 |

The bounded 1-NN surrogate matches exact IGD on ZDT1/ZDT2 while spending ~69%
fewer true evaluations — the same surrogate-assisted story, on Pareto fronts.

## Roadmap

- Trust-region and uncertainty-aware model management (GP variance).
- Incremental predictor coevolution — evolve the predictor *population* continuously (rather than re-evolving from scratch each generation).
- Real expensive applications: neural-architecture search and protein-fitness oracles.

## Design principles

- **Minimal dependencies** — `numpy` + `scipy` only; no ML framework required.
- **Batch-first** — algorithms evaluate whole generations at once, which is what
  makes surrogate pre-selection natural.
- **Reproducible** — every algorithm and problem takes an explicit `seed`.
- **Research-log friendly** — results expose `history`, `true_evaluations`, and
  predictor diagnostics for plotting in `docs/research/`.

## License

MIT — see [LICENSE](LICENSE).
