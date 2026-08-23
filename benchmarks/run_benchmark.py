"""Reproducible benchmark: plain vs. surrogate-assisted evolutionary algorithms.

Two modes:

* **generations mode** (default) — run every config for a fixed number of
  generations and report the median best fitness together with the number of
  *true* evaluations spent.
* **budget mode** (``--budget N``) — run every config until it has spent ~N true
  evaluations, comparing final fitness at a matched evaluation budget.

Usage
-----
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --budget 5000 --seeds 5
"""

from __future__ import annotations

import argparse
from statistics import median

from coevo import (
    ClippedPredictor,
    CoevolvedPredictor,
    DifferentialEvolution,
    GeneticAlgorithm,
    NearestNeighborSurrogate,
    ParticleSwarmOptimization,
    RBFSurrogate,
    SurrogateEvaluator,
    TrueEvaluator,
    benchmarks,
)

ALGORITHMS = {
    "DE": DifferentialEvolution,
    "GA": GeneticAlgorithm,
    "PSO": ParticleSwarmOptimization,
}

SURROGATES = [
    ("exact", None),
    ("surrogate (1-NN)", CoevolvedPredictor(NearestNeighborSurrogate())),
    ("surrogate (RBF)", CoevolvedPredictor(RBFSurrogate())),
    ("surrogate (RBF, clipped)", CoevolvedPredictor(ClippedPredictor(RBFSurrogate()))),
]

POP_SIZE = 50


def _make_evaluator(problem, surrogate):
    if surrogate is None:
        return TrueEvaluator(problem)
    return SurrogateEvaluator(
        problem, surrogate, eval_fraction=0.25, warmup=5, archive_size=100
    )


def _median_best(algo_cls, make_problem, seeds, generations, surrogate):
    bests, evals = [], []
    for seed in seeds:
        problem = make_problem()
        algo = algo_cls(pop_size=POP_SIZE, generations=generations, seed=seed)
        evaluator = _make_evaluator(problem, surrogate)
        result = algo.optimize(problem, evaluator)
        bests.append(result.best_fitness)
        evals.append(result.true_evaluations)
    return median(bests), int(median(evals))


def _median_best_at_budget(algo_cls, make_problem, seeds, budget, surrogate):
    bests, evals = [], []
    for seed in seeds:
        problem = make_problem()
        algo = algo_cls(
            pop_size=POP_SIZE, generations=10_000_000, seed=seed, max_evaluations=budget
        )
        evaluator = _make_evaluator(problem, surrogate)
        result = algo.optimize(problem, evaluator)
        bests.append(result.best_fitness)
        evals.append(result.true_evaluations)
    return median(bests), int(median(evals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--budget", type=int, default=None)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    problems = [
        ("sphere(5)", lambda: benchmarks.sphere(5)),
        ("ackley(5)", lambda: benchmarks.ackley(5)),
        ("noisy_ackley(5)", lambda: benchmarks.noisy(benchmarks.ackley(5), sigma=0.1)),
    ]

    if args.budget is not None:
        print(f"# coevo benchmark (budget={args.budget} true evals, seeds={args.seeds})")
        print()
        print("| algorithm | problem | evaluator | best fitness @ budget (median) |")
        print("|---|---|---|---|")
        for name, make_problem in problems:
            for algo_name, algo_cls in ALGORITHMS.items():
                for label, surrogate in SURROGATES:
                    best, _ = _median_best_at_budget(
                        algo_cls, make_problem, seeds, args.budget, surrogate
                    )
                    print(f"| {algo_name} | {name} | {label} | {best:.4g} |")
        return

    print(
        f"# coevo benchmark (generations={args.generations}, seeds={args.seeds}, pop_size={POP_SIZE})"
    )
    print()
    print("| algorithm | problem | evaluator | best fitness (median) | true evals |")
    print("|---|---|---|---|---|")
    for name, make_problem in problems:
        for algo_name, algo_cls in ALGORITHMS.items():
            for label, surrogate in SURROGATES:
                best, evals = _median_best(
                    algo_cls, make_problem, seeds, args.generations, surrogate
                )
                print(f"| {algo_name} | {name} | {label} | {best:.4g} | {evals} |")


if __name__ == "__main__":
    main()
