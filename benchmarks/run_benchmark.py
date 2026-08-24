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
    NSGA2,
    ParticleSwarmOptimization,
    RBFSurrogate,
    SurrogateEvaluator,
    SurrogateMultiObjectiveEvaluator,
    TrueEvaluator,
    benchmarks,
)
from coevo.core.metrics import igd

ALGORITHMS = {
    "DE": DifferentialEvolution,
    "GA": GeneticAlgorithm,
    "PSO": ParticleSwarmOptimization,
}

# Factories, not instances. A shared predictor object would accumulate
# `error_trace` across every algorithm x problem x seed in the sweep, making the
# library's own diagnostic meaningless, and would leak state into any future
# surrogate that keeps more than the last fit.
SURROGATES = [
    ("exact", None),
    ("surrogate (1-NN)", lambda: CoevolvedPredictor(NearestNeighborSurrogate())),
    ("surrogate (RBF)", lambda: CoevolvedPredictor(RBFSurrogate())),
    ("surrogate (RBF, clipped)", lambda: CoevolvedPredictor(ClippedPredictor(RBFSurrogate()))),
]

POP_SIZE = 50


def _make_evaluator(problem, factory):
    if factory is None:
        return TrueEvaluator(problem)
    return SurrogateEvaluator(
        problem, factory(), eval_fraction=0.25, warmup=5, archive_size=100
    )


def _median_best(algo_cls, make_problem, seeds, generations, factory):
    bests, evals = [], []
    for seed in seeds:
        problem = make_problem()
        algo = algo_cls(pop_size=POP_SIZE, generations=generations, seed=seed)
        evaluator = _make_evaluator(problem, factory)
        result = algo.optimize(problem, evaluator)
        bests.append(result.best_fitness)
        evals.append(result.true_evaluations)
    return median(bests), int(median(evals))


def _median_best_at_budget(algo_cls, make_problem, seeds, budget, factory):
    bests, evals = [], []
    for seed in seeds:
        problem = make_problem()
        algo = algo_cls(
            pop_size=POP_SIZE, generations=10_000_000, seed=seed, max_evaluations=budget
        )
        evaluator = _make_evaluator(problem, factory)
        result = algo.optimize(problem, evaluator)
        bests.append(result.best_fitness)
        evals.append(result.true_evaluations)
    return median(bests), int(median(evals))


def _run_mo_benchmark(generations: int, seeds: int) -> None:
    problems = [
        ("zdt1", benchmarks.zdt1, benchmarks.zdt1_front),
        ("zdt2", benchmarks.zdt2, benchmarks.zdt2_front),
        ("zdt3", benchmarks.zdt3, benchmarks.zdt3_front),
    ]
    surrogates = [
        ("exact", None),
        ("surrogate (1-NN)", lambda: CoevolvedPredictor(NearestNeighborSurrogate())),
        ("surrogate (RBF, clipped)", lambda: CoevolvedPredictor(ClippedPredictor(RBFSurrogate()))),
    ]
    print(f"# coevo multi-objective benchmark (generations={generations}, seeds={seeds})")
    print()
    print("| problem | evaluator | IGD (median) | true evals |")
    print("|---|---|---|---|")
    for name, make, make_front in problems:
        ref = make_front()
        for label, factory in surrogates:
            igds, evals = [], []
            for seed in range(seeds):
                problem = make(dim=10)
                evaluator = (
                    SurrogateMultiObjectiveEvaluator(
                        problem, factory, eval_fraction=0.3, warmup=5
                    )
                    if factory is not None
                    else None
                )
                result = NSGA2(pop_size=100, generations=generations, seed=seed).optimize(
                    problem, evaluator
                )
                igds.append(igd(result.objectives, ref))
                evals.append(result.true_evaluations)
            print(f"| {name} | {label} | {median(igds):.4g} | {int(median(evals))} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--mo", action="store_true", help="run the multi-objective benchmark")
    args = parser.parse_args()

    if args.mo:
        _run_mo_benchmark(args.generations, args.seeds)
        return

    seeds = list(range(args.seeds))
    problems = [
        ("sphere(5)", lambda: benchmarks.sphere(5)),
        ("ackley(5)", lambda: benchmarks.ackley(5)),
        ("noisy_ackley(5)", lambda: benchmarks.noisy(benchmarks.ackley(5), sigma=0.1)),
    ]

    if args.budget is not None:
        print(f"# coevo benchmark (budget={args.budget} true evals, seeds={args.seeds})")
        print()
        print("| algorithm | problem | evaluator | best fitness @ budget (median) | true evals |")
        print("|---|---|---|---|---|")
        for name, make_problem in problems:
            for algo_name, algo_cls in ALGORITHMS.items():
                for label, factory in SURROGATES:
                    best, evals = _median_best_at_budget(
                        algo_cls, make_problem, seeds, args.budget, factory
                    )
                    print(f"| {algo_name} | {name} | {label} | {best:.4g} | {evals} |")
        return

    print(
        f"# coevo benchmark (generations={args.generations}, seeds={args.seeds}, pop_size={POP_SIZE})"
    )
    print()
    print("| algorithm | problem | evaluator | best fitness (median) | true evals |")
    print("|---|---|---|---|---|")
    for name, make_problem in problems:
        for algo_name, algo_cls in ALGORITHMS.items():
            for label, factory in SURROGATES:
                best, evals = _median_best(
                    algo_cls, make_problem, seeds, args.generations, factory
                )
                print(f"| {algo_name} | {name} | {label} | {best:.4g} | {evals} |")


if __name__ == "__main__":
    main()
