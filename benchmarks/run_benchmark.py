"""Reproducible benchmark: plain vs. surrogate-assisted evolutionary algorithms.

For each (algorithm, problem) pair we run both a fully-exact evaluator and a
surrogate-assisted one (with a coevolved RBF predictor), and report the median
best fitness across seeds together with the number of *true* evaluations spent.

Usage
-----
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --generations 300 --seeds 5
"""

from __future__ import annotations

import argparse
from statistics import median

from coevo import (
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

POP_SIZE = 50


def _median_best(algo_cls, make_problem, seeds, generations, surrogate):
    bests, evals = [], []
    for seed in seeds:
        problem = make_problem()
        algo = algo_cls(pop_size=POP_SIZE, generations=generations, seed=seed)
        if surrogate is None:
            evaluator = TrueEvaluator(problem)
        else:
            evaluator = SurrogateEvaluator(
                problem, surrogate, eval_fraction=0.25, warmup=5, archive_size=100
            )
        result = algo.optimize(problem, evaluator)
        bests.append(result.best_fitness)
        evals.append(result.true_evaluations)
    return median(bests), int(median(evals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    problems = [
        ("sphere(5)", lambda: benchmarks.sphere(5)),
        ("ackley(5)", lambda: benchmarks.ackley(5)),
        ("noisy_ackley(5)", lambda: benchmarks.noisy(benchmarks.ackley(5), sigma=0.1)),
    ]

    print(
        f"# coevo benchmark (generations={args.generations}, seeds={args.seeds}, pop_size={POP_SIZE})"
    )
    print()
    print("| algorithm | problem | evaluator | best fitness (median) | true evals |")
    print("|---|---|---|---|---|")

    for name, make_problem in problems:
        for algo_name, algo_cls in ALGORITHMS.items():
            for label, surrogate in [
                ("exact", None),
                ("surrogate (1-NN)", CoevolvedPredictor(NearestNeighborSurrogate())),
                ("surrogate (RBF)", CoevolvedPredictor(RBFSurrogate())),
            ]:
                best, evals = _median_best(
                    algo_cls, make_problem, seeds, args.generations, surrogate
                )
                print(
                    f"| {algo_name} | {name} | {label} | {best:.4g} | {evals} |"
                )


if __name__ == "__main__":
    main()
