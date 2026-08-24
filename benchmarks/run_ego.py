"""EGO against the evolutionary algorithms at matched true-evaluation budgets.

Efficient global optimization (Jones, Schonlau & Welch 1998) is the reference
method for expensive black-box problems. Any surrogate-assisted EA needs to say
where it stands against it -- and, just as usefully, where it does not.

Usage
-----
    python benchmarks/run_ego.py --seeds 5
    python benchmarks/run_ego.py --budgets 50 100 250 500
"""

from __future__ import annotations

import argparse
import time
from statistics import median

from coevo import (
    CoevolvedPredictor,
    DifferentialEvolution,
    EfficientGlobalOptimization,
    GeneticAlgorithm,
    NearestNeighborSurrogate,
    ParticleSwarmOptimization,
    SurrogateEvaluator,
    TrueEvaluator,
    benchmarks,
)

PROBLEMS = [
    ("sphere(5)", lambda: benchmarks.sphere(5)),
    ("ackley(5)", lambda: benchmarks.ackley(5)),
    ("rastrigin(5)", lambda: benchmarks.rastrigin(5)),
]
EAS = {
    "DE": DifferentialEvolution,
    "GA": GeneticAlgorithm,
    "PSO": ParticleSwarmOptimization,
}


def _run(kind: str, make_problem, budget: int, seed: int, pop_size: int):
    problem = make_problem()
    started = time.perf_counter()
    if kind == "EGO":
        result = EfficientGlobalOptimization(
            n_initial=min(10, max(2, budget // 3)), max_evaluations=budget, seed=seed
        ).optimize(problem, TrueEvaluator(problem))
    elif kind.endswith("+1NN"):
        algo = EAS[kind.split("+")[0]]
        evaluator = SurrogateEvaluator(
            problem,
            CoevolvedPredictor(NearestNeighborSurrogate()),
            eval_fraction=0.25,
            warmup=3,
            archive_size=100,
        )
        result = algo(
            pop_size=pop_size, generations=10_000_000, seed=seed, max_evaluations=budget
        ).optimize(problem, evaluator)
    else:
        result = EAS[kind](
            pop_size=pop_size, generations=10_000_000, seed=seed, max_evaluations=budget
        ).optimize(problem, TrueEvaluator(problem))
    return result.best_fitness, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--pop-size", type=int, default=10)
    parser.add_argument("--budgets", type=int, nargs="+", default=[50, 100, 250])
    args = parser.parse_args()

    kinds = ["EGO", "DE", "GA", "PSO", "DE+1NN"]
    print(f"# coevo: EGO vs evolutionary search at matched budget "
          f"(median of {args.seeds} seeds, pop_size={args.pop_size})")
    print()

    for name, make_problem in PROBLEMS:
        print(f"## {name}")
        print("| budget | " + " | ".join(kinds) + " | EGO secs |")
        print("|" + "---|" * (len(kinds) + 2))
        for budget in args.budgets:
            cells, ego_secs = [], 0.0
            for kind in kinds:
                runs = [_run(kind, make_problem, budget, s, args.pop_size) for s in range(args.seeds)]
                cells.append(f"{median(v for v, _ in runs):.4g}")
                if kind == "EGO":
                    ego_secs = median(t for _, t in runs)
            print(f"| {budget} | " + " | ".join(cells) + f" | {ego_secs:.1f} |")
        print()

    print("EGO is sequential -- one true evaluation per iteration, each informed by")
    print("every point before it -- while the EAs evaluate a generation at a time. At a")
    print("matched budget EGO therefore extracts more per evaluation and wins at small")
    print("budgets, and pays for it with cubic GP scaling and no parallelism.")
if __name__ == "__main__":
    main()
