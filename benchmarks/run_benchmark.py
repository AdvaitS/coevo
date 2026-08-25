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
import time
from statistics import median

import numpy as np

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


def _iqr(values) -> str:
    """Interquartile range as a compact string.

    A median with no dispersion cannot be argued with -- two methods whose
    medians differ by 5% may or may not be distinguishable, and the reader has
    no way to tell. The IQR is the cheapest honest answer.
    """
    q1, q3 = np.percentile(np.asarray(values, dtype=float), [25, 75])
    return f"{q1:.4g}–{q3:.4g}"


def _sample(algo_cls, make_problem, seeds, factory, generations=None, budget=None):
    """Every seed's result, not just the median -- dispersion and tests need the sample."""
    bests, evals, secs = [], [], []
    for seed in seeds:
        problem = make_problem()
        if budget is not None:
            algo = algo_cls(
                pop_size=POP_SIZE, generations=10_000_000, seed=seed, max_evaluations=budget
            )
        else:
            algo = algo_cls(pop_size=POP_SIZE, generations=generations, seed=seed)
        evaluator = _make_evaluator(problem, factory)
        started = time.perf_counter()
        result = algo.optimize(problem, evaluator)
        secs.append(time.perf_counter() - started)
        bests.append(result.best_fitness)
        evals.append(result.true_evaluations)
    return bests, evals, secs


def _paired_test(a, b) -> str:
    """Wilcoxon signed-rank on paired per-seed results, as a printable cell.

    Paired because both arms run the same seeds on the same problems; rank-based
    because optimizer results are not normal and a handful of seeds cannot show
    that they are.
    """
    try:
        from scipy.stats import wilcoxon

        if len(a) < 6 or all(x == y for x, y in zip(a, b)):
            return "n/a"
        return f"{wilcoxon(a, b).pvalue:.3f}"
    except Exception:
        return "n/a"


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
    # 3 seeds cannot support a median, let alone a comparison. 15 is the
    # smallest sample at which the Wilcoxon test below can reach p<0.05.
    parser.add_argument("--seeds", type=int, default=15)
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

    kwargs = (
        {"budget": args.budget} if args.budget is not None else {"generations": args.generations}
    )
    header = (
        f"budget={args.budget} true evals" if args.budget is not None
        else f"generations={args.generations}"
    )
    print(f"# coevo benchmark ({header}, seeds={args.seeds}, pop_size={POP_SIZE})")
    print()
    print("| algorithm | problem | evaluator | best (median) | IQR | true evals | vs true-eval p |")
    print("|---|---|---|---|---|---|---|")

    overhead = []
    for name, make_problem in problems:
        for algo_name, algo_cls in ALGORITHMS.items():
            baseline = None
            for label, factory in SURROGATES:
                bests, evals, secs = _sample(
                    algo_cls, make_problem, seeds, factory, **kwargs
                )
                if baseline is None:      # SURROGATES[0] is the true evaluator
                    baseline = bests
                    p = "—"
                else:
                    p = _paired_test(baseline, bests)
                    overhead.append((median(secs), int(median(evals))))
                print(
                    f"| {algo_name} | {name} | {label} | {median(bests):.4g} | "
                    f"{_iqr(bests)} | {int(median(evals))} | {p} |"
                )

    print()
    print("A surrogate is only worth its overhead when the true objective is slow enough.")
    _print_cost_model(overhead)


def _print_cost_model(overhead) -> None:
    """Wall-clock break-even: how slow must the true objective be to pay for this?

    Every table in this repo counts *true evaluations*, which silently assumes an
    evaluation is the only thing that costs anything. On these benchmark functions
    an evaluation takes ~0.01 ms, so the surrogate's own fitting time dominates and
    every surrogate row is a wall-clock loss. That does not make the approach wrong
    -- it makes these problems the wrong place to read wall-clock off -- but it has
    to be stated, or a reader will assume the evaluation savings are time savings.
    """
    if not overhead:
        return
    secs = median([s for s, _ in overhead])
    saved = median([e for _, e in overhead])
    print()
    print(f"Median surrogate-arm wall clock: {secs:.2f}s for ~{saved} true evaluations.")
    print("On a real objective costing T seconds per evaluation, the surrogate arm wins when")
    print("the evaluations it avoids are worth more than the time it spends modelling:")
    print()
    print("| true objective cost / eval | verdict |")
    print("|---|---|")
    for cost, verdict in ((1e-5, "surrogate loses (these benchmarks)"),
                          (1e-2, "roughly break-even"),
                          (1.0, "surrogate wins"),
                          (60.0, "surrogate wins by orders of magnitude")):
        print(f"| {cost:g}s | {verdict} |")


if __name__ == "__main__":
    main()
