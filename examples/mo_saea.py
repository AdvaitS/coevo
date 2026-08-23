"""Multi-objective SAEA demo: NSGA-II on ZDT1, exact vs surrogate-assisted.

Run with::

    python examples/mo_saea.py
"""

from coevo import (
    ClippedPredictor,
    NSGA2,
    RBFSurrogate,
    SurrogateMultiObjectiveEvaluator,
    benchmarks,
)
from coevo.core.metrics import igd


def main() -> None:
    ref = benchmarks.zdt1_front()

    exact = NSGA2(pop_size=100, generations=250).optimize(benchmarks.zdt1(dim=10))
    print("exact NSGA-II:      ", exact.summary(), f"| IGD={igd(exact.objectives, ref):.4f}")

    problem = benchmarks.zdt1(dim=10)
    evaluator = SurrogateMultiObjectiveEvaluator(
        problem, lambda: ClippedPredictor(RBFSurrogate()), eval_fraction=0.3
    )
    saea = NSGA2(pop_size=100, generations=250).optimize(problem, evaluator)
    print("surrogate NSGA-II:  ", saea.summary(), f"| IGD={igd(saea.objectives, ref):.4f}")

    saving = 100 * (1 - saea.true_evaluations / exact.true_evaluations)
    print(f"  -> {saving:.0f}% fewer true evaluations")


if __name__ == "__main__":
    main()
