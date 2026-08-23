"""Minimal end-to-end example.

Run with::

    python examples/quickstart.py
"""

from coevo import (
    CoevolvedPredictor,
    DifferentialEvolution,
    NearestNeighborSurrogate,
    ParticleSwarmOptimization,
    SurrogateEvaluator,
    benchmarks,
)


def main() -> None:
    # 1) Solve a classic benchmark exactly.
    problem = benchmarks.ackley(dim=5)
    result = DifferentialEvolution(pop_size=50, generations=300).optimize(problem)
    print("exact DE:", result.summary())

    # 2) Solve the same problem with a surrogate-assisted evaluator, which
    #    spends far fewer *true* evaluations. The bounded 1-NN predictor is the
    #    robust choice; swap in RBFSurrogate() for smoother landscapes.
    surrogate = CoevolvedPredictor(NearestNeighborSurrogate())
    sa_problem = benchmarks.ackley(dim=5)
    sa_evaluator = SurrogateEvaluator(sa_problem, surrogate, eval_fraction=0.25)
    sa_result = DifferentialEvolution(pop_size=50, generations=300).optimize(
        sa_problem, sa_evaluator
    )
    print("surrogate-assisted DE:", sa_result.summary())
    print(
        f"  -> {100 * (1 - sa_result.true_evaluations / result.true_evaluations):.1f}% "
        "fewer true evaluations"
    )

    # 3) Any algorithm can be swapped in.
    pso = ParticleSwarmOptimization(pop_size=50, generations=300)
    print("exact PSO:", pso.optimize(benchmarks.sphere(dim=5)).summary())


if __name__ == "__main__":
    main()
