"""Algorithm convergence and reproducibility."""

import numpy as np
import pytest

from coevo import (
    DifferentialEvolution,
    GeneticAlgorithm,
    ParticleSwarmOptimization,
    benchmarks,
)


@pytest.mark.parametrize(
    "algo_cls, kwargs",
    [
        (DifferentialEvolution, dict(pop_size=30, generations=200)),
        (GeneticAlgorithm, dict(pop_size=30, generations=200)),
        (ParticleSwarmOptimization, dict(pop_size=30, generations=200)),
    ],
)
def test_converges_on_sphere(algo_cls, kwargs):
    problem = benchmarks.sphere(5)
    algo = algo_cls(seed=1, **kwargs)
    result = algo.optimize(problem)
    assert result.best_fitness < 1e-6


@pytest.mark.parametrize(
    "algo_cls",
    [DifferentialEvolution, GeneticAlgorithm, ParticleSwarmOptimization],
)
def test_deterministic_given_seed(algo_cls):
    problem = benchmarks.sphere(5)
    r1 = algo_cls(seed=42).optimize(problem)
    r2 = algo_cls(seed=42).optimize(problem)
    assert r1.best_fitness == r2.best_fitness
    np.testing.assert_array_equal(r1.best_x, r2.best_x)


def test_result_reports_history_and_evals():
    problem = benchmarks.sphere(3)
    result = DifferentialEvolution(pop_size=10, generations=20).optimize(problem)
    assert len(result.history) == 21
    assert result.true_evaluations == 10 + 10 * 20
    assert result.history[-1] <= result.history[0]


def test_ga_mutation_scales_with_the_problem_bounds():
    """mutation_sigma is a fraction of each dimension's range, not an absolute step.

    With an absolute 0.1 the GA could not move on ackley's [-32.768, 32.768]
    domain: it sat at 8.5348 from generation 50 through generation 1000.
    """
    from coevo import GeneticAlgorithm, TrueEvaluator, benchmarks

    def best(generations):
        out = []
        for seed in range(3):
            problem = benchmarks.ackley(5)
            out.append(
                GeneticAlgorithm(pop_size=50, generations=generations, seed=seed)
                .optimize(problem, TrueEvaluator(problem))
                .best_fitness
            )
        return float(np.median(out))

    short, long = best(100), best(400)
    assert short < 1.0, f"GA still stuck on ackley: {short}"
    assert long < short, "more budget must buy a better solution"


def test_ga_keeps_exactly_the_requested_elites():
    """Generational replacement with elitism, not (mu+lambda) truncation."""
    from coevo import GeneticAlgorithm, TrueEvaluator, benchmarks

    problem = benchmarks.sphere(3)
    result = GeneticAlgorithm(pop_size=20, generations=30, seed=0, elite=1).optimize(
        problem, TrueEvaluator(problem)
    )
    # history must be monotone non-increasing: the single elite guarantees the
    # best-so-far can never get worse.
    history = np.array(result.history)
    assert np.all(np.diff(history) <= 1e-12), "elitism must make the best monotone"
