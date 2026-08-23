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
