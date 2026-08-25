"""Reporting a noisy run's answer must not itself be a noisy draw."""

import numpy as np
import pytest

from coevo import (
    DifferentialEvolution,
    EfficientGlobalOptimization,
    GeneticAlgorithm,
    ParticleSwarmOptimization,
    TrueEvaluator,
    benchmarks,
)

ALGORITHMS = [
    GeneticAlgorithm,
    DifferentialEvolution,
    ParticleSwarmOptimization,
    EfficientGlobalOptimization,
]


def _noisy_ackley(sigma=0.5, seed=0):
    problem = benchmarks.ackley(4)
    problem.noise = sigma
    problem.seed = seed
    problem.__post_init__()
    return problem


def test_evaluate_noiseless_strips_the_noise():
    problem = _noisy_ackley(sigma=1.0)
    x = np.zeros((1, problem.dim))
    clean = [float(problem.evaluate_noiseless(x)[0]) for _ in range(20)]
    assert len(set(clean)) == 1, "the noiseless path must be deterministic"
    assert clean[0] == pytest.approx(0.0, abs=1e-9), "ackley's optimum is 0 at the origin"

    noisy = [float(problem.evaluate(x)[0]) for _ in range(20)]
    assert len(set(noisy)) > 1, "the noisy path must still be noisy"


@pytest.mark.parametrize("algo_cls", ALGORITHMS)
def test_reported_fitness_never_beats_the_true_optimum(algo_cls):
    """The bug this closes: a single noisy draw at the end could land below the
    true minimum, so a noisy benchmark reported solving the problem better than
    the problem allows. ackley's minimum is exactly 0.
    """
    for seed in range(4):
        problem = _noisy_ackley(sigma=0.5, seed=seed)
        result = algo_cls(
            pop_size=10, generations=10_000_000, seed=seed, max_evaluations=120
        ).optimize(problem, TrueEvaluator(problem))
        assert result.best_fitness >= 0.0, (
            f"{algo_cls.__name__} reported {result.best_fitness:.5f}, "
            "below ackley's true minimum of 0"
        )


def test_noise_still_reaches_the_optimizer():
    """Fixing the report must not quietly make the problem noiseless."""
    problem = _noisy_ackley(sigma=0.5)
    evaluator = TrueEvaluator(problem)
    x = np.zeros((3, problem.dim))
    seen = {float(v) for _ in range(6) for v in evaluator(x)}
    assert len(seen) > 1, "the optimizer must still see a noisy objective"


def test_noiseless_matches_evaluate_when_there_is_no_noise():
    problem = benchmarks.sphere(3)
    rng = np.random.default_rng(0)
    X = rng.uniform(problem.lower, problem.upper, size=(10, problem.dim))
    np.testing.assert_allclose(problem.evaluate(X), problem.evaluate_noiseless(X))
