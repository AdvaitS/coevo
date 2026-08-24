"""Efficient global optimization and the acquisition functions."""

import numpy as np
import pytest

from coevo import EfficientGlobalOptimization, TrueEvaluator, benchmarks
from coevo.acquisition import (
    acquisition,
    expected_improvement,
    lower_confidence_bound,
    probability_of_improvement,
)


def test_expected_improvement_is_zero_where_the_model_is_certain():
    """A point with no uncertainty offers nothing to learn, however good its mean."""
    ei = expected_improvement(np.array([0.0, -5.0]), np.array([0.0, 0.0]), best=1.0)
    np.testing.assert_allclose(ei, [0.0, 0.0])


def test_expected_improvement_rewards_both_promise_and_uncertainty():
    best = 1.0
    # same mean, more uncertainty -> more expected improvement
    low = expected_improvement(np.array([0.9]), np.array([0.1]), best)[0]
    high = expected_improvement(np.array([0.9]), np.array([1.0]), best)[0]
    assert high > low

    # same uncertainty, better mean -> more expected improvement
    worse = expected_improvement(np.array([1.5]), np.array([0.5]), best)[0]
    better = expected_improvement(np.array([0.2]), np.array([0.5]), best)[0]
    assert better > worse
    assert worse >= 0.0, "expected improvement is never negative"


def test_lcb_and_pi_have_the_expected_shapes():
    mean, std = np.array([1.0, 1.0]), np.array([0.1, 2.0])
    lcb = lower_confidence_bound(mean, std, kappa=2.0)
    assert lcb[1] > lcb[0], "larger-is-better, so more uncertainty must score higher"

    pi = probability_of_improvement(mean, std, best=1.0)
    assert np.all((pi >= 0) & (pi <= 1)), "a probability must stay in [0, 1]"


def test_unknown_acquisition_is_rejected():
    with pytest.raises(ValueError, match="unknown acquisition"):
        acquisition("bogus")


def test_ego_respects_its_evaluation_budget():
    problem = benchmarks.sphere(3)
    evaluator = TrueEvaluator(problem)
    result = EfficientGlobalOptimization(n_initial=5, max_evaluations=25, seed=0).optimize(
        problem, evaluator
    )
    assert result.true_evaluations == 25
    assert len(result.history) >= 1


def test_ego_beats_random_sampling_at_a_small_budget():
    """The whole point of an acquisition function: spend evaluations better than chance."""
    budgets, ego, sampled = 40, [], []
    for seed in range(4):
        problem = benchmarks.sphere(4)
        ego.append(
            EfficientGlobalOptimization(n_initial=8, max_evaluations=budgets, seed=seed)
            .optimize(problem, TrueEvaluator(problem))
            .best_fitness
        )
        rng = np.random.default_rng(seed)
        X = rng.uniform(problem.lower, problem.upper, size=(budgets, problem.dim))
        sampled.append(float(np.min(problem.evaluate(X))))
    assert np.median(ego) < np.median(sampled)


def test_ego_history_is_monotone():
    problem = benchmarks.ackley(3)
    result = EfficientGlobalOptimization(n_initial=6, max_evaluations=30, seed=1).optimize(
        problem, TrueEvaluator(problem)
    )
    history = np.array(result.history)
    assert np.all(np.diff(history) <= 1e-12), "best-so-far can never get worse"


def test_gp_surrogate_exposes_posterior_uncertainty():
    """The variance is the whole reason to use a GP; it must be reachable."""
    from coevo import GaussianProcessSurrogate

    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(20, 2))
    y = (X**2).sum(axis=1)
    gp = GaussianProcessSurrogate().fit(X, y)
    mean, std = gp.predict_with_std(X)
    assert mean.shape == std.shape == (20,)
    assert np.all(std >= 0)
    # uncertainty is lower on training points than far outside their range
    _, far_std = gp.predict_with_std(np.full((3, 2), 50.0))
    assert far_std.mean() > std.mean()
