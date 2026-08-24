"""Evaluator plumbing and the surrogate-assisted loop."""

import numpy as np
import pytest

from coevo import (
    CoevolvedPredictor,
    DifferentialEvolution,
    GeneticAlgorithm,
    NearestNeighborSurrogate,
    RBFSurrogate,
    SurrogateEvaluator,
    TrueEvaluator,
    benchmarks,
)


def test_true_evaluator_counts_evaluations():
    problem = benchmarks.sphere(5)
    ev = TrueEvaluator(problem)
    x = np.random.default_rng(0).normal(size=(7, 5))
    ev(x)
    ev(x)
    assert ev.n_true == 14


def test_surrogate_evaluator_reduces_true_evaluations():
    problem = benchmarks.sphere(5)
    rng = np.random.default_rng(0)
    ev = SurrogateEvaluator(
        problem, CoevolvedPredictor(RBFSurrogate()), eval_fraction=0.25, warmup=3
    )
    for _ in range(20):
        ev(rng.normal(size=(40, 5)))
    assert ev.n_true < 20 * 40  # fewer true evals than a fully-exact evaluator


def test_surrogate_assisted_de_converges():
    problem = benchmarks.sphere(5)
    algo = DifferentialEvolution(pop_size=30, generations=200, seed=2)
    ev = SurrogateEvaluator(
        problem,
        CoevolvedPredictor(NearestNeighborSurrogate()),
        eval_fraction=0.3,
        warmup=5,
    )
    result = algo.optimize(problem, ev)
    assert result.best_fitness < 1e-6
    assert result.true_evaluations < 30 + 30 * 200


def test_surrogate_evaluator_prediction_errors_exposed():
    problem = benchmarks.sphere(4)
    ev = SurrogateEvaluator(
        problem, CoevolvedPredictor(RBFSurrogate()), eval_fraction=0.5, warmup=2
    )
    rng = np.random.default_rng(1)
    for _ in range(6):
        ev(rng.normal(size=(20, 4)))
    assert len(ev.prediction_errors) >= 1


def test_generation_strategy_reduces_evals_and_converges():
    problem = benchmarks.sphere(5)
    algo = DifferentialEvolution(pop_size=30, generations=100, seed=3)
    ev = SurrogateEvaluator(
        problem,
        CoevolvedPredictor(NearestNeighborSurrogate()),
        eval_fraction=0.2,
        warmup=4,
        strategy="generation",
    )
    result = algo.optimize(problem, ev)
    assert result.true_evaluations < 30 + 30 * 100
    # sphere(5) starts around 40; converging to <0.1 on a fifth of the budget is
    # the property under test. A tighter bound here only tracks seed luck --
    # across seeds this configuration lands between 0.008 and 0.06.
    assert result.best_fitness < 0.1


def test_invalid_strategy_raises():
    problem = benchmarks.sphere(5)
    with pytest.raises(ValueError, match="strategy"):
        SurrogateEvaluator(problem, NearestNeighborSurrogate(), strategy="bogus")


# --- predicted fitness must not outlive the batch it was produced for -------


def test_evaluators_report_which_values_are_true():
    problem = benchmarks.sphere(4)

    exact = TrueEvaluator(problem)
    exact(np.zeros((6, 4)))
    assert exact.true_mask(6).all(), "an exact evaluator returns only true values"

    ev = SurrogateEvaluator(
        problem, NearestNeighborSurrogate(), eval_fraction=0.25, warmup=2
    )
    rng = np.random.default_rng(0)
    for _ in range(4):
        ev(rng.uniform(-5, 5, size=(20, 4)))
    mask = ev.true_mask(20)
    assert mask.sum() == 5, "eval_fraction=0.25 of 20 candidates is 5 true evaluations"
    assert not mask.all(), "the rest are predictions and must be flagged as such"


def test_reported_best_is_never_a_surrogate_artefact():
    """An unclipped RBF extrapolates wildly; the reported best must still be real.

    Before the incumbent was verified, the GA's internal best on this
    configuration was off from the true value by a median of ~1e57.
    """
    gaps = []
    for seed in range(4):
        problem = benchmarks.sphere(5)
        ev = SurrogateEvaluator(
            problem,
            CoevolvedPredictor(RBFSurrogate()),
            eval_fraction=0.25,
            warmup=5,
            archive_size=100,
        )
        result = GeneticAlgorithm(pop_size=50, generations=60, seed=seed).optimize(problem, ev)
        gaps.append(abs(result.best_fitness - result.metadata["selected_fitness"]))
    assert np.median(gaps) < 1.0, f"internal best diverged from reality: {gaps}"


def test_surrogate_evaluate_true_bypasses_the_surrogate():
    problem = benchmarks.sphere(3)
    ev = SurrogateEvaluator(problem, NearestNeighborSurrogate(), warmup=1)
    before = ev.n_true
    y = ev.evaluate_true(np.ones((2, 3)))
    assert ev.n_true == before + 2
    np.testing.assert_allclose(y, problem.evaluate(np.ones((2, 3))))
    assert ev.true_mask(2).all()
