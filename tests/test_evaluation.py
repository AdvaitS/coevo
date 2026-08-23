"""Evaluator plumbing and the surrogate-assisted loop."""

import numpy as np

from coevo import (
    CoevolvedPredictor,
    DifferentialEvolution,
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
        problem, CoevolvedPredictor(RBFSurrogate()), eval_fraction=0.3, warmup=5
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
