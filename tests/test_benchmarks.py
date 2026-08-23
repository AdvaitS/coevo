"""Benchmark function correctness and Problem plumbing."""

import numpy as np
import pytest

from coevo import benchmarks
from coevo.core.problem import Problem


def _at_origin(f):
    return np.zeros((1, 10))


@pytest.mark.parametrize(
    "problem, x",
    [
        (benchmarks.sphere(10), np.zeros((1, 10))),
        (benchmarks.rastrigin(10), np.zeros((1, 10))),
        (benchmarks.ackley(10), np.zeros((1, 10))),
        (benchmarks.griewank(10), np.zeros((1, 10))),
        (benchmarks.rosenbrock(10), np.ones((1, 10))),
    ],
)
def test_known_optimum(problem, x):
    np.testing.assert_allclose(problem.evaluate(x), 0.0, atol=1e-9)


def test_batch_evaluation_matches_elementwise():
    p = benchmarks.sphere(4)
    x = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]])
    batch = p.evaluate(x)
    single = np.array([p.evaluate(x[i : i + 1])[0] for i in range(len(x))])
    np.testing.assert_allclose(batch, single)


def test_bounds_normalise_to_dim():
    p = Problem("toy", lambda x: np.sum(x, axis=1), [0.0, 1.0], dim=5)
    assert p.bounds.shape == (5, 2)


def test_noisy_problem_is_reproducible():
    base = benchmarks.sphere(5)
    a = benchmarks.noisy(base, sigma=0.1, seed=7)
    b = benchmarks.noisy(base, sigma=0.1, seed=7)
    x = np.random.default_rng(0).normal(size=(20, 5))
    np.testing.assert_allclose(a.evaluate(x), b.evaluate(x))


def test_shifted_sphere_optimum():
    p = benchmarks.shifted_sphere(5, offset=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    np.testing.assert_allclose(p.evaluate(np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])), 0.0, atol=1e-9)
