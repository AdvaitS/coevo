"""Classic continuous benchmark functions and helpers.

Every function is vectorised over the first axis (batch dimension) and returns
an ``(n,)`` array of fitness values, matching the contract expected by
:class:`~coevo.core.problem.Problem`.
"""

from __future__ import annotations

import numpy as np

from coevo.core.problem import Problem


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=1)


def _rastrigin(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


def _rosenbrock(x: np.ndarray) -> np.ndarray:
    xa, xb = x[:, :-1], x[:, 1:]
    return np.sum(100 * (xb - xa**2) ** 2 + (1 - xa) ** 2, axis=1)


def _ackley(x: np.ndarray) -> np.ndarray:
    a = -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2, axis=1)))
    b = -np.exp(np.mean(np.cos(2 * np.pi * x), axis=1))
    return a + b + 20 + np.e


def _griewank(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    s = np.sum(x**2, axis=1) / 4000
    p = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))), axis=1)
    return 1 + s - p


def sphere(dim: int = 10) -> Problem:
    """Shift-invariant unimodal bowl, optimum 0 at the origin."""
    return Problem("sphere", _sphere, [-5.12, 5.12], dim, known_optimum=0.0)


def rastrigin(dim: int = 10) -> Problem:
    """Highly multimodal separable function, optimum 0 at the origin."""
    return Problem("rastrigin", _rastrigin, [-5.12, 5.12], dim, known_optimum=0.0)


def rosenbrock(dim: int = 10) -> Problem:
    """Non-convex valley function, optimum 0 at ``x = [1, 1, ...]``."""
    return Problem("rosenbrock", _rosenbrock, [-2.048, 2.048], dim, known_optimum=0.0)


def ackley(dim: int = 10) -> Problem:
    """Multimodal function with many local optima, optimum 0 at the origin."""
    return Problem("ackley", _ackley, [-32.768, 32.768], dim, known_optimum=0.0)


def griewank(dim: int = 10) -> Problem:
    """Multimodal, non-separable function, optimum 0 at the origin."""
    return Problem("griewank", _griewank, [-600, 600], dim, known_optimum=0.0)


def shifted_sphere(dim: int = 10, offset: np.ndarray | None = None, seed: int = 0) -> Problem:
    """Sphere with a shifted optimum, for testing rotation/translation handling."""
    if offset is None:
        offset = np.random.default_rng(seed).uniform(-5, 5, size=dim)
    offset = np.asarray(offset, dtype=float)

    def f(x: np.ndarray) -> np.ndarray:
        return np.sum((x - offset) ** 2, axis=1)

    return Problem("shifted_sphere", f, [-5.12, 5.12], dim, known_optimum=0.0)


def noisy(problem: Problem, sigma: float = 0.05, seed: int = 0) -> Problem:
    """Return a copy of ``problem`` with additive Gaussian evaluation noise."""
    return Problem(
        name=f"{problem.name}_noisy",
        func=problem.func,
        bounds=problem.bounds,
        dim=problem.dim,
        known_optimum=problem.known_optimum,
        noise=sigma,
        seed=seed,
    )
