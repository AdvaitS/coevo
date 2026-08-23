"""Problem definitions for continuous black-box optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Problem:
    """A continuous black-box objective to be minimized.

    Parameters
    ----------
    name:
        Identifier used in logs, results, and benchmarks.
    func:
        The objective function. Must accept an ``(n, dim)`` array and return an
        ``(n,)`` array of fitness values.
    bounds:
        Per-dimension bounds as a ``(dim, 2)`` array, or a single ``(2,)`` array
        of ``[lower, upper]`` applied to every dimension.
    dim:
        Number of decision variables.
    known_optimum:
        The best achievable fitness, when known (used in benchmarks to report the
        gap-to-optimum).
    noise:
        Standard deviation of Gaussian noise added to every evaluation. Useful to
        emulate the stochasticity of real, expensive objectives.
    seed:
        Seed for the objective's internal noise generator (keeps noisy benchmarks
        reproducible).
    """

    name: str
    func: Callable[[np.ndarray], np.ndarray]
    bounds: np.ndarray
    dim: int
    known_optimum: float | None = None
    noise: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        self.bounds = np.asarray(self.bounds, dtype=float)
        if self.bounds.ndim == 1:
            self.bounds = np.tile(self.bounds, (self.dim, 1))
        if self.bounds.shape != (self.dim, 2):
            raise ValueError(
                f"bounds must have shape ({self.dim}, 2), got {self.bounds.shape}"
            )
        self._rng = np.random.default_rng(self.seed)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the objective on a batch ``x`` of shape ``(n, dim)``."""
        x = np.atleast_2d(x)
        fitness = np.asarray(self.func(x), dtype=float).ravel()
        if self.noise:
            fitness = fitness + self._rng.normal(0.0, self.noise, size=fitness.shape)
        return fitness

    @property
    def lower(self) -> np.ndarray:
        """Lower bound per dimension, shape ``(dim,)``."""
        return self.bounds[:, 0]

    @property
    def upper(self) -> np.ndarray:
        """Upper bound per dimension, shape ``(dim,)``."""
        return self.bounds[:, 1]
