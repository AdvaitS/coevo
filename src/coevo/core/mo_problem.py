"""Multi-objective problem definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class MultiObjectiveProblem:
    """A continuous multi-objective black-box problem to be minimized.

    Parameters
    ----------
    name:
        Identifier used in logs, results, and benchmarks.
    func:
        The objective function. Must accept an ``(n, dim)`` array and return an
        ``(n, n_objectives)`` array of objective values (all to be minimized).
    bounds:
        Per-dimension bounds as a ``(dim, 2)`` array, or a single ``(2,)`` array
        of ``[lower, upper]`` applied to every dimension.
    dim:
        Number of decision variables.
    n_objectives:
        Number of objectives (``m``).
    reference_point:
        A point dominated by the whole Pareto front, used to compute hypervolume.
    noise:
        Standard deviation of Gaussian noise added to every evaluation.
    seed:
        Seed for the objective's internal noise generator.
    """

    name: str
    func: Callable[[np.ndarray], np.ndarray]
    bounds: np.ndarray
    dim: int
    n_objectives: int
    reference_point: np.ndarray | None = None
    noise: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        self.bounds = np.asarray(self.bounds, dtype=float)
        if self.bounds.ndim == 1:
            self.bounds = np.tile(self.bounds, (self.dim, 1))
        if self.bounds.shape != (self.dim, 2):
            raise ValueError(f"bounds must have shape ({self.dim}, 2), got {self.bounds.shape}")
        if self.reference_point is not None:
            self.reference_point = np.asarray(self.reference_point, dtype=float)
        self._rng = np.random.default_rng(self.seed)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the objectives on a batch ``x`` of shape ``(n, dim)``.

        Returns an ``(n, n_objectives)`` array.
        """
        x = np.atleast_2d(x)
        out = np.asarray(self.func(x), dtype=float)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        if self.noise:
            out = out + self._rng.normal(0.0, self.noise, size=out.shape)
        return out

    @property
    def lower(self) -> np.ndarray:
        return self.bounds[:, 0]

    @property
    def upper(self) -> np.ndarray:
        return self.bounds[:, 1]
