"""Optimization result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from coevo.core.problem import Problem


@dataclass
class OptimizationResult:
    """Outcome of an optimization run.

    Attributes
    ----------
    problem:
        The problem that was solved.
    best_x:
        Best solution found, shape ``(dim,)``.
    best_fitness:
        Fitness of ``best_x``.
    history:
        Best fitness observed after each generation.
    true_evaluations:
        Number of true (non-surrogate) objective evaluations performed.
    metadata:
        Free-form diagnostics (e.g. the coevolved predictor's error trace).
    """

    problem: Problem
    best_x: np.ndarray
    best_fitness: float
    history: list[float] = field(default_factory=list)
    true_evaluations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        opt = self.problem.known_optimum
        gap = "" if opt is None else f" | gap-to-optimum {self.best_fitness - opt:.3g}"
        return (
            f"{self.problem.name}: best={self.best_fitness:.6g}{gap} "
            f"| true_evals={self.true_evaluations}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "problem": self.problem.name,
            "best_fitness": float(self.best_fitness),
            "true_evaluations": self.true_evaluations,
            "best_x": np.asarray(self.best_x).tolist(),
        }
