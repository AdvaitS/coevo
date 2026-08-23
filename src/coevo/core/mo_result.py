"""Multi-objective optimization result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from coevo.core.mo_problem import MultiObjectiveProblem


@dataclass
class MultiObjectiveResult:
    """Outcome of a multi-objective optimization run.

    Attributes
    ----------
    problem:
        The problem that was solved.
    solutions:
        Non-dominated solutions, shape ``(k, dim)``.
    objectives:
        Objective vectors of ``solutions``, shape ``(k, n_objectives)``.
    history:
        Per-generation hypervolume (if the problem defines a reference point),
        otherwise per-generation front sizes.
    true_evaluations:
        Number of true (non-surrogate) evaluations.
    metadata:
        Free-form diagnostics.
    """

    problem: MultiObjectiveProblem
    solutions: np.ndarray
    objectives: np.ndarray
    history: list[float] = field(default_factory=list)
    true_evaluations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        hv = ""
        if self.problem.reference_point is not None:
            from coevo.core.metrics import hypervolume

            hv = f" | HV={hypervolume(self.objectives, self.problem.reference_point):.4g}"
        return (
            f"{self.problem.name}: {len(self.objectives)} non-dominated solutions{hv} "
            f"| true_evals={self.true_evaluations}"
        )
