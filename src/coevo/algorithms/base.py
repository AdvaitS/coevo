"""Base class for population-based evolutionary algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod

from coevo.core.problem import Problem
from coevo.core.result import OptimizationResult
from coevo.evaluation import Evaluator, TrueEvaluator


class BaseAlgorithm(ABC):
    """Abstract base for population-based optimization algorithms.

    All concrete algorithms follow the same contract: they accept a
    :class:`~coevo.core.problem.Problem` and an optional
    :class:`~coevo.evaluation.Evaluator`, and return an
    :class:`~coevo.core.result.OptimizationResult`.
    """

    def __init__(
        self,
        pop_size: int = 50,
        generations: int = 100,
        seed: int = 0,
        max_evaluations: int | None = None,
    ) -> None:
        self.pop_size = pop_size
        self.generations = generations
        self.seed = seed
        self.max_evaluations = max_evaluations

    @abstractmethod
    def optimize(
        self, problem: Problem, evaluator: Evaluator | None = None
    ) -> OptimizationResult:
        """Run the algorithm and return an :class:`OptimizationResult`."""

    @staticmethod
    def _resolve_evaluator(problem: Problem, evaluator: Evaluator | None) -> Evaluator:
        return evaluator if evaluator is not None else TrueEvaluator(problem)
