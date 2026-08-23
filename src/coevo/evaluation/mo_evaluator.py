"""Multi-objective fitness evaluators (exact and surrogate-assisted)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

from coevo.core.metrics import fast_non_dominated_sort
from coevo.core.mo_problem import MultiObjectiveProblem


class MultiObjectiveEvaluator(ABC):
    """Provides objective vectors for a batch of candidate solutions."""

    @property
    @abstractmethod
    def n_true(self) -> int:
        """Number of true objective evaluations performed so far."""

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return objective vectors for batch ``x``, shape ``(n, n_objectives)``."""


class TrueMultiObjectiveEvaluator(MultiObjectiveEvaluator):
    """Evaluates every candidate against the true objectives."""

    def __init__(self, problem: MultiObjectiveProblem) -> None:
        self.problem = problem
        self._n_true = 0

    @property
    def n_true(self) -> int:
        return self._n_true

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        self._n_true += len(x)
        return self.problem.evaluate(x)


class SurrogateMultiObjectiveEvaluator(MultiObjectiveEvaluator):
    """Surrogate-assisted pre-selection for multi-objective problems.

    One surrogate is trained per objective. Each call predicts the whole batch,
    then truly evaluates only the most promising candidates — those in the
    earliest *predicted* Pareto fronts — and returns predicted values for the
    rest. Surrogates are retrained on the archive of true evaluations.
    """

    def __init__(
        self,
        problem: MultiObjectiveProblem,
        surrogate_factory: Callable[[], Any],
        eval_fraction: float = 0.25,
        warmup: int = 3,
        retrain_every: int = 1,
        archive_size: int | None = 200,
    ) -> None:
        self.problem = problem
        self.surrogates = [surrogate_factory() for _ in range(problem.n_objectives)]
        self.eval_fraction = eval_fraction
        self.warmup = warmup
        self.retrain_every = retrain_every
        self.archive_size = archive_size
        self._n_true = 0
        self._calls = 0
        self._X: list[np.ndarray] = []
        self._Y: list[np.ndarray] = []

    @property
    def n_true(self) -> int:
        return self._n_true

    def _archive(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._X:
            return np.empty((0, self.problem.dim)), np.empty((0, self.problem.n_objectives))
        X = np.vstack(self._X)
        Y = np.vstack(self._Y)
        if self.archive_size is not None and len(X) > self.archive_size:
            X, Y = X[-self.archive_size :], Y[-self.archive_size :]
        return X, Y

    def _true_eval(self, x: np.ndarray) -> np.ndarray:
        y = self.problem.evaluate(x)
        self._n_true += len(x)
        self._X.append(np.asarray(x, dtype=float))
        self._Y.append(np.asarray(y, dtype=float))
        return y

    def _retrain(self) -> None:
        X, Y = self._archive()
        if len(X) < 2:
            return
        for j, surrogate in enumerate(self.surrogates):
            surrogate.fit(X, Y[:, j])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        self._calls += 1

        if self._calls <= self.warmup or not self._X:
            y = self._true_eval(x)
            if self._calls >= self.warmup:
                self._retrain()
            return y

        y_hat = np.column_stack([s.predict(x) for s in self.surrogates])

        n = len(x)
        k = max(1, int(np.ceil(n * self.eval_fraction)))
        selected: list[int] = []
        for front in fast_non_dominated_sort(y_hat):
            for i in front:
                if len(selected) >= k:
                    break
                selected.append(i)
            if len(selected) >= k:
                break

        y_true = self._true_eval(x[selected])
        y_hat[selected] = y_true

        if self._calls % self.retrain_every == 0:
            self._retrain()

        return y_hat
