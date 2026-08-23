"""Fitness evaluators: exact evaluation and surrogate-assisted pre-selection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from coevo.core.problem import Problem


class Evaluator(ABC):
    """Provides fitness values for a batch of candidate solutions."""

    @property
    @abstractmethod
    def n_true(self) -> int:
        """Number of true objective evaluations performed so far."""

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return fitness for a batch ``x`` of shape ``(n, dim)``."""


class TrueEvaluator(Evaluator):
    """Evaluates every candidate against the true objective."""

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self._n_true = 0

    @property
    def n_true(self) -> int:
        return self._n_true

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        self._n_true += len(x)
        return self.problem.evaluate(x)


class SurrogateEvaluator(Evaluator):
    """Pre-selection evaluator driven by a surrogate fitness predictor.

    Each call predicts fitness for the whole batch with a surrogate model, sends
    only the most promising ``eval_fraction`` of candidates to the true
    objective, and returns predicted values for the rest. The surrogate is
    periodically retrained on the archive of true evaluations, so it adapts to
    the shifting population — a coevolutionary fitness-prediction loop.

    Parameters
    ----------
    problem:
        The underlying true objective.
    surrogate:
        A ``Surrogate`` used to predict fitness.
    eval_fraction:
        Fraction of each batch (after warm-up) that is truly evaluated.
    warmup:
        Number of initial batches to evaluate fully, to seed the surrogate.
    retrain_every:
        Retrain the surrogate every ``retrain_every`` batches.
    archive_size:
        Cap the true-evaluation archive to the most recent ``archive_size``
        points. Capping keeps surrogates fast and makes the predictor focus on
        the current population (rather than stale history) — the coevolutionary
        flavour of the method.
    """

    def __init__(
        self,
        problem: Problem,
        surrogate: Any,
        eval_fraction: float = 0.25,
        warmup: int = 3,
        retrain_every: int = 1,
        archive_size: int | None = 200,
    ) -> None:
        self.problem = problem
        self.surrogate = surrogate
        self.eval_fraction = eval_fraction
        self.warmup = warmup
        self.retrain_every = retrain_every
        self.archive_size = archive_size
        self._n_true = 0
        self._calls = 0
        self._X: list[np.ndarray] = []
        self._y: list[np.ndarray] = []

    @property
    def n_true(self) -> int:
        return self._n_true

    @property
    def prediction_errors(self) -> list[float]:
        """RMSE of the surrogate after each retrain, if it tracks error."""
        trace = getattr(self.surrogate, "error_trace", None)
        return list(trace) if trace is not None else []

    def _archive(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._X:
            return np.empty((0, self.problem.dim)), np.empty((0,))
        X = np.vstack(self._X)
        y = np.concatenate(self._y)
        if self.archive_size is not None and len(X) > self.archive_size:
            X, y = X[-self.archive_size :], y[-self.archive_size :]
        return X, y

    def _true_eval(self, x: np.ndarray) -> np.ndarray:
        y = self.problem.evaluate(x)
        self._n_true += len(x)
        self._X.append(np.asarray(x, dtype=float))
        self._y.append(np.asarray(y, dtype=float))
        return y

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        self._calls += 1

        # Warm-up (and any degenerate empty-archive state): evaluate truly.
        if self._calls <= self.warmup or not self._X:
            y = self._true_eval(x)
            # Fit the surrogate once warm-up is complete so the first
            # prediction is valid.
            if self._calls >= self.warmup:
                X, yt = self._archive()
                self.surrogate.fit(X, yt)
            return y

        y_hat = np.asarray(self.surrogate.predict(x), dtype=float).ravel()

        n = len(x)
        k = max(1, int(np.ceil(n * self.eval_fraction)))
        top = np.argsort(y_hat)[:k]

        y_true = self._true_eval(x[top])
        y_hat[top] = y_true

        if self._calls % self.retrain_every == 0:
            X, y = self._archive()
            self.surrogate.fit(X, y)

        return y_hat
