"""Fitness evaluators: exact evaluation and surrogate-assisted pre-selection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from coevo.core.problem import Problem


class Evaluator(ABC):
    """Provides fitness values for a batch of candidate solutions."""

    #: Boolean mask over the most recent batch: True where the returned fitness
    #: came from the true objective rather than a surrogate prediction. An
    #: optimizer that carries fitness across generations needs this, otherwise
    #: an over-optimistic prediction is never revisited and occupies the
    #: population for the rest of the run.
    last_true_mask: np.ndarray | None = None

    @property
    @abstractmethod
    def n_true(self) -> int:
        """Number of true objective evaluations performed so far."""

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return fitness for a batch ``x`` of shape ``(n, dim)``."""

    def true_mask(self, n: int) -> np.ndarray:
        """Mask for the last batch, defaulting to all-true for exact evaluators."""
        mask = self.last_true_mask
        if mask is None or len(mask) != n:
            return np.ones(n, dtype=bool)
        return mask

    def evaluate_true(self, x: np.ndarray) -> np.ndarray:
        """Force a true evaluation of ``x``, bypassing any surrogate."""
        return self(x)


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
        self.last_true_mask = np.ones(len(x), dtype=bool)
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
    strategy:
        Model-management strategy. ``"individual"`` (default) true-evaluates the
        most promising ``eval_fraction`` of each batch (pre-selection);
        ``"generation"`` alternates whole-true-evaluation generations with
        surrogate-only generations (best candidate always truly re-evaluated).
    """

    def __init__(
        self,
        problem: Problem,
        surrogate: Any,
        eval_fraction: float = 0.25,
        warmup: int = 3,
        retrain_every: int = 1,
        archive_size: int | None = 200,
        strategy: str = "individual",
    ) -> None:
        if strategy not in ("individual", "generation"):
            raise ValueError(f"Unknown strategy {strategy!r}; use 'individual' or 'generation'.")
        self.problem = problem
        self.surrogate = surrogate
        self.eval_fraction = eval_fraction
        self.warmup = warmup
        self.retrain_every = retrain_every
        self.archive_size = archive_size
        self.strategy = strategy
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

    def evaluate_true(self, x: np.ndarray) -> np.ndarray:
        """Truly evaluate ``x``, bypassing the surrogate and counting the cost."""
        x = np.atleast_2d(x)
        y = self._true_eval(x)
        self.last_true_mask = np.ones(len(x), dtype=bool)
        return y

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        self._calls += 1

        # Warm-up (and any degenerate empty-archive state): evaluate truly.
        if self._calls <= self.warmup or not self._X:
            y = self._true_eval(x)
            self.last_true_mask = np.ones(len(x), dtype=bool)
            # Fit the surrogate once warm-up is complete so the first
            # prediction is valid.
            if self._calls >= self.warmup:
                X, yt = self._archive()
                self.surrogate.fit(X, yt)
            return y

        if self.strategy == "generation":
            period = max(1, int(round(1.0 / self.eval_fraction)))
            if (self._calls - self.warmup) % period == 0:
                # True generation: evaluate the whole batch, then refresh the
                # surrogate on the updated archive.
                y = self._true_eval(x)
                self.last_true_mask = np.ones(len(x), dtype=bool)
                X, yt = self._archive()
                self.surrogate.fit(X, yt)
                return y
            # Surrogate-only generation: predict, but always truly re-evaluate
            # the single best candidate so the search cannot drift into
            # surrogate minima.
            y_hat = np.asarray(self.surrogate.predict(x), dtype=float).ravel()
            best = int(np.argmin(y_hat))
            y_hat[best] = self._true_eval(x[best : best + 1])[0]
            mask = np.zeros(len(x), dtype=bool)
            mask[best] = True
            self.last_true_mask = mask
            return y_hat

        # Default "individual" strategy: pre-selection.
        y_hat = np.asarray(self.surrogate.predict(x), dtype=float).ravel()

        n = len(x)
        k = max(1, int(np.ceil(n * self.eval_fraction)))
        top = np.argsort(y_hat)[:k]

        y_true = self._true_eval(x[top])
        y_hat[top] = y_true
        mask = np.zeros(n, dtype=bool)
        mask[top] = True
        self.last_true_mask = mask

        if self._calls % self.retrain_every == 0:
            X, y = self._archive()
            self.surrogate.fit(X, y)

        return y_hat
