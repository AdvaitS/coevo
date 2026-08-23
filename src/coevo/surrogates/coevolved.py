"""Coevolved fitness predictor: a surrogate that tracks how well it adapts."""

from __future__ import annotations

import numpy as np

from coevo.surrogates.base import Surrogate


class CoevolvedPredictor(Surrogate):
    """Wraps a surrogate and records how well it tracks the population.

    Re-fitting the predictor each generation on the (evolving) archive of true
    evaluations mirrors the coevolved fitness predictors of Schmidt & Lipson
    (2008): instead of approximating the whole fitness landscape, the predictor
    specialises to the current population and exposes an error trace describing
    how well it is tracking it.
    """

    def __init__(self, surrogate: Surrogate) -> None:
        self._surrogate = surrogate
        self.fitted = False
        self.error_trace: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CoevolvedPredictor":
        self._surrogate.fit(X, y)
        self.fitted = bool(getattr(self._surrogate, "fitted", False))
        pred = np.asarray(self._surrogate.predict(X), dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        self.error_trace.append(float(np.sqrt(np.mean((pred - y) ** 2))))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._surrogate.predict(X)
