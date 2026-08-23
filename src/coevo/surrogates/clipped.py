"""Bounded surrogate wrapper: clip predictions to the observed fitness range."""

from __future__ import annotations

import numpy as np

from coevo.surrogates.base import Surrogate


class ClippedPredictor(Surrogate):
    """Wraps a surrogate and clips predictions to the training-fitness range.

    Unbounded surrogates (e.g. thin-plate RBFs, GPs) can extrapolate wildly
    outside their training data, and an aggressive optimizer will chase those
    imaginary minima — a classic surrogate-assisted-evolution failure mode.
    Clipping predictions to ``[min(y), max(y)]`` is the cheapest form of model
    management: it keeps the predictor honest without any tuning.
    """

    def __init__(self, surrogate: Surrogate) -> None:
        self._surrogate = surrogate
        self.fitted = False
        self._ymin = 0.0
        self._ymax = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClippedPredictor":
        self._surrogate.fit(X, y)
        y = np.asarray(y, dtype=float).ravel()
        self._ymin = float(np.min(y))
        self._ymax = float(np.max(y))
        self.fitted = bool(getattr(self._surrogate, "fitted", False))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(self._surrogate.predict(X), dtype=float).ravel()
        return np.clip(raw, self._ymin, self._ymax)
