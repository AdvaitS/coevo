"""Nearest-neighbour surrogate: a dependency-free baseline predictor."""

from __future__ import annotations

import numpy as np

from coevo.surrogates.base import Surrogate


class NearestNeighborSurrogate(Surrogate):
    """Predicts the fitness of the closest training point (1-NN).

    A cheap, robust baseline. It is exact on the archive it was trained on and
    degrades gracefully elsewhere, making it a useful sanity check for the
    surrogate-assisted loop.
    """

    def __init__(self) -> None:
        self.fitted = False
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NearestNeighborSurrogate":
        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        self._X = X.astype(float)
        self._y = y
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        if not self.fitted or self._X is None or self._y is None or len(self._X) == 0:
            raise RuntimeError("Surrogate must be fitted before predict().")
        dists = ((X[:, None, :] - self._X[None, :, :]) ** 2).sum(axis=-1)
        idx = np.argmin(dists, axis=1)
        return self._y[idx]
