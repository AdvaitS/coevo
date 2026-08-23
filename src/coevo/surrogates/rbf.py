"""Radial-basis-function surrogate backed by scipy."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator

from coevo.surrogates.base import Surrogate


class RBFSurrogate(Surrogate):
    """Interpolates fitness with a thin-plate radial basis function.

    A simple, exact interpolator that captures local structure better than the
    1-NN baseline while requiring no external machine-learning framework.

    .. caution::
        Thin-plate splines extrapolate unboundedly away from the training data,
        so an aggressive optimizer can *exploit* the surrogate's imaginary
        minima. For robust surrogate-assisted optimisation prefer
        :class:`~coevo.surrogates.nearest.NearestNeighborSurrogate`, or apply
        model management (see the roadmap).
    """

    def __init__(self, kernel: str = "thin_plate_spline", smoothing: float = 0.0) -> None:
        self.kernel = kernel
        self.smoothing = smoothing
        self.fitted = False
        self._model: RBFInterpolator | None = None
        self._constant = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RBFSurrogate":
        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        if len(X) < 2:
            self._model = None
            self._constant = float(y[0]) if len(y) else 0.0
            self.fitted = True
            return self
        try:
            self._model = RBFInterpolator(X, y, kernel=self.kernel, smoothing=self.smoothing)
        except np.linalg.LinAlgError:
            # Near-duplicate / collinear archive points make the system singular;
            # regularize slightly to stay robust as the population collapses.
            self._model = RBFInterpolator(X, y, kernel=self.kernel, smoothing=1e-6)
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        if not self.fitted:
            raise RuntimeError("Surrogate must be fitted before predict().")
        if self._model is None:
            return np.full(len(X), self._constant)
        return np.asarray(self._model(X), dtype=float).ravel()
