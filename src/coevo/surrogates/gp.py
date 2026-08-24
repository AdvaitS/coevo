"""Gaussian-process surrogate (sklearn-backed, optional dependency)."""

from __future__ import annotations

import numpy as np

from coevo.surrogates.base import Surrogate


class GaussianProcessSurrogate(Surrogate):
    """Gaussian-process regression surrogate.

    Requires ``scikit-learn`` (install with ``pip install coevo[sklearn]``).

    Unlike the other surrogates this one is *probabilistic*: it returns a
    predictive standard deviation alongside the mean. That is what makes
    uncertainty-aware model management possible -- an acquisition function can
    then trade exploitation of a promising mean against exploration of an
    uncertain region, rather than chasing the mean alone and getting stuck.
    See :mod:`coevo.acquisition`.
    """

    def __init__(self, alpha: float = 1e-8, normalize_y: bool = True, **kwargs) -> None:
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.kwargs = kwargs
        self.fitted = False
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessSurrogate":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel

        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        kernel = self.kwargs.pop("kernel", ConstantKernel(1.0) * RBF(1.0))
        self._model = GaussianProcessRegressor(
            kernel=kernel, alpha=self.alpha, normalize_y=self.normalize_y, **self.kwargs
        )
        self._model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        if not self.fitted or self._model is None:
            raise RuntimeError("Surrogate must be fitted before predict().")
        return np.asarray(self._model.predict(X), dtype=float).ravel()

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, standard deviation)`` of the posterior at ``X``."""
        X = np.atleast_2d(X)
        if not self.fitted or self._model is None:
            raise RuntimeError("Surrogate must be fitted before predict().")
        mean, std = self._model.predict(X, return_std=True)
        return np.asarray(mean, dtype=float).ravel(), np.asarray(std, dtype=float).ravel()
