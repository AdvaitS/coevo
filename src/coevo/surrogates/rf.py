"""Random-forest surrogate (sklearn-backed, optional dependency)."""

from __future__ import annotations

import numpy as np

from coevo.surrogates.base import Surrogate


class RandomForestSurrogate(Surrogate):
    """Random-forest regression surrogate.

    Requires ``scikit-learn`` (install with ``pip install coevo[sklearn]``).
    Fast to fit and robust to noisy objectives, at the cost of being a weaker
    extrapolator than a GP or RBF.
    """

    def __init__(self, n_estimators: int = 100, **kwargs) -> None:
        self.n_estimators = n_estimators
        self.kwargs = kwargs
        self.fitted = False
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestSurrogate":
        from sklearn.ensemble import RandomForestRegressor

        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        self._model = RandomForestRegressor(n_estimators=self.n_estimators, **self.kwargs)
        self._model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        if not self.fitted or self._model is None:
            raise RuntimeError("Surrogate must be fitted before predict().")
        return np.asarray(self._model.predict(X), dtype=float).ravel()
