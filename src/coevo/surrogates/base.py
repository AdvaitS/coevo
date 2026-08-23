"""Surrogate model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Surrogate(ABC):
    """A model that predicts fitness from decision vectors."""

    fitted: bool

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Surrogate":
        """Train on ``(n, dim)`` inputs ``X`` and ``(n,)`` targets ``y``."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict fitness for ``(n, dim)`` inputs ``X``."""
