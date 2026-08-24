"""Acquisition functions: choosing where to spend the next true evaluation.

Every other model-management strategy in this library ranks candidates by the
surrogate's *mean* prediction. That is exploitation only, and it has a specific
failure mode: the search concentrates where the model already looks good, the
model never learns anything about the regions it is uncertain about, and a
better optimum somewhere else is never found.

An acquisition function scores a candidate by mean *and* uncertainty together.
Expected improvement -- the basis of efficient global optimisation (Jones,
Schonlau & Welch, *J. Global Optimization* 13, 1998) -- is the reference method
for expensive black-box problems and the baseline this library was missing:
under a Gaussian posterior it has a closed form, so it costs nothing beyond a
predictive standard deviation.

All functions here are written for **minimisation** and follow the convention
that *larger is better*, so an optimizer picks candidates by ``argmax``.
"""

from __future__ import annotations

import math

import numpy as np


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    # Vectorised standard normal CDF via the error function, so scipy is not
    # required on this path.
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def expected_improvement(
    mean: np.ndarray, std: np.ndarray, best: float, xi: float = 0.01
) -> np.ndarray:
    """Expected improvement over the incumbent ``best`` (for minimisation).

    ``EI = (best - mean - xi)·Φ(z) + std·φ(z)``, with ``z = (best - mean - xi)/std``.
    The first term rewards a promising mean, the second rewards uncertainty --
    which is exactly the balance a mean-only ranking cannot express. ``xi`` adds
    a small exploration margin.

    Zero where ``std`` is zero: a point the model is certain about offers no
    expected improvement, however good its mean, because there is nothing left
    to learn there.
    """
    mean = np.asarray(mean, dtype=float).ravel()
    std = np.asarray(std, dtype=float).ravel()
    out = np.zeros_like(mean)
    positive = std > 1e-12
    if not np.any(positive):
        return out
    improvement = best - mean[positive] - xi
    z = improvement / std[positive]
    out[positive] = improvement * _normal_cdf(z) + std[positive] * _normal_pdf(z)
    return np.maximum(out, 0.0)


def lower_confidence_bound(mean: np.ndarray, std: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    """Negated LCB, so larger is better (for minimisation).

    Simpler and cheaper than expected improvement, with ``kappa`` setting the
    exploration weight explicitly rather than implicitly.
    """
    mean = np.asarray(mean, dtype=float).ravel()
    std = np.asarray(std, dtype=float).ravel()
    return -(mean - kappa * std)


def probability_of_improvement(
    mean: np.ndarray, std: np.ndarray, best: float, xi: float = 0.01
) -> np.ndarray:
    """Probability that a candidate improves on ``best`` (for minimisation).

    Included for comparison: it is famously over-exploitative, because a tiny
    near-certain gain scores higher than a large uncertain one.
    """
    mean = np.asarray(mean, dtype=float).ravel()
    std = np.asarray(std, dtype=float).ravel()
    out = np.zeros_like(mean)
    positive = std > 1e-12
    if np.any(positive):
        out[positive] = _normal_cdf((best - mean[positive] - xi) / std[positive])
    return out


ACQUISITIONS = {
    "ei": expected_improvement,
    "lcb": lower_confidence_bound,
    "pi": probability_of_improvement,
}


def acquisition(name: str):
    """Resolve an acquisition function by name."""
    if name not in ACQUISITIONS:
        raise ValueError(f"unknown acquisition {name!r}; expected one of {sorted(ACQUISITIONS)}")
    return ACQUISITIONS[name]
