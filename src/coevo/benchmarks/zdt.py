"""ZDT multi-objective benchmarks (Zitzler–Deb–Thiele, 2000).

Each returns a :class:`~coevo.core.mo_problem.MultiObjectiveProblem` with two
objectives over ``[0, 1]^dim``, plus an analytic reference front (``*_front``)
for IGD evaluation.
"""

from __future__ import annotations

import numpy as np

from coevo.core.mo_problem import MultiObjectiveProblem


def _zdt1(x: np.ndarray) -> np.ndarray:
    f1 = x[:, 0]
    g = 1 + 9 * np.mean(x[:, 1:], axis=1)
    return np.column_stack([f1, g * (1 - np.sqrt(f1 / g))])


def _zdt2(x: np.ndarray) -> np.ndarray:
    f1 = x[:, 0]
    g = 1 + 9 * np.mean(x[:, 1:], axis=1)
    return np.column_stack([f1, g * (1 - (f1 / g) ** 2)])


def _zdt3(x: np.ndarray) -> np.ndarray:
    f1 = x[:, 0]
    g = 1 + 9 * np.mean(x[:, 1:], axis=1)
    return np.column_stack([f1, g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))])


def zdt1_front(n: int = 500) -> np.ndarray:
    t = np.linspace(0, 1, n)
    return np.column_stack([t, 1 - np.sqrt(t)])


def zdt2_front(n: int = 500) -> np.ndarray:
    t = np.linspace(0, 1, n)
    return np.column_stack([t, 1 - t**2])


def zdt3_front(n: int = 500) -> np.ndarray:
    t = np.linspace(0, 1, n)
    return np.column_stack([t, 1 - np.sqrt(t) - t * np.sin(10 * np.pi * t)])


def zdt1(dim: int = 30) -> MultiObjectiveProblem:
    return MultiObjectiveProblem(
        "zdt1", _zdt1, [0.0, 1.0], dim, 2, reference_point=np.array([11.0, 11.0])
    )


def zdt2(dim: int = 30) -> MultiObjectiveProblem:
    return MultiObjectiveProblem(
        "zdt2", _zdt2, [0.0, 1.0], dim, 2, reference_point=np.array([11.0, 11.0])
    )


def zdt3(dim: int = 30) -> MultiObjectiveProblem:
    return MultiObjectiveProblem(
        "zdt3", _zdt3, [0.0, 1.0], dim, 2, reference_point=np.array([2.0, 2.0])
    )
