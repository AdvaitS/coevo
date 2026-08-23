"""Pareto-dominance metrics and helpers."""

from __future__ import annotations

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if ``a`` dominates ``b`` (all objectives <=, at least one <)."""
    return bool(np.all(a <= b) and np.any(a < b))


def _dominance_matrix(objectives: np.ndarray) -> np.ndarray:
    """Return a boolean matrix ``D`` where ``D[i, j]`` means ``i`` dominates ``j``."""
    obj = np.asarray(objectives, dtype=float)
    le = obj[:, None, :] <= obj[None, :, :]
    sl = obj[:, None, :] < obj[None, :, :]
    dom = le.all(axis=-1) & sl.any(axis=-1)
    np.fill_diagonal(dom, False)
    return dom


def nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    """Return a boolean mask marking the non-dominated rows of ``objectives``."""
    objectives = np.asarray(objectives, dtype=float)
    if len(objectives) == 0:
        return np.zeros(0, dtype=bool)
    return ~_dominance_matrix(objectives).any(axis=0)


def nondominated(objectives: np.ndarray) -> np.ndarray:
    """Return the non-dominated subset of ``objectives``."""
    objectives = np.asarray(objectives, dtype=float)
    return objectives[nondominated_mask(objectives)]


def fast_non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Sort ``objectives`` into Pareto fronts (list of index lists)."""
    objectives = np.asarray(objectives, dtype=float)
    n = len(objectives)
    if n == 0:
        return []
    dom = _dominance_matrix(objectives)
    domination_count = dom.sum(axis=0)
    dominated_sets = [np.where(dom[i])[0].tolist() for i in range(n)]

    fronts: list[list[int]] = []
    current = np.where(domination_count == 0)[0].tolist()
    while current:
        fronts.append(current)
        next_front: list[int] = []
        for p in current:
            for q in dominated_sets[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        current = next_front
    return fronts


def crowding_distance(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    """Crowding distance for the indices in ``front`` (same order as ``front``)."""
    objectives = np.asarray(objectives, dtype=float)
    f = objectives[front]
    k, m = f.shape
    dist = np.zeros(k)
    if k <= 2:
        dist[:] = np.inf
        return dist
    for j in range(m):
        order = np.argsort(f[:, j])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        fmin, fmax = f[order[0], j], f[order[-1], j]
        if fmax - fmin == 0:
            continue
        for r in range(1, k - 1):
            dist[order[r]] += (f[order[r + 1], j] - f[order[r - 1], j]) / (fmax - fmin)
    return dist


def igd(front: np.ndarray, reference: np.ndarray) -> float:
    """Inverted generational distance (lower is better).

    The average distance from each reference point to its nearest point in the
    obtained front.
    """
    front = np.asarray(front, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(front) == 0:
        return np.inf
    diff = reference[:, None, :] - front[None, :, :]
    d = np.sqrt((diff**2).sum(axis=-1)).min(axis=1)
    return float(d.mean())


def hypervolume(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Hypervolume of ``front`` relative to ``ref_point`` (2-objective, exact).

    Higher is better: it measures the volume dominated by the front.
    """
    front = np.asarray(front, dtype=float)
    ref = np.asarray(ref_point, dtype=float)
    if front.shape[1] != 2:
        raise NotImplementedError("Exact hypervolume is implemented for 2 objectives only.")
    pts = front[np.all(front <= ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    xs: list[float] = []
    ys: list[float] = []
    best = np.inf
    for i in range(len(pts)):
        if pts[i, 1] < best:
            xs.append(pts[i, 0])
            ys.append(pts[i, 1])
            best = pts[i, 1]
    hv = 0.0
    for i in range(len(xs)):
        next_x = xs[i + 1] if i + 1 < len(xs) else ref[0]
        hv += (next_x - xs[i]) * (ref[1] - ys[i])
    return hv
