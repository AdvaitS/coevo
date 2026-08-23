"""NSGA-II: the non-dominated sorting genetic algorithm."""

from __future__ import annotations

import numpy as np

from coevo.algorithms.base import BaseAlgorithm
from coevo.core.metrics import (
    crowding_distance,
    fast_non_dominated_sort,
    hypervolume,
    nondominated_mask,
)
from coevo.core.mo_problem import MultiObjectiveProblem
from coevo.core.mo_result import MultiObjectiveResult
from coevo.evaluation.mo_evaluator import (
    MultiObjectiveEvaluator,
    TrueMultiObjectiveEvaluator,
)


class NSGA2(BaseAlgorithm):
    """NSGA-II with simulated-binary crossover and polynomial mutation.

    Uses the canonical Deb et al. (2002) operators for continuous variables.
    """

    def __init__(
        self,
        pop_size: int = 100,
        generations: int = 100,
        crossover_p: float = 0.9,
        mutation_p: float = 1.0,
        eta_c: float = 20.0,
        eta_m: float = 20.0,
        seed: int = 0,
        max_evaluations: int | None = None,
    ) -> None:
        super().__init__(pop_size, generations, seed, max_evaluations)
        self.crossover_p = crossover_p
        self.mutation_p = mutation_p
        self.eta_c = eta_c
        self.eta_m = eta_m

    def _tournament(
        self, rng: np.random.Generator, rank: np.ndarray, crowd: np.ndarray
    ) -> int:
        n = len(rank)
        i, j = rng.integers(0, n, size=2)
        if rank[i] < rank[j] or (rank[i] == rank[j] and crowd[i] > crowd[j]):
            return int(i)
        return int(j)

    def _variation(
        self,
        rng: np.random.Generator,
        pop: np.ndarray,
        rank: np.ndarray,
        crowd: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> np.ndarray:
        n, dim = pop.shape
        offspring = np.empty_like(pop)
        for i in range(n):
            p1 = pop[self._tournament(rng, rank, crowd)]
            p2 = pop[self._tournament(rng, rank, crowd)]
            if rng.random() < self.crossover_p:
                c1, _ = self._sbx(rng, p1, p2, lo, hi)
            else:
                c1 = p1.copy()
            offspring[i] = self._poly_mutate(rng, c1, lo, hi)
        return offspring

    def _sbx(
        self, rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray, lo: np.ndarray, hi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        u = rng.random(len(p1))
        beta = np.where(
            u <= 0.5,
            (2 * u) ** (1.0 / (self.eta_c + 1)),
            (1.0 / (2 * (1 - u))) ** (1.0 / (self.eta_c + 1)),
        )
        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

    def _poly_mutate(
        self, rng: np.random.Generator, x: np.ndarray, lo: np.ndarray, hi: np.ndarray
    ) -> np.ndarray:
        prob = self.mutation_p / max(len(x), 1)
        mask = rng.random(len(x)) < prob
        u = rng.random(len(x))
        delta = np.where(
            u < 0.5,
            (2 * u) ** (1.0 / (self.eta_m + 1)) - 1.0,
            1.0 - (2 * (1 - u)) ** (1.0 / (self.eta_m + 1)),
        )
        y = np.where(mask, x + delta * (hi - lo), x)
        return np.clip(y, lo, hi)

    def optimize(
        self, problem: MultiObjectiveProblem, evaluator: MultiObjectiveEvaluator | None = None
    ) -> MultiObjectiveResult:
        evaluator = evaluator if evaluator is not None else TrueMultiObjectiveEvaluator(problem)
        rng = np.random.default_rng(self.seed)
        lo, hi = problem.lower, problem.upper
        dim, n = problem.dim, self.pop_size

        pop = rng.uniform(lo, hi, size=(n, dim))
        obj = evaluator(pop)

        def _hv() -> float:
            if problem.reference_point is None:
                return float(len(obj))
            front = obj[nondominated_mask(obj)]
            return hypervolume(front, problem.reference_point)

        history = [_hv()]

        for _ in range(self.generations):
            if self.max_evaluations is not None and evaluator.n_true >= self.max_evaluations:
                break
            fronts = fast_non_dominated_sort(obj)
            rank = np.empty(n, dtype=int)
            crowd = np.zeros(n)
            for fi, front in enumerate(fronts):
                rank[front] = fi
                crowd[front] = crowding_distance(obj, front)

            offspring = self._variation(rng, pop, rank, crowd, lo, hi)
            off_obj = evaluator(offspring)

            combined = np.vstack([pop, offspring])
            comb_obj = np.vstack([obj, off_obj])

            fronts = fast_non_dominated_sort(comb_obj)
            new_pop: list[np.ndarray] = []
            new_obj: list[np.ndarray] = []
            count = 0
            fi = 0
            while fi < len(fronts) and count + len(fronts[fi]) <= n:
                idx = fronts[fi]
                new_pop.append(combined[idx])
                new_obj.append(comb_obj[idx])
                count += len(idx)
                fi += 1
            if count < n and fi < len(fronts):
                idx = fronts[fi]
                cd = crowding_distance(comb_obj, idx)
                order = np.argsort(cd)[::-1]
                take = n - count
                new_pop.append(combined[idx][order[:take]])
                new_obj.append(comb_obj[idx][order[:take]])

            pop = np.concatenate(new_pop)
            obj = np.concatenate(new_obj)
            history.append(_hv())

        mask = nondominated_mask(obj)
        return MultiObjectiveResult(
            problem,
            pop[mask].copy(),
            obj[mask].copy(),
            history,
            evaluator.n_true,
        )
