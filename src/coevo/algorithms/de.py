"""Differential evolution (DE/rand/1/bin)."""

from __future__ import annotations

import numpy as np

from coevo.algorithms.base import BaseAlgorithm
from coevo.core.problem import Problem
from coevo.core.result import OptimizationResult
from coevo.evaluation import Evaluator


class DifferentialEvolution(BaseAlgorithm):
    """Classic differential evolution with the rand/1/bin scheme.

    Updates are applied synchronously per generation (all trials are generated,
    then evaluated as a batch), which keeps evaluation batch-friendly for
    surrogate-assisted operation.
    """

    def __init__(
        self,
        pop_size: int = 50,
        generations: int = 100,
        F: float = 0.8,
        CR: float = 0.9,
        seed: int = 0,
        max_evaluations: int | None = None,
    ) -> None:
        super().__init__(pop_size, generations, seed, max_evaluations)
        self.F = F
        self.CR = CR

    @staticmethod
    def _distinct_indices(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
        """Return ``(k, n)`` indices where each column has ``k`` distinct
        values, none equal to the column index."""
        idx = np.empty((k, n), dtype=int)
        pool = np.arange(n)
        for j in range(n):
            idx[:, j] = rng.choice(np.delete(pool, j), size=k, replace=False)
        return idx

    def optimize(
        self, problem: Problem, evaluator: Evaluator | None = None
    ) -> OptimizationResult:
        evaluator = self._resolve_evaluator(problem, evaluator)
        rng = np.random.default_rng(self.seed)
        lo, hi = problem.lower, problem.upper
        dim, n = problem.dim, self.pop_size

        pop = rng.uniform(lo, hi, size=(n, dim))
        fitness = evaluator(pop)
        # Which entries of `fitness` came from the true objective. Predicted
        # values are carried across generations by the selection below, so an
        # optimistic prediction would otherwise sit in the population forever,
        # never re-checked and impossible to displace.
        is_true = evaluator.true_mask(n).copy()
        best_idx = int(np.argmin(fitness))
        best_x = pop[best_idx].copy()
        best_f = float(fitness[best_idx])
        history = [best_f]

        for _ in range(self.generations):
            if self.max_evaluations is not None and evaluator.n_true >= self.max_evaluations:
                break
            a = self._distinct_indices(rng, n, 3)
            mutant = pop[a[0]] + self.F * (pop[a[1]] - pop[a[2]])

            mask = rng.random((n, dim)) < self.CR
            mask[np.arange(n), rng.integers(0, dim, size=n)] = True
            trial = np.clip(np.where(mask, mutant, pop), lo, hi)

            f_trial = evaluator(trial)
            trial_is_true = evaluator.true_mask(n)
            improved = f_trial <= fitness
            pop[improved] = trial[improved]
            fitness[improved] = f_trial[improved]
            is_true[improved] = trial_is_true[improved]

            new_best = int(np.argmin(fitness))
            # Verify the incumbent before believing it. Without this the run's
            # best-so-far can be a surrogate artefact that never existed.
            if not is_true[new_best]:
                fitness[new_best] = float(
                    np.asarray(evaluator.evaluate_true(pop[new_best : new_best + 1])).ravel()[0]
                )
                is_true[new_best] = True
                new_best = int(np.argmin(fitness))
            if fitness[new_best] < best_f:
                best_f = float(fitness[new_best])
                best_x = pop[new_best].copy()
            history.append(best_f)

        true_best = float(problem.evaluate_noiseless(best_x)[0])
        return OptimizationResult(
            problem,
            best_x,
            true_best,
            history,
            evaluator.n_true,
            metadata={"selected_fitness": float(best_f)},
        )
