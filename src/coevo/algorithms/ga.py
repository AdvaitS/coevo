"""Real-coded genetic algorithm with tournament selection and elitism."""

from __future__ import annotations

import numpy as np

from coevo.algorithms.base import BaseAlgorithm
from coevo.core.problem import Problem
from coevo.core.result import OptimizationResult
from coevo.evaluation import Evaluator


class GeneticAlgorithm(BaseAlgorithm):
    """A simple real-coded GA: tournament selection, arithmetic crossover,
    Gaussian mutation, and elitism."""

    def __init__(
        self,
        pop_size: int = 50,
        generations: int = 100,
        crossover_p: float = 0.9,
        mutation_p: float = 0.1,
        mutation_sigma: float = 0.1,
        tournament_size: int = 3,
        elite: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__(pop_size, generations, seed)
        self.crossover_p = crossover_p
        self.mutation_p = mutation_p
        self.mutation_sigma = mutation_sigma
        self.tournament_size = tournament_size
        self.elite = elite

    def _tournament(
        self, rng: np.random.Generator, pop: np.ndarray, fitness: np.ndarray, count: int
    ) -> np.ndarray:
        n = len(pop)
        selected = np.empty((count, pop.shape[1]))
        for i in range(count):
            idx = rng.choice(n, self.tournament_size, replace=False)
            selected[i] = pop[idx[np.argmin(fitness[idx])]]
        return selected

    def optimize(
        self, problem: Problem, evaluator: Evaluator | None = None
    ) -> OptimizationResult:
        evaluator = self._resolve_evaluator(problem, evaluator)
        rng = np.random.default_rng(self.seed)
        lo, hi = problem.lower, problem.upper
        dim, n = problem.dim, self.pop_size

        pop = rng.uniform(lo, hi, size=(n, dim))
        fitness = evaluator(pop)
        history = [float(np.min(fitness))]

        for _ in range(self.generations):
            n_parents = max(1, n - self.elite)
            parents = self._tournament(rng, pop, fitness, n_parents)

            # Arithmetic crossover between randomly paired parents.
            order = rng.permutation(n_parents)
            offspring = parents[order].copy()
            for i in range(0, n_parents - 1, 2):
                if rng.random() < self.crossover_p:
                    alpha = rng.random()
                    x, y = offspring[i], offspring[i + 1]
                    offspring[i] = alpha * x + (1 - alpha) * y
                    offspring[i + 1] = (1 - alpha) * x + alpha * y

            # Gaussian mutation.
            mask = rng.random(offspring.shape) < self.mutation_p
            noise = rng.normal(0.0, self.mutation_sigma, offspring.shape)
            offspring = np.clip(np.where(mask, offspring + noise, offspring), lo, hi)

            f_off = evaluator(offspring)

            # Elitism: keep the best `elite` parents plus the best offspring.
            combined = np.vstack([pop, offspring])
            f_combined = np.concatenate([fitness, f_off])
            order = np.argsort(f_combined)
            pop = combined[order[:n]]
            fitness = f_combined[order[:n]]

            history.append(float(fitness[0]))

        best_idx = int(np.argmin(fitness))
        best_x = pop[best_idx].copy()
        true_best = float(problem.evaluate(best_x)[0])
        return OptimizationResult(
            problem,
            best_x,
            true_best,
            history,
            evaluator.n_true,
            metadata={"selected_fitness": float(fitness[best_idx])},
        )
