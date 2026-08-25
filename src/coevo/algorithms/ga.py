"""Real-coded genetic algorithm with tournament selection and elitism."""

from __future__ import annotations

import numpy as np

from coevo.algorithms.base import BaseAlgorithm
from coevo.core.problem import Problem
from coevo.core.result import OptimizationResult
from coevo.evaluation import Evaluator


class GeneticAlgorithm(BaseAlgorithm):
    """A simple real-coded GA: tournament selection, arithmetic crossover,
    Gaussian mutation, and elitism.

    ``mutation_sigma`` is a *fraction of each dimension's range*, not an
    absolute step. An absolute default cannot be right for every problem: 0.1
    is a reasonable step on sphere's [-5.12, 5.12] but only 0.15% of ackley's
    [-32.768, 32.768], which is far too small to escape a local basin. That is
    what kept this GA pinned at 8.53 on ackley(5) no matter how long it ran.
    """

    def __init__(
        self,
        pop_size: int = 50,
        generations: int = 100,
        crossover_p: float = 0.9,
        mutation_p: float = 0.1,
        mutation_sigma: float = 0.01,
        tournament_size: int = 3,
        elite: int = 1,
        seed: int = 0,
        max_evaluations: int | None = None,
    ) -> None:
        super().__init__(pop_size, generations, seed, max_evaluations)
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
        is_true = evaluator.true_mask(n).copy()
        history = [float(np.min(fitness))]

        for _ in range(self.generations):
            if self.max_evaluations is not None and evaluator.n_true >= self.max_evaluations:
                break
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

            # Gaussian mutation, scaled to each dimension's range.
            mask = rng.random(offspring.shape) < self.mutation_p
            noise = rng.normal(0.0, self.mutation_sigma * (hi - lo), offspring.shape)
            offspring = np.clip(np.where(mask, offspring + noise, offspring), lo, hi)

            f_off = evaluator(offspring)
            off_is_true = evaluator.true_mask(len(offspring))

            # Generational replacement with elitism: the best `elite` parents
            # survive and the rest of the population is replaced by offspring.
            #
            # Sorting [parents; offspring] and taking the top n instead -- which
            # is what this used to do -- is (mu+lambda) truncation selection, and
            # is far greedier than the docstring's "tournament selection with
            # elitism". It converges the population within a few dozen
            # generations and then cannot escape: on ackley(5) the run sat at
            # 8.5348 from generation 50 through generation 1000, unmoved by 20x
            # more budget.
            elite = max(0, min(self.elite, n))
            keep = np.argsort(fitness)[:elite]
            pop = np.vstack([pop[keep], offspring])[:n]
            fitness = np.concatenate([fitness[keep], f_off])[:n]
            is_true = np.concatenate([is_true[keep], off_is_true])[:n]

            # Verify the incumbent before recording it, so the reported best is
            # never a surrogate artefact carried forward from an earlier batch.
            best_idx = int(np.argmin(fitness))
            if not is_true[best_idx]:
                fitness[best_idx] = float(
                    np.asarray(evaluator.evaluate_true(pop[best_idx : best_idx + 1])).ravel()[0]
                )
                is_true[best_idx] = True

            history.append(float(np.min(fitness)))

        best_idx = int(np.argmin(fitness))
        best_x = pop[best_idx].copy()
        true_best = float(problem.evaluate_noiseless(best_x)[0])
        return OptimizationResult(
            problem,
            best_x,
            true_best,
            history,
            evaluator.n_true,
            metadata={"selected_fitness": float(fitness[best_idx])},
        )
