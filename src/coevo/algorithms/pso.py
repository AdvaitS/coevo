"""Particle swarm optimization (global best topology)."""

from __future__ import annotations

import numpy as np

from coevo.algorithms.base import BaseAlgorithm
from coevo.core.problem import Problem
from coevo.core.result import OptimizationResult
from coevo.evaluation import Evaluator


class ParticleSwarmOptimization(BaseAlgorithm):
    """Canonical PSO with inertia weight and global best topology.

    Default coefficients (``w=0.7298``, ``c1=c2=1.49618``) follow the Clerc
    constriction coefficient for good default behaviour.
    """

    def __init__(
        self,
        pop_size: int = 50,
        generations: int = 100,
        w: float = 0.7298,
        c1: float = 1.49618,
        c2: float = 1.49618,
        seed: int = 0,
    ) -> None:
        super().__init__(pop_size, generations, seed)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize(
        self, problem: Problem, evaluator: Evaluator | None = None
    ) -> OptimizationResult:
        evaluator = self._resolve_evaluator(problem, evaluator)
        rng = np.random.default_rng(self.seed)
        lo, hi = problem.lower, problem.upper
        dim, n = problem.dim, self.pop_size

        pos = rng.uniform(lo, hi, size=(n, dim))
        vel = np.zeros((n, dim))
        fit = evaluator(pos)

        pbest = pos.copy()
        pbest_f = fit.copy()
        gbest_idx = int(np.argmin(pbest_f))
        gbest = pbest[gbest_idx].copy()
        gbest_f = float(pbest_f[gbest_idx])
        history = [gbest_f]

        for _ in range(self.generations):
            r1 = rng.random((n, dim))
            r2 = rng.random((n, dim))
            vel = self.w * vel + self.c1 * r1 * (pbest - pos) + self.c2 * r2 * (gbest - pos)
            pos = np.clip(pos + vel, lo, hi)
            fit = evaluator(pos)

            improved = fit < pbest_f
            pbest[improved] = pos[improved]
            pbest_f[improved] = fit[improved]

            gbest_idx = int(np.argmin(pbest_f))
            if pbest_f[gbest_idx] < gbest_f:
                gbest_f = float(pbest_f[gbest_idx])
                gbest = pbest[gbest_idx].copy()
            history.append(gbest_f)

        true_best = float(problem.evaluate(gbest)[0])
        return OptimizationResult(
            problem,
            gbest,
            true_best,
            history,
            evaluator.n_true,
            metadata={"selected_fitness": float(gbest_f)},
        )
