"""Efficient global optimization (Jones, Schonlau & Welch, 1998).

The reference method for expensive black-box problems, and the baseline any
surrogate-assisted evolutionary algorithm has to be measured against: fit a
Gaussian process to everything evaluated so far, then spend the next true
evaluation wherever expected improvement is highest.

EGO differs from this library's other algorithms in a way worth stating plainly,
because it decides which comparisons are fair. It is *sequential* -- one true
evaluation per iteration, each chosen using every point before it -- whereas
GA/DE/PSO evaluate a whole generation at a time. At a matched evaluation budget
EGO therefore extracts strictly more information per evaluation, and tends to
win at small budgets. It pays for that with cubic scaling in the number of
evaluations (the GP fit) and no parallelism, so the ranking reverses once
evaluations are cheap enough to batch or numerous enough to make the GP the
bottleneck.

Reported as an optimizer here so the benchmark can put the two side by side and
show where each is the right tool, rather than implying one dominates.
"""

from __future__ import annotations

import numpy as np

import warnings

from coevo.acquisition import acquisition
from coevo.algorithms.base import BaseAlgorithm
from coevo.core.problem import Problem
from coevo.core.result import OptimizationResult
from coevo.evaluation import Evaluator
from coevo.surrogates.gp import GaussianProcessSurrogate


class EfficientGlobalOptimization(BaseAlgorithm):
    """Bayesian optimization with a Gaussian process and an acquisition function.

    Parameters
    ----------
    n_initial:
        Size of the initial space-filling design, evaluated before any modelling.
    candidates:
        Candidate points sampled per iteration; the acquisition is maximised over
        this set rather than by an inner continuous optimizer, which keeps the
        implementation dependency-light and is standard practice for a baseline.
    acquisition_fn:
        ``"ei"`` (default), ``"lcb"`` or ``"pi"`` -- see :mod:`coevo.acquisition`.
    """

    def __init__(
        self,
        pop_size: int = 1,
        generations: int = 100,
        n_initial: int = 10,
        candidates: int = 500,
        acquisition_fn: str = "ei",
        xi: float = 0.01,
        seed: int = 0,
        max_evaluations: int | None = None,
        surrogate: GaussianProcessSurrogate | None = None,
    ) -> None:
        super().__init__(pop_size, generations, seed, max_evaluations)
        self.n_initial = n_initial
        self.candidates = candidates
        self.acquisition_fn = acquisition_fn
        self.xi = xi
        self.surrogate = surrogate

    def _acquire(self, mean: np.ndarray, std: np.ndarray, best: float) -> np.ndarray:
        fn = acquisition(self.acquisition_fn)
        if self.acquisition_fn == "lcb":
            return fn(mean, std)
        return fn(mean, std, best, self.xi)

    def optimize(
        self, problem: Problem, evaluator: Evaluator | None = None
    ) -> OptimizationResult:
        evaluator = self._resolve_evaluator(problem, evaluator)
        rng = np.random.default_rng(self.seed)
        lo, hi = problem.lower, problem.upper
        dim = problem.dim

        budget = self.max_evaluations if self.max_evaluations is not None else (
            self.n_initial + self.generations
        )
        n_initial = max(2, min(self.n_initial, budget))

        X = rng.uniform(lo, hi, size=(n_initial, dim))
        y = np.asarray(evaluator(X), dtype=float).ravel()

        best_idx = int(np.argmin(y))
        best_x, best_f = X[best_idx].copy(), float(y[best_idx])
        history = [best_f]

        surrogate = self.surrogate or GaussianProcessSurrogate()
        while evaluator.n_true < budget:
            try:
                with warnings.catch_warnings():
                    # sklearn's GP warns when a kernel hyperparameter converges to
                    # its bound. That is expected on bounded benchmark functions
                    # and is not actionable here.
                    warnings.simplefilter("ignore")
                    surrogate.fit(X, y)
                candidates = rng.uniform(lo, hi, size=(self.candidates, dim))
                mean, std = surrogate.predict_with_std(candidates)
                scores = self._acquire(mean, std, best_f)
                nxt = candidates[int(np.argmax(scores))]
            except Exception:
                # A GP fit can fail on degenerate data (duplicate points, a
                # singular kernel matrix). Falling back to a random probe keeps
                # the run going and adds the information that unsticks it,
                # rather than aborting a benchmark mid-sweep.
                nxt = rng.uniform(lo, hi, size=dim)

            value = float(np.asarray(evaluator(nxt.reshape(1, -1))).ravel()[0])
            X = np.vstack([X, nxt])
            y = np.append(y, value)
            if value < best_f:
                best_f, best_x = value, nxt.copy()
            history.append(best_f)

        true_best = float(problem.evaluate_noiseless(best_x)[0])
        return OptimizationResult(
            problem,
            best_x,
            true_best,
            history,
            evaluator.n_true,
            metadata={"selected_fitness": best_f, "acquisition": self.acquisition_fn},
        )
