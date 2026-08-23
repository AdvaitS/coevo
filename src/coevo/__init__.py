"""coevo: surrogate-assisted and coevolutionary evolutionary computation.

``coevo`` is a small, dependency-light research library for evolutionary
algorithms that solve *expensive* black-box problems. Its distinctive feature
is a family of surrogate-assisted strategies that learn to predict fitness, so
the optimizer can spend its evaluation budget where it matters — an idea rooted
in coevolved fitness predictors (Schmidt & Lipson, 2008).
"""

from coevo import benchmarks
from coevo.algorithms import (
    DifferentialEvolution,
    GeneticAlgorithm,
    ParticleSwarmOptimization,
)
from coevo.core import OptimizationResult, Problem
from coevo.evaluation import SurrogateEvaluator, TrueEvaluator
from coevo.surrogates import (
    ClippedPredictor,
    CoevolvedPredictor,
    GaussianProcessSurrogate,
    NearestNeighborSurrogate,
    RBFSurrogate,
    RandomForestSurrogate,
)

__version__ = "0.2.0"

__all__ = [
    "Problem",
    "OptimizationResult",
    "DifferentialEvolution",
    "GeneticAlgorithm",
    "ParticleSwarmOptimization",
    "TrueEvaluator",
    "SurrogateEvaluator",
    "NearestNeighborSurrogate",
    "RBFSurrogate",
    "GaussianProcessSurrogate",
    "RandomForestSurrogate",
    "ClippedPredictor",
    "CoevolvedPredictor",
    "benchmarks",
]
