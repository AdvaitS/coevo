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
    NSGA2,
    ParticleSwarmOptimization,
)
from coevo.core import (
    MultiObjectiveProblem,
    MultiObjectiveResult,
    OptimizationResult,
    Problem,
)
from coevo.evaluation import (
    SurrogateEvaluator,
    SurrogateMultiObjectiveEvaluator,
    TrueEvaluator,
    TrueMultiObjectiveEvaluator,
)
from coevo.surrogates import (
    ClippedPredictor,
    CoevolvedPredictor,
    EvolvedPredictor,
    GaussianProcessSurrogate,
    NearestNeighborSurrogate,
    RBFSurrogate,
    RandomForestSurrogate,
    SymbolicRegressor,
)

__version__ = "0.4.0"

__all__ = [
    "Problem",
    "OptimizationResult",
    "MultiObjectiveProblem",
    "MultiObjectiveResult",
    "DifferentialEvolution",
    "GeneticAlgorithm",
    "ParticleSwarmOptimization",
    "NSGA2",
    "TrueEvaluator",
    "SurrogateEvaluator",
    "TrueMultiObjectiveEvaluator",
    "SurrogateMultiObjectiveEvaluator",
    "NearestNeighborSurrogate",
    "RBFSurrogate",
    "GaussianProcessSurrogate",
    "RandomForestSurrogate",
    "ClippedPredictor",
    "CoevolvedPredictor",
    "SymbolicRegressor",
    "EvolvedPredictor",
    "benchmarks",
]
