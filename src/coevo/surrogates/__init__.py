from coevo.surrogates.base import Surrogate
from coevo.surrogates.clipped import ClippedPredictor
from coevo.surrogates.coevolved import CoevolvedPredictor
from coevo.surrogates.evolved import EvolvedPredictor, SymbolicRegressor
from coevo.surrogates.gp import GaussianProcessSurrogate
from coevo.surrogates.nearest import NearestNeighborSurrogate
from coevo.surrogates.rbf import RBFSurrogate
from coevo.surrogates.rf import RandomForestSurrogate

__all__ = [
    "Surrogate",
    "NearestNeighborSurrogate",
    "RBFSurrogate",
    "GaussianProcessSurrogate",
    "RandomForestSurrogate",
    "ClippedPredictor",
    "CoevolvedPredictor",
    "SymbolicRegressor",
    "EvolvedPredictor",
]
