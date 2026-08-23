from coevo.surrogates.base import Surrogate
from coevo.surrogates.coevolved import CoevolvedPredictor
from coevo.surrogates.nearest import NearestNeighborSurrogate
from coevo.surrogates.rbf import RBFSurrogate

__all__ = [
    "Surrogate",
    "NearestNeighborSurrogate",
    "RBFSurrogate",
    "CoevolvedPredictor",
]
