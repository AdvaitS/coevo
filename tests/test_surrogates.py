"""Surrogate models: fitting, prediction, and the coevolved wrapper."""

import numpy as np
import pytest

from coevo import CoevolvedPredictor, NearestNeighborSurrogate, RBFSurrogate


def _data(n=30, d=3):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2
    return X, y


@pytest.mark.parametrize(
    "surrogate",
    [NearestNeighborSurrogate(), RBFSurrogate()],
)
def test_predict_requires_fit(surrogate):
    X, _ = _data()
    with pytest.raises(RuntimeError):
        surrogate.predict(X)


def test_nearest_neighbor_is_exact_on_train_points():
    X, y = _data()
    sur = NearestNeighborSurrogate().fit(X, y)
    np.testing.assert_allclose(sur.predict(X), y, rtol=1e-12)


def test_rbf_interpolates_train_points():
    X, y = _data()
    sur = RBFSurrogate().fit(X, y)
    np.testing.assert_allclose(sur.predict(X), y, atol=1e-8)


def test_coevolved_predictor_records_error_trace():
    X, y = _data()
    pred = CoevolvedPredictor(RBFSurrogate())
    pred.fit(X, y)
    pred.fit(X[:10], y[:10])
    assert len(pred.error_trace) == 2
    assert pred.error_trace[-1] < 1e-6  # exact on its own training set
