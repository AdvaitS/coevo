"""Surrogate models: fitting, prediction, and the coevolved wrapper."""

import numpy as np
import pytest

from coevo import (
    ClippedPredictor,
    CoevolvedPredictor,
    EvolvedPredictor,
    GaussianProcessSurrogate,
    NearestNeighborSurrogate,
    RBFSurrogate,
    RandomForestSurrogate,
    SymbolicRegressor,
)


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


def test_clipped_predictor_bounds_predictions():
    X, y = _data()
    pred = ClippedPredictor(RBFSurrogate()).fit(X, y)
    # Query far outside the training region; the RBF extrapolates wildly there,
    # but the clipped predictor must stay within the observed fitness range.
    Xq = np.random.default_rng(1).normal(size=(50, 3)) * 20
    out = pred.predict(Xq)
    assert out.min() >= y.min()
    assert out.max() <= y.max()


def test_gaussian_process_surrogate():
    pytest.importorskip("sklearn")
    X, y = _data()
    sur = GaussianProcessSurrogate().fit(X, y)
    pred = sur.predict(X)
    np.testing.assert_allclose(pred, y, atol=0.05)


def test_random_forest_surrogate():
    pytest.importorskip("sklearn")
    X, y = _data()
    sur = RandomForestSurrogate(n_estimators=50).fit(X, y)
    pred = sur.predict(X)
    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))


def test_symbolic_regressor_beats_mean_baseline():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(60, 3))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + X[:, 2] ** 2

    reg = SymbolicRegressor(population_size=200, generations=30, seed=0).fit(X, y)
    pred = reg.predict(X)
    gp_rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mean_rmse = float(np.sqrt(np.mean((np.full_like(y, y.mean()) - y) ** 2)))

    assert np.all(np.isfinite(pred))
    assert gp_rmse < mean_rmse
    assert "x" in reg.expression()


def test_evolved_predictor_tracks_error_and_expression():
    X, y = _data()
    pred = EvolvedPredictor(population_size=120, generations=15, seed=1)
    pred.fit(X, y)
    pred.fit(X[:10], y[:10])
    assert len(pred.error_trace) == 2
    assert isinstance(pred.expression, str)
    assert np.all(np.isfinite(pred.predict(X)))
