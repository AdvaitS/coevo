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


# --- GP engine correctness regressions -------------------------------------


def test_symbolic_regressor_treats_1d_x_as_one_feature():
    """A 1-D X is n samples of one feature, not one sample of n features.

    np.atleast_2d would reshape (n,) to (1, n), silently fitting a model over n
    imaginary features to a single data point.
    """
    from coevo.surrogates.evolved import SymbolicRegressor

    x = np.linspace(0.0, 3.0, 40)
    model = SymbolicRegressor(population_size=60, generations=8, seed=0).fit(x, 2.0 * x)
    assert model.predict(x).shape == (40,)
    # only feature x0 can appear; x1.. would mean the shape was misread
    assert "x1" not in model.expression()


def test_symbolic_regressor_rejects_length_mismatch():
    from coevo.surrogates.evolved import SymbolicRegressor

    with pytest.raises(ValueError, match="length mismatch"):
        SymbolicRegressor(population_size=20, generations=2).fit(
            np.zeros((10, 1)), np.zeros(9)
        )


def test_crossover_respects_max_size():
    """Subtree crossover is the dominant bloat source; it must honour max_size."""
    from coevo.surrogates.evolved import SymbolicRegressor, _size

    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 5.0, size=(80, 1))
    y = X[:, 0] ** 1.5
    for seed in range(6):
        model = SymbolicRegressor(
            population_size=150,
            generations=25,
            seed=seed,
            parsimony=0.0,  # no size pressure: only the hard bound applies
            max_size=20,
            refine=False,
        ).fit(X, y)
        assert _size(model.best_) <= 20, f"seed {seed}: {_size(model.best_)} nodes > max_size"


def test_simplify_preserves_protected_operator_semantics():
    """exp(log(x)) is not x for the protected log, so the rewrite must not fire."""
    from coevo.surrogates.evolved import (
        _DEFAULT_FUNCTIONS,
        Node,
        _eval,
        _simplify_fixpoint,
    )

    tree = Node("exp", args=(Node("log", args=(Node("x", idx=0),)),))
    X = np.array([[-3.0], [0.0], [2.0]])
    before = _eval(tree, X, _DEFAULT_FUNCTIONS)
    after = _eval(_simplify_fixpoint(tree, _DEFAULT_FUNCTIONS), X, _DEFAULT_FUNCTIONS)
    np.testing.assert_allclose(before, after)


def test_reported_rmse_matches_predict():
    """best_rmse_ must describe the model that predict() actually evaluates."""
    from coevo.surrogates.evolved import SymbolicRegressor

    rng = np.random.default_rng(1)
    X = rng.uniform(-2.0, 2.0, size=(60, 1))
    y = np.sin(X[:, 0]) + 0.05 * rng.normal(size=60)
    for seed in range(8):
        model = SymbolicRegressor(population_size=100, generations=12, seed=seed).fit(X, y)
        actual = float(np.sqrt(np.mean((model.predict(X) - y) ** 2)))
        assert actual == pytest.approx(model.best_rmse_, rel=1e-6, abs=1e-9)


def test_refine_constants_survives_more_params_than_points():
    """Levenberg-Marquardt needs n_residuals >= n_params; refinement must not raise."""
    from coevo.surrogates.evolved import Node, SymbolicRegressor

    model = SymbolicRegressor(population_size=10, generations=1, seed=0)
    model.best_ = Node(
        "+",
        args=(
            Node("+", args=(Node("c", val=1.0), Node("c", val=2.0))),
            Node("+", args=(Node("c", val=3.0), Node("x", idx=0))),
        ),
    )
    model.refine_constants(np.array([[0.0], [1.0]]), np.array([0.5, 1.5]))  # 3 consts, 2 points


def test_pareto_front_models_predict_on_new_data():
    """Front entries must carry the model, not just its rendering."""
    from coevo.surrogates.evolved import SymbolicRegressor

    rng = np.random.default_rng(0)
    X = rng.uniform(0.5, 6.0, size=(70, 1))
    y = 3.0 * np.log(X[:, 0]) + 1.0
    model = SymbolicRegressor(population_size=200, generations=25, seed=0, parsimony=0.01).fit(X, y)

    front = model.pareto_front()
    assert front, "hall of fame should not be empty"
    assert [m.complexity for m in front] == sorted(m.complexity for m in front)
    assert [m.rmse for m in front] == sorted((m.rmse for m in front), reverse=True)
    for entry in front:
        # the recorded rmse must be the one predict() actually achieves
        actual = float(np.sqrt(np.mean((entry.predict(X) - y) ** 2)))
        assert actual == pytest.approx(entry.rmse, rel=1e-6, abs=1e-9)
        assert entry.predict(X[:5]).shape == (5,)


def test_linear_scaling_expression_and_predict_agree():
    from coevo.surrogates.evolved import SymbolicRegressor

    rng = np.random.default_rng(2)
    X = rng.uniform(0.0, 4.0, size=(60, 1))
    y = 5.0 * X[:, 0] - 2.0
    model = SymbolicRegressor(population_size=150, generations=20, seed=0).fit(X, y)
    actual = float(np.sqrt(np.mean((model.predict(X) - y) ** 2)))
    assert actual == pytest.approx(model.best_rmse_, rel=1e-6, abs=1e-9)
    # complexity accounts for the affine wrapper it reports in the expression
    from coevo.surrogates.evolved import _AFFINE_NODES, _size

    assert model.complexity == _size(model.best_) + _AFFINE_NODES
