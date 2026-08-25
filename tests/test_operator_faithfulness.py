"""The rendered expression must describe what the model actually computes.

Protected operators exist to stop overflow, but a clamp that binds on ordinary
data silently changes the function. That matters more than it sounds: the
rendered string, the sympy conversion used to score "did we recover the law",
and the numeric model all have to agree, or the metric measures a different
function than the one that was fitted.
"""

import numpy as np
import pytest

from coevo.surrogates.evolved import _DEFAULT_FUNCTIONS as BASE_FUNCTIONS
from coevo.surrogates.evolved import Node, _raw_eval


def _tree(op, *args):
    return Node(op, tuple(args))


C = lambda v: Node("c", val=v)
X0 = Node("x", idx=0)


def test_log_of_exp_is_the_identity_over_realistic_data():
    """The exploit this closes: with a tight clamp, log(exp(u)) computes
    min(u, clamp) -- a free saturating nonlinearity that the rendering does not
    show and that sympy, which simplifies log(exp(u)) to u, cannot see.
    """
    X = np.linspace(0.0, 12.0, 60).reshape(-1, 1)
    tree = _tree("log", _tree("exp", _tree("*", X0, X0)))
    got = _raw_eval(tree, X, BASE_FUNCTIONS)
    np.testing.assert_allclose(got, X[:, 0] ** 2, rtol=1e-9)


def test_exp_still_guards_against_overflow():
    """Widening the clamp must not reintroduce inf/nan."""
    X = np.array([[0.0], [1e5], [-1e5]])
    got = _raw_eval(_tree("exp", X0), X, BASE_FUNCTIONS)
    assert got is None or np.all(np.isfinite(got))


def test_exp_matches_numpy_where_it_does_not_overflow():
    X = np.linspace(-20.0, 20.0, 41).reshape(-1, 1)
    got = _raw_eval(_tree("exp", X0), X, BASE_FUNCTIONS)
    np.testing.assert_allclose(got, np.exp(X[:, 0]), rtol=1e-9)


@pytest.mark.parametrize("scale", [1.0, 5.0, 12.0])
def test_nested_exponentials_stay_faithful(scale):
    """Gompertz-shaped exp(-a*exp(-b*x)) is a real target; it must evaluate exactly."""
    X = np.linspace(0.0, scale, 40).reshape(-1, 1)
    tree = _tree("exp", _tree("*", C(-3.0), _tree("exp", _tree("*", C(-0.45), X0))))
    got = _raw_eval(tree, X, BASE_FUNCTIONS)
    np.testing.assert_allclose(got, np.exp(-3.0 * np.exp(-0.45 * X[:, 0])), rtol=1e-9)


def test_log_still_guards_its_domain():
    """log(x) for x <= 0 must not produce nan; the domain guard stays."""
    X = np.array([[-5.0], [0.0], [1.0]])
    got = _raw_eval(_tree("log", X0), X, BASE_FUNCTIONS)
    assert got is None or np.all(np.isfinite(got))
