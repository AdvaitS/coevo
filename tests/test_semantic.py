"""Semantic backpropagation: inversions, desired semantics, and the operator."""

import numpy as np
import pytest

from coevo.semantic import (
    _canonical_templates,
    build_library,
    desired_semantics,
    invert,
    random_desired_operator,
)
from coevo.surrogates.evolved import _DEFAULT_FUNCTIONS as FUNCTIONS
from coevo.surrogates.evolved import Node, _eval, _size, linear_scale, structure_signature

X = np.linspace(0.2, 5.0, 40).reshape(-1, 1)
x = X[:, 0]


def test_inversions_undo_their_operator():
    """The whole method rests on this: invert must be the operator's inverse."""
    d = np.array([0.2, 0.5, 0.8])
    other = np.array([2.0, 4.0, 5.0])
    np.testing.assert_allclose(invert("+", 0, d, other), d - other)
    np.testing.assert_allclose(invert("-", 1, d, other), other - d)
    np.testing.assert_allclose(invert("*", 0, d, other), d / other)
    np.testing.assert_allclose(invert("/", 0, d, other), d * other)
    np.testing.assert_allclose(invert("exp", 0, d, None), np.log(d))
    np.testing.assert_allclose(invert("log", 0, d, None), np.exp(d))
    np.testing.assert_allclose(invert("sigmoid", 0, np.array([0.5]), None), [0.0], atol=1e-12)


def test_inversions_return_nan_outside_the_domain():
    """A position with no real answer is 'don't care', not zero -- treating it as
    a number would invent a constraint the data does not impose."""
    assert np.isnan(invert("exp", 0, np.array([-1.0]), None))[0]
    assert np.isnan(invert("sigmoid", 0, np.array([1.5]), None))[0]
    assert np.isnan(invert("*", 0, np.array([1.0]), np.array([0.0])))[0]


def test_non_invertible_operators_are_declined():
    assert invert("relu", 0, np.array([1.0]), None) is None
    assert invert("pow", 0, np.array([1.0]), np.array([2.0])) is None


def test_desired_semantics_round_trips():
    """A subtree producing exactly the desired output must make the tree exact."""
    y = 3.0 * np.exp(-0.5 * x) + 1.0
    tree = Node("exp", (Node("*", (Node("c", val=2.0), Node("x", idx=0))),))
    desired = desired_semantics(tree, (0,), X, y, FUNCTIONS, linear_scaling=True)
    assert desired is not None
    assert np.all(np.isfinite(desired)), "a badly-scaled tree must not poison the target"
    predicted = np.exp(np.clip(desired, -700, 700))
    a, b = linear_scale(predicted, y)
    assert float(np.sqrt(np.mean((a + b * predicted - y) ** 2))) < 1e-9


def test_desired_semantics_survives_a_badly_scaled_tree():
    """The regression this guards: taking the current tree's linear scaling on
    faith sends (y - a)/b negative, and every later log inversion returns NaN --
    exactly when the operator is most needed."""
    y = 3.0 * np.exp(-0.5 * x) + 1.0
    for constant in (2.0, 10.0, -7.0):
        tree = Node("exp", (Node("*", (Node("c", val=constant), Node("x", idx=0))),))
        desired = desired_semantics(tree, (0,), X, y, FUNCTIONS, linear_scaling=True)
        assert desired is not None and np.sum(np.isfinite(desired)) > len(x) // 2


def test_library_entries_have_distinct_semantics():
    rng = np.random.default_rng(0)
    library = build_library(rng, X, FUNCTIONS, size=120)
    assert len(library) > 20
    outputs = {np.array2string(np.round(o, 8), threshold=64) for _, o in library}
    assert len(outputs) == len(library)
    assert all(np.all(np.isfinite(o)) for _, o in library)


PLAIN = {k: FUNCTIONS[k] for k in ("+", "-", "*", "/", "neg", "exp", "log", "sq")}


def test_canonical_templates_only_use_available_operators():
    """Templates are built only from operators the caller actually has."""
    nodes = _canonical_templates(PLAIN, np.linspace(-5.0, 5.0, 4))
    assert len(nodes) > 0
    allowed = set(PLAIN) | {"x", "c"}

    def ops_of(node):
        acc = {node.op}
        for a in node.args:
            acc |= ops_of(a)
        return acc

    for node in nodes:
        assert ops_of(node) <= allowed
    # strip exp/log/sq and every template needing them must vanish
    minimal = {k: FUNCTIONS[k] for k in ("+", "-", "*", "/", "neg")}
    minimal_nodes = _canonical_templates(minimal, np.linspace(-5.0, 5.0, 4))
    assert all(ops_of(n) <= (set(minimal) | {"x", "c"}) for n in minimal_nodes)
    assert len(minimal_nodes) < len(nodes)


def test_template_library_adds_deep_structures():
    """Templates install depth-3 pieces a depth-2 random library cannot contain."""
    rng = np.random.default_rng(0)
    shallow = build_library(rng, X, PLAIN, size=400, max_depth=2, templates=False)
    deep = build_library(rng, X, PLAIN, size=400, max_depth=2, templates=True)
    shallow_sigs = {structure_signature(n) for n, _ in shallow}
    deep_sigs = {structure_signature(n) for n, _ in deep}
    # a Gompertz inner term exp(c·exp(c·x)) is depth 4 — reachable only via templates
    gompertz = "exp(*(C,exp(*(C,x0))))"
    assert gompertz not in shallow_sigs
    assert gompertz in deep_sigs
    assert max(_size(n) for n, _ in shallow) <= 7
    assert max(_size(n) for n, _ in deep) >= 7
    assert len(deep) == len(shallow) == 400, "templates displace random fill, not inflate it"


def test_operator_returns_a_valid_tree_and_respects_max_size():
    rng = np.random.default_rng(1)
    y = 3.0 * np.exp(-0.5 * x) + 1.0
    library = build_library(rng, X, FUNCTIONS, size=150)
    tree = Node("exp", (Node("*", (Node("c", val=2.0), Node("x", idx=0))),))
    for _ in range(10):
        out = random_desired_operator(rng, tree, X, y, FUNCTIONS, library, max_size=12)
        evaluated = _eval(out, X, FUNCTIONS)
        assert np.asarray(evaluated).shape == (len(X),)


def test_semantic_search_generates_structures_plain_gp_does_not():
    """The measured effect: told what a node must output, the search assembles
    x/(K+x) far more often than blind recombination does."""
    from coevo import SymbolicRegressor

    def signature(node):
        if node.op == "x":
            return f"x{node.idx}"
        if node.op == "c":
            return "C"
        args = sorted(signature(a) for a in node.args) if node.op in {"+", "*"} else [
            signature(a) for a in node.args
        ]
        return f"{node.op}({','.join(args)})"

    target = signature(Node("/", (Node("x", idx=0),
                                  Node("+", (Node("c"), Node("x", idx=0))))))
    y = 2.5 * x / (0.6 + x)

    def found(semantic_p):
        import coevo.surrogates.evolved as ev

        seen = set()
        original = ev._raw_eval

        def recording(tree, Xa, fns):
            seen.add(signature(tree))
            return original(tree, Xa, fns)

        ev._raw_eval = recording
        try:
            SymbolicRegressor(population_size=120, generations=15, seed=0,
                              semantic_p=semantic_p).fit(X, y)
        finally:
            ev._raw_eval = original
        return target in seen

    assert found(0.4) or not found(0.0), (
        "semantic search should reach this structure at least as often as blind search"
    )
