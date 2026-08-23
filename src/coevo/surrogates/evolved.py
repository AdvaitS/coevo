"""Evolved symbolic fitness predictor (genetic programming).

Implements the core idea of Schmidt & Lipson (2008): rather than fixing a
surrogate model *a priori*, *evolve* a compact symbolic expression that predicts
fitness, and re-evolve it each generation so it specialises to the current
population. The predictor is therefore small, fast to evaluate, and — unlike a
black-box GP/RF — human-readable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

import numpy as np

from coevo.surrogates.base import Surrogate

_BINARY = {"+", "-", "*", "/"}
_UNARY = {"neg", "sin", "cos", "exp", "log", "sq"}
_OPS = tuple(sorted(_BINARY | _UNARY))


@dataclass
class Node:
    """A node in a symbolic expression tree."""

    op: str
    args: tuple = ()
    idx: int = 0  # feature index, for op == "x"
    val: float = 0.0  # constant value, for op == "c"


def _eval(node: Node, X: np.ndarray) -> np.ndarray:
    op = node.op
    if op == "x":
        return X[:, node.idx]
    if op == "c":
        return np.full(X.shape[0], node.val)
    if op == "neg":
        return -_eval(node.args[0], X)
    if op == "sin":
        return np.sin(_eval(node.args[0], X))
    if op == "cos":
        return np.cos(_eval(node.args[0], X))
    if op == "exp":
        return np.exp(np.clip(_eval(node.args[0], X), -50.0, 50.0))
    if op == "log":
        return np.log(np.clip(_eval(node.args[0], X), 1e-9, None))
    if op == "sq":
        a = _eval(node.args[0], X)
        return a * a
    a = _eval(node.args[0], X)
    b = _eval(node.args[1], X)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    # '/'
    return a / np.where(np.abs(b) < 1e-9, 1e-9, b)


def _size(node: Node) -> int:
    return 1 + sum(_size(a) for a in node.args)


def _collect(node: Node, acc: list[Node]) -> None:
    acc.append(node)
    for a in node.args:
        _collect(a, acc)


def _replace(node: Node, target: Node, replacement: Node) -> Node:
    if node is target:
        return replacement
    return Node(
        node.op,
        args=tuple(_replace(a, target, replacement) for a in node.args),
        idx=node.idx,
        val=node.val,
    )


def _random_node(
    rng: np.random.Generator, dim: int, depth: int, max_depth: int, const_range: tuple[float, float]
) -> Node:
    is_terminal = depth >= max_depth or rng.random() < 0.3
    if is_terminal:
        if rng.random() < 0.7:
            return Node("x", idx=int(rng.integers(0, dim)))
        return Node("c", val=float(rng.uniform(*const_range)))
    op = _OPS[int(rng.integers(0, len(_OPS)))]
    if op in _UNARY:
        return Node(op, args=(_random_node(rng, dim, depth + 1, max_depth, const_range),))
    return Node(
        op,
        args=(
            _random_node(rng, dim, depth + 1, max_depth, const_range),
            _random_node(rng, dim, depth + 1, max_depth, const_range),
        ),
    )


def _fitness(tree: Node, X: np.ndarray, y: np.ndarray, parsimony: float) -> float:
    try:
        pred = _eval(tree, X)
    except (OverflowError, FloatingPointError):
        return inf
    pred = np.clip(pred, -1e12, 1e12)
    if not np.all(np.isfinite(pred)):
        return inf
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return rmse + parsimony * _size(tree)


def _to_str(node: Node) -> str:
    if node.op == "x":
        return f"x{node.idx}"
    if node.op == "c":
        return f"{node.val:.4g}"
    if node.op == "neg":
        return f"(-{_to_str(node.args[0])})"
    if node.op == "sq":
        return f"({_to_str(node.args[0])})**2"
    if node.op in _UNARY:
        return f"{node.op}({_to_str(node.args[0])})"
    return f"({_to_str(node.args[0])} {node.op} {_to_str(node.args[1])})"


class SymbolicRegressor:
    """Evolves a symbolic expression ``x -> fitness`` via genetic programming."""

    def __init__(
        self,
        population_size: int = 300,
        generations: int = 40,
        max_depth: int = 5,
        max_size: int = 60,
        tournament_size: int = 3,
        crossover_p: float = 0.9,
        mutation_p: float = 0.2,
        parsimony: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.max_size = max_size
        self.tournament_size = tournament_size
        self.crossover_p = crossover_p
        self.mutation_p = mutation_p
        self.parsimony = parsimony
        self.seed = seed
        self.best_: Node | None = None
        self.best_rmse_: float = inf

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SymbolicRegressor":
        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        rng = np.random.default_rng(self.seed)
        dim = X.shape[1]

        pop = [_random_node(rng, dim, 0, self.max_depth, (-1.0, 1.0)) for _ in range(self.population_size)]
        fits = np.array([_fitness(t, X, y, self.parsimony) for t in pop])

        best, best_fit = pop[int(np.argmin(fits))], float(np.min(fits))

        for _ in range(self.generations):
            new_pop = [best]  # elitism
            while len(new_pop) < self.population_size:
                p1 = pop[int(_tournament(rng, fits, self.tournament_size))]
                p2 = pop[int(_tournament(rng, fits, self.tournament_size))]
                if rng.random() < self.crossover_p:
                    c1, c2 = _crossover(rng, p1, p2)
                else:
                    c1, c2 = p1, p2
                c1 = _mutate(rng, c1, dim, self.max_depth, self.max_size) if rng.random() < self.mutation_p else c1
                c2 = _mutate(rng, c2, dim, self.max_depth, self.max_size) if rng.random() < self.mutation_p else c2
                new_pop.extend((c1, c2))
            pop = new_pop[: self.population_size]
            fits = np.array([_fitness(t, X, y, self.parsimony) for t in pop])
            gi = int(np.argmin(fits))
            if fits[gi] < best_fit:
                best, best_fit = pop[gi], float(fits[gi])

        self.best_ = best
        self.best_rmse_ = float(np.sqrt(max(best_fit - self.parsimony * _size(best), 0.0)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        if self.best_ is None:
            raise RuntimeError("SymbolicRegressor must be fitted before predict().")
        return np.asarray(_eval(self.best_, X), dtype=float).ravel()

    def expression(self) -> str:
        if self.best_ is None:
            raise RuntimeError("SymbolicRegressor must be fitted before expression().")
        return _to_str(self.best_)


def _tournament(rng: np.random.Generator, fits: np.ndarray, k: int) -> int:
    idx = rng.choice(len(fits), k, replace=False)
    return int(idx[np.argmin(fits[idx])])


def _crossover(rng: np.random.Generator, p1: Node, p2: Node) -> tuple[Node, Node]:
    n1: list[Node] = []
    n2: list[Node] = []
    _collect(p1, n1)
    _collect(p2, n2)
    s1 = n1[int(rng.integers(len(n1)))]
    s2 = n2[int(rng.integers(len(n2)))]
    return _replace(p1, s1, s2), _replace(p2, s2, s1)


def _mutate(rng: np.random.Generator, tree: Node, dim: int, max_depth: int, max_size: int) -> Node:
    nodes: list[Node] = []
    _collect(tree, nodes)
    sub = nodes[int(rng.integers(len(nodes)))]
    new_tree = _replace(tree, sub, _random_node(rng, dim, 0, max_depth, (-1.0, 1.0)))
    return new_tree if _size(new_tree) <= max_size else tree


class EvolvedPredictor(Surrogate):
    """A surrogate whose model is itself evolved (genetic programming).

    Re-fitting (re-evolving) the expression each generation on the shifting
    archive makes the predictor track the population — the coevolutionary loop of
    Schmidt & Lipson (2008) — while remaining a compact, interpretable formula.
    """

    def __init__(self, regressor: SymbolicRegressor | None = None, **kwargs) -> None:
        self._regressor = regressor if regressor is not None else SymbolicRegressor(**kwargs)
        self.fitted = False
        self.error_trace: list[float] = []
        self.expression_trace: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EvolvedPredictor":
        self._regressor.fit(X, y)
        self.fitted = True
        self.error_trace.append(self._regressor.best_rmse_)
        self.expression_trace.append(self._regressor.expression())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._regressor.predict(X)

    @property
    def expression(self) -> str:
        return self._regressor.expression()
