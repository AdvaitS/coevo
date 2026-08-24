"""Evolved symbolic fitness predictor (genetic programming).

Implements the core idea of Schmidt & Lipson (2008): rather than fixing a
surrogate model *a priori*, *evolve* a compact symbolic expression that predicts
fitness, and re-evolve it each generation so it specialises to the current
population. The predictor is therefore small, fast to evaluate, and — unlike a
black-box GP/RF — human-readable.

The operator library is pluggable: pass ``functions`` to :class:`SymbolicRegressor`
to add domain-specific primitives (e.g. biological Hill / Michaelis–Menten terms).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Callable

import numpy as np

from coevo.surrogates.base import Surrogate


def _infix(sym: str) -> Callable[[list[str]], str]:
    return lambda c: f"({c[0]} {sym} {c[1]})"


# name -> (numpy function, arity, string renderer). Functions are vectorised
# over the batch axis and must be numerically stable on arbitrary inputs.
_DEFAULT_FUNCTIONS: dict[str, tuple[Callable, int, Callable[[list[str]], str]]] = {
    "+": (lambda a, b: a + b, 2, _infix("+")),
    "-": (lambda a, b: a - b, 2, _infix("-")),
    "*": (lambda a, b: a * b, 2, _infix("*")),
    "/": (lambda a, b: a / np.where(np.abs(b) < 1e-9, 1e-9, b), 2, _infix("/")),
    "neg": (lambda a: -a, 1, lambda c: f"(-{c[0]})"),
    "sin": (np.sin, 1, lambda c: f"sin({c[0]})"),
    "cos": (np.cos, 1, lambda c: f"cos({c[0]})"),
    "exp": (lambda a: np.exp(np.clip(a, -50.0, 50.0)), 1, lambda c: f"exp({c[0]})"),
    "log": (lambda a: np.log(np.clip(a, 1e-9, None)), 1, lambda c: f"log({c[0]})"),
    "sq": (lambda a: a * a, 1, lambda c: f"({c[0]})**2"),
}


@dataclass
class Node:
    """A node in a symbolic expression tree."""

    op: str
    args: tuple = ()
    idx: int = 0  # feature index, for op == "x"
    val: float = 0.0  # constant value, for op == "c"


def _as_2d(X: np.ndarray) -> np.ndarray:
    """Coerce ``X`` to ``(n_samples, n_features)``.

    ``np.atleast_2d`` turns a 1-D array of ``n`` samples into ``(1, n)`` -- one
    sample with ``n`` features -- which silently produces a model over imaginary
    features instead of raising. A 1-D input is always a single feature here.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 0:
        return X.reshape(1, 1)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim > 2:
        raise ValueError(f"X must be 1- or 2-dimensional, got shape {X.shape}")
    return X


def _eval(node: Node, X: np.ndarray, functions: dict) -> np.ndarray:
    op = node.op
    if op == "x":
        return X[:, node.idx]
    if op == "c":
        return np.full(X.shape[0], node.val)
    fn = functions[op][0]
    args = tuple(_eval(a, X, functions) for a in node.args)
    with np.errstate(all="ignore"):
        return fn(*args)


def _size(node: Node) -> int:
    return 1 + sum(_size(a) for a in node.args)


def _collect(node: Node, acc: list[Node]) -> None:
    acc.append(node)
    for a in node.args:
        _collect(a, acc)


def _collect_constants(node: Node, acc: list[Node]) -> None:
    if node.op == "c":
        acc.append(node)
        return
    for a in node.args:
        _collect_constants(a, acc)


def _replace(node: Node, target: Node, replacement: Node) -> Node:
    if node is target:
        return replacement
    return Node(
        node.op,
        args=tuple(_replace(a, target, replacement) for a in node.args),
        idx=node.idx,
        val=node.val,
    )


def _deep_copy(node: Node) -> Node:
    """A structurally identical tree that shares no nodes with the original.

    Crossover splices the *same* subtree object into a child rather than a copy,
    so nodes are shared across the population. Optimising an individual's
    constants in place would therefore silently rewrite every other individual
    that happens to share that subtree.
    """
    return Node(node.op, tuple(_deep_copy(a) for a in node.args), node.idx, node.val)


def _random_node(
    rng: np.random.Generator,
    dim: int,
    depth: int,
    max_depth: int,
    const_range: tuple[float, float],
    functions: dict,
) -> Node:
    is_terminal = depth >= max_depth or rng.random() < 0.3
    if is_terminal:
        if rng.random() < 0.7:
            return Node("x", idx=int(rng.integers(0, dim)))
        return Node("c", val=float(rng.uniform(*const_range)))
    names = list(functions.keys())
    op = names[int(rng.integers(0, len(names)))]
    arity = functions[op][1]
    args = tuple(_random_node(rng, dim, depth + 1, max_depth, const_range, functions) for _ in range(arity))
    return Node(op, args=args)


#: Node cost of the ``a + b * f(x)`` wrapper that linear scaling adds, so a
#: scaled expression is not reported as cheaper than it really is.
_AFFINE_NODES = 4


def _raw_eval(tree: Node, X: np.ndarray, functions: dict) -> np.ndarray | None:
    """Evaluate ``tree`` and clip; return ``None`` if the result is unusable."""
    try:
        with np.errstate(all="ignore"):
            pred = _eval(tree, X, functions)
    except (OverflowError, FloatingPointError, ValueError, ZeroDivisionError):
        return None
    pred = np.clip(np.asarray(pred, dtype=float), -1e12, 1e12)
    if not np.all(np.isfinite(pred)):
        return None
    return pred


def linear_scale(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope and intercept mapping ``pred`` onto ``y``.

    Keijzer's linear scaling (EuroGP 2003): score a candidate by how well
    ``a + b * f(x)`` fits the target rather than ``f(x)`` itself, so the search
    only has to discover the *shape* of the relationship and gets its scale and
    offset for free. Returns ``(a, b)``; a constant prediction scales to the
    mean of ``y``.
    """
    var = float(np.var(pred))
    if not np.isfinite(var) or var < 1e-18:
        return float(np.mean(y)), 0.0
    b = float(np.cov(pred, y, bias=True)[0, 1] / var)
    a = float(np.mean(y) - b * np.mean(pred))
    if not (np.isfinite(a) and np.isfinite(b)):
        return float(np.mean(y)), 0.0
    return a, b


def _fitness(
    tree: Node,
    X: np.ndarray,
    y: np.ndarray,
    parsimony: float,
    functions: dict,
    linear_scaling: bool = False,
) -> tuple[float, float]:
    """Return ``(penalised fitness, raw RMSE)`` for ``tree``."""
    pred = _raw_eval(tree, X, functions)
    if pred is None:
        return inf, inf
    size = _size(tree)
    if linear_scaling:
        a, b = linear_scale(pred, y)
        pred = a + b * pred
        size += _AFFINE_NODES
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    if not np.isfinite(rmse):
        return inf, inf
    return rmse + parsimony * size, rmse


def _to_str(node: Node, functions: dict) -> str:
    if node.op == "x":
        return f"x{node.idx}"
    if node.op == "c":
        return f"{node.val:.4g}"
    render = functions[node.op][2]
    return render([_to_str(a, functions) for a in node.args])


def _fold(node: Node, functions: dict) -> float:
    """Evaluate a constant-only subtree to a scalar."""
    if node.op == "c":
        return node.val
    fn = functions[node.op][0]
    vals = [_fold(a, functions) for a in node.args]
    out = fn(*[np.array([v], dtype=float) for v in vals])
    return float(np.asarray(out).ravel()[0])


def _same(a: Node, b: Node) -> bool:
    """Structural equality of two expression trees."""
    if a.op != b.op:
        return False
    if a.op == "x":
        return a.idx == b.idx
    if a.op == "c":
        return abs(a.val - b.val) < 1e-12
    return len(a.args) == len(b.args) and all(_same(x, y) for x, y in zip(a.args, b.args))


def _simplify(node: Node, functions: dict) -> Node:
    """Fold constants and apply algebraic identities to reduce bloat."""
    if node.op in ("x", "c"):
        return node
    args = tuple(_simplify(a, functions) for a in node.args)
    op = node.op
    arity = len(args)

    # constant folding
    if all(a.op == "c" for a in args):
        try:
            with np.errstate(all="ignore"):
                return Node("c", val=_fold(Node(op, args=args), functions))
        except (OverflowError, FloatingPointError, ValueError, ZeroDivisionError):
            pass

    if arity == 1:
        a = args[0]
        if op == "neg" and a.op == "neg":
            return a.args[0]
        # NOTE: exp(log(x)) -> x and log(exp(x)) -> x are *not* valid rewrites
        # here. ``log`` clips its argument to 1e-9 and ``exp`` clips to +/-50, so
        # the protected pair is not the identity outside those domains --
        # exp(log(-3)) evaluates to 1e-9, not -3. Applying the rewrite silently
        # changes the model's predictions. Do not reintroduce them without an
        # interval-arithmetic domain check (Keijzer, EuroGP 2003).
        return Node(op, args=args)

    if arity == 2:
        a, b = args
        if op == "+":
            if a.op == "c" and a.val == 0.0:
                return b
            if b.op == "c" and b.val == 0.0:
                return a
        elif op == "-":
            if b.op == "c" and b.val == 0.0:
                return a
            if _same(a, b):
                return Node("c", val=0.0)
        elif op == "*":
            if (a.op == "c" and a.val == 0.0) or (b.op == "c" and b.val == 0.0):
                return Node("c", val=0.0)
            if a.op == "c" and a.val == 1.0:
                return b
            if b.op == "c" and b.val == 1.0:
                return a
        elif op == "/":
            if b.op == "c" and b.val == 1.0:
                return a
            if a.op == "c" and a.val == 0.0:
                return Node("c", val=0.0)
            if _same(a, b):
                return Node("c", val=1.0)
        return Node(op, args=args)

    return Node(op, args=args)


def _simplify_fixpoint(node: Node, functions: dict, max_iter: int = 6) -> Node:
    """Apply ``_simplify`` repeatedly until the tree stops shrinking."""
    for _ in range(max_iter):
        simplified = _simplify(node, functions)
        if _size(simplified) >= _size(node):
            return node
        node = simplified
    return node


@dataclass
class ParetoModel:
    """One model on the accuracy-vs-complexity frontier, ready to apply.

    Carries the expression tree, not just its rendering, so the model can be
    scored on held-out data — which is the only way to tell a discovered law
    from a memorised one.
    """

    expression: str
    rmse: float
    complexity: int
    tree: Node
    intercept: float
    scale: float
    functions: dict

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply the model to ``X``, including its linear-scaling terms."""
        X = _as_2d(X)
        pred = _raw_eval(self.tree, X, self.functions)
        if pred is None:
            return np.full(len(X), self.intercept)
        return self.intercept + self.scale * pred

    def __str__(self) -> str:
        return f"{self.expression}  [rmse={self.rmse:.4g}, complexity={self.complexity}]"


def optimize_tree_constants(
    tree: Node,
    X: np.ndarray,
    y: np.ndarray,
    functions: dict,
    linear_scaling: bool = True,
    max_nfev: int = 60,
) -> Node:
    """Return a copy of ``tree`` with its constants refit by least squares.

    Genetic programming discovers structure quickly but carries whatever
    ephemeral constants a subtree happened to inherit. Judging structures on
    those constants means a correct structure with poor constants loses to a
    wrong structure with lucky ones -- and is discarded before anything can
    refine it. Running this during the search, rather than only on the winner,
    is what lets the right shape survive long enough to be found.
    """
    tree = _deep_copy(tree)
    constants: list[Node] = []
    _collect_constants(tree, constants)
    if not constants or len(constants) > len(y):
        return tree

    init = np.array([c.val for c in constants], dtype=float)

    def _eval_params(node: Node, p: np.ndarray, counter: list[int]) -> np.ndarray:
        if node.op == "x":
            return X[:, node.idx]
        if node.op == "c":
            v = p[counter[0]]
            counter[0] += 1
            return np.full(X.shape[0], v)
        return functions[node.op][0](*(_eval_params(a, p, counter) for a in node.args))

    def residual(p: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            pred = np.nan_to_num(
                np.clip(_eval_params(tree, p, [0]), -1e6, 1e6), nan=1e6, posinf=1e6, neginf=-1e6
            )
            if linear_scaling:
                a, b = linear_scale(pred, y)
                pred = a + b * pred
        return pred - y

    from scipy.optimize import least_squares

    try:
        with np.errstate(all="ignore"):
            before = float(np.sqrt(np.mean(residual(init) ** 2)))
            result = least_squares(residual, init, method="lm", max_nfev=max_nfev)
            after = float(np.sqrt(np.mean(residual(result.x) ** 2)))
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return tree
    if np.isfinite(after) and after < before:
        for c, v in zip(constants, result.x):
            c.val = float(v)
    return tree


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
        functions: dict[str, tuple[Callable, int, Callable[[list[str]], str]]] | None = None,
        const_range: tuple[float, float] = (-1.0, 1.0),
        refine: bool = True,
        linear_scaling: bool = True,
        optimize_every: int = 0,
        optimize_top_k: int = 3,
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
        self.const_range = const_range
        self.refine = refine
        self.linear_scaling = linear_scaling
        self.optimize_every = optimize_every
        self.optimize_top_k = optimize_top_k
        self._functions = dict(_DEFAULT_FUNCTIONS)
        if functions:
            for name, spec in functions.items():
                if spec is None:  # allow removing a base operator
                    self._functions.pop(name, None)
                else:
                    self._functions[name] = spec
        self.best_: Node | None = None
        self.best_rmse_: float = inf
        self.intercept_: float = 0.0
        self.scale_: float = 1.0
        #: complexity -> (rmse, tree, intercept, scale) for the best expression
        #: seen at each size. Populated during :meth:`fit`.
        self.hall_of_fame_: dict[int, tuple[float, Node, float, float]] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SymbolicRegressor":
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} samples vs {len(y)} targets")
        rng = np.random.default_rng(self.seed)
        dim = X.shape[1]

        self.hall_of_fame_ = {}

        def _score(trees: list[Node]) -> np.ndarray:
            """Penalised fitness for a population, recording each size's best."""
            out = np.empty(len(trees))
            for i, tree in enumerate(trees):
                out[i], rmse = _fitness(
                    tree, X, y, self.parsimony, self._functions, self.linear_scaling
                )
                if not np.isfinite(rmse):
                    continue
                complexity = _size(tree) + (_AFFINE_NODES if self.linear_scaling else 0)
                known = self.hall_of_fame_.get(complexity)
                if known is None or rmse < known[0]:
                    a, b = self._scaling_for(tree, X, y)
                    self.hall_of_fame_[complexity] = (float(rmse), tree, a, b)
            return out

        pop = [
            _random_node(rng, dim, 0, self.max_depth, self.const_range, self._functions)
            for _ in range(self.population_size)
        ]
        fits = _score(pop)

        best, best_fit = pop[int(np.argmin(fits))], float(np.min(fits))

        for generation in range(self.generations):
            # Periodically refit the constants of the best few individuals, so a
            # promising structure is judged on constants that fit it rather than
            # on whatever it inherited. Replaces the individuals in place (with
            # copies -- nodes are shared across the population).
            if self.optimize_every > 0 and (generation + 1) % self.optimize_every == 0:
                for idx in np.argsort(fits)[: max(1, self.optimize_top_k)]:
                    idx = int(idx)
                    tuned = optimize_tree_constants(
                        pop[idx], X, y, self._functions, self.linear_scaling
                    )
                    penalised, rmse = _fitness(
                        tuned, X, y, self.parsimony, self._functions, self.linear_scaling
                    )
                    if penalised < fits[idx]:
                        pop[idx], fits[idx] = tuned, penalised
                        if penalised < best_fit:
                            best, best_fit = tuned, float(penalised)
                        if np.isfinite(rmse):
                            complexity = _size(tuned) + (
                                _AFFINE_NODES if self.linear_scaling else 0
                            )
                            known = self.hall_of_fame_.get(complexity)
                            if known is None or rmse < known[0]:
                                a, b = self._scaling_for(tuned, X, y)
                                self.hall_of_fame_[complexity] = (float(rmse), tuned, a, b)

            new_pop = [best]  # elitism
            while len(new_pop) < self.population_size:
                p1 = pop[int(_tournament(rng, fits, self.tournament_size))]
                p2 = pop[int(_tournament(rng, fits, self.tournament_size))]
                if rng.random() < self.crossover_p:
                    c1, c2 = _crossover(rng, p1, p2, max_size=self.max_size)
                else:
                    c1, c2 = p1, p2
                c1 = _mutate(rng, c1, dim, self.max_depth, self.max_size, self._functions, self.const_range) if rng.random() < self.mutation_p else c1
                c2 = _mutate(rng, c2, dim, self.max_depth, self.max_size, self._functions, self.const_range) if rng.random() < self.mutation_p else c2
                new_pop.extend((c1, c2))
            pop = new_pop[: self.population_size]
            fits = _score(pop)
            gi = int(np.argmin(fits))
            if fits[gi] < best_fit:
                best, best_fit = pop[gi], float(fits[gi])

        def _rmse_of(tree: Node) -> float:
            return _fitness(tree, X, y, 0.0, self._functions, self.linear_scaling)[1]

        # Simplification rewrites the tree using algebraic identities that the
        # protected operators do not always satisfy, so accept it only when it
        # does not make the model worse.
        raw_rmse = _rmse_of(best)
        simplified = _simplify_fixpoint(best, self._functions)
        simplified_rmse = _rmse_of(simplified)
        if simplified_rmse <= raw_rmse + 1e-12:
            self.best_, self.best_rmse_ = simplified, simplified_rmse
        else:
            self.best_, self.best_rmse_ = best, raw_rmse
        if self.refine:
            self.refine_constants(X, y)
        self.intercept_, self.scale_ = self._scaling_for(self.best_, X, y)
        self.best_rmse_ = _rmse_of(self.best_)
        return self

    def _scaling_for(self, tree: Node, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """The ``(intercept, scale)`` pair this model applies to ``tree``."""
        if not self.linear_scaling:
            return 0.0, 1.0
        pred = _raw_eval(tree, X, self._functions)
        if pred is None:
            return 0.0, 1.0
        return linear_scale(pred, y)

    def pareto_front(self) -> list["ParetoModel"]:
        """Non-dominated models from this run, ordered by increasing complexity.

        Genetic programming visits far more of the accuracy/complexity trade-off
        than a single winner records. The hall of fame keeps the best expression
        seen at every size, so one run yields the whole front rather than one
        point — and there is no need to sweep the parsimony coefficient to
        approximate it.

        Each entry is a :class:`ParetoModel`, which carries the expression tree
        and so can be applied to new data. Returning only the rendered string
        would make held-out evaluation impossible without re-parsing it.
        """
        best_so_far = inf
        front: list[ParetoModel] = []
        for complexity in sorted(self.hall_of_fame_):
            rmse, tree, a, b = self.hall_of_fame_[complexity]
            if rmse < best_so_far:
                best_so_far = rmse
                front.append(
                    ParetoModel(
                        expression=self._render(tree, a, b),
                        rmse=float(rmse),
                        complexity=int(complexity),
                        tree=tree,
                        intercept=a,
                        scale=b,
                        functions=self._functions,
                    )
                )
        return front

    def refine_constants(self, X: np.ndarray, y: np.ndarray) -> "SymbolicRegressor":
        """Refit the evolved expression's constants by nonlinear least squares.

        Genetic programming finds the *structure* of a model quickly but its
        ephemeral constants are imprecise; a least-squares pass over just the
        constants recovers accurate parameters.
        """
        if self.best_ is None:
            return self
        constants: list[Node] = []
        _collect_constants(self.best_, constants)
        if not constants:
            return self

        X = _as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        functions = self._functions
        init = np.array([c.val for c in constants], dtype=float)
        # Levenberg-Marquardt requires at least as many residuals as parameters.
        if len(init) > len(y):
            return self

        def _eval_params(node: Node, p: np.ndarray, counter: list[int]) -> np.ndarray:
            if node.op == "x":
                return X[:, node.idx]
            if node.op == "c":
                v = p[counter[0]]
                counter[0] += 1
                return np.full(X.shape[0], v)
            fn = functions[node.op][0]
            return fn(*(_eval_params(a, p, counter) for a in node.args))

        def residual(p: np.ndarray) -> np.ndarray:
            pred = np.clip(_eval_params(self.best_, p, [0]), -1e6, 1e6)
            pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
            if self.linear_scaling:
                # Refine against the same objective the search optimised: the
                # error *after* scaling, not before. Otherwise the solver fights
                # to fix an offset that linear scaling supplies for free.
                a, b = linear_scale(pred, y)
                pred = a + b * pred
            return pred - y

        from scipy.optimize import least_squares

        rng = np.random.default_rng(0)
        with np.errstate(all="ignore"):
            best_params, best_err = init, float(np.sqrt(np.mean(residual(init) ** 2)))
            for restart in range(3):
                init_try = init if restart == 0 else init + rng.normal(0.0, 0.5, size=init.shape)
                try:
                    result = least_squares(residual, init_try, method="lm", max_nfev=300)
                except (ValueError, TypeError, np.linalg.LinAlgError):
                    continue
                err = float(np.sqrt(np.mean(residual(result.x) ** 2)))
                if err < best_err:
                    best_err, best_params = err, result.x
        if not np.isfinite(best_err):
            return self

        for c, v in zip(constants, best_params):
            c.val = float(v)
        self.best_rmse_ = best_err
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        if self.best_ is None:
            raise RuntimeError("SymbolicRegressor must be fitted before predict().")
        with np.errstate(all="ignore"):
            pred = np.asarray(_eval(self.best_, X, self._functions), dtype=float).ravel()
        pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=1e12, neginf=-1e12), -1e12, 1e12)
        return self.intercept_ + self.scale_ * pred

    def _render(self, tree: Node, intercept: float, scale: float) -> str:
        """Render ``tree`` including the affine terms actually applied to it."""
        inner = _to_str(tree, self._functions)
        if scale == 0.0:
            return f"{intercept:.4g}"
        if abs(scale - 1.0) > 1e-9:
            inner = f"({scale:.4g} * {inner})"
        if abs(intercept) > 1e-9:
            inner = f"({intercept:.4g} + {inner})"
        return inner

    def expression(self) -> str:
        if self.best_ is None:
            raise RuntimeError("SymbolicRegressor must be fitted before expression().")
        return self._render(self.best_, self.intercept_, self.scale_)

    @property
    def complexity(self) -> int:
        """Node count of the expression, including any linear-scaling terms."""
        if self.best_ is None:
            return 0
        return _size(self.best_) + (_AFFINE_NODES if self.linear_scaling else 0)


def _tournament(rng: np.random.Generator, fits: np.ndarray, k: int) -> int:
    idx = rng.choice(len(fits), k, replace=False)
    return int(idx[np.argmin(fits[idx])])


def _crossover(
    rng: np.random.Generator, p1: Node, p2: Node, max_size: int = 0, tries: int = 6
) -> tuple[Node, Node]:
    """Subtree crossover, respecting ``max_size`` if it is positive.

    Without a size bound, subtree crossover is the dominant source of bloat: a
    small parent can adopt an arbitrarily large subtree, so ``max_size`` ends up
    enforced only on mutation and the tree grows without limit. Resample the cut
    points a few times and fall back to the parents if no child fits.
    """
    n1: list[Node] = []
    n2: list[Node] = []
    _collect(p1, n1)
    _collect(p2, n2)
    for _ in range(max(1, tries)):
        s1 = n1[int(rng.integers(len(n1)))]
        s2 = n2[int(rng.integers(len(n2)))]
        c1, c2 = _replace(p1, s1, s2), _replace(p2, s2, s1)
        if max_size <= 0 or (_size(c1) <= max_size and _size(c2) <= max_size):
            return c1, c2
    return p1, p2


def _mutate(
    rng: np.random.Generator,
    tree: Node,
    dim: int,
    max_depth: int,
    max_size: int,
    functions: dict,
    const_range: tuple[float, float],
) -> Node:
    nodes: list[Node] = []
    _collect(tree, nodes)
    sub = nodes[int(rng.integers(len(nodes)))]
    new_tree = _replace(tree, sub, _random_node(rng, dim, 0, max_depth, const_range, functions))
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
