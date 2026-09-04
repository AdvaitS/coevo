"""Semantic backpropagation: constructing the subtree the search needs.

Standard genetic programming builds structures by recombining whatever the
population happens to contain, and hopes the right shape assembles. Measured on
biosym's benchmark, it does not: the ground-truth structure for Gompertz growth
is six nodes deep by three, comfortably inside the depth limit, and is never
once generated across ~18,600 evaluations per run. Repairing the constant
optimizer does not help, because the structure never reaches it.

Semantic backpropagation (Pawlak, Wieloch & Krawiec, *Semantic Backpropagation
for Designing Search Operators in GP*) attacks that directly. Rather than
mutating blindly, it picks a node, works out what that node *would have to
output* for the whole tree to fit the target, and then goes looking for a
subtree that produces it:

1. choose a node, giving a path from the root;
2. propagate the target backwards along that path, inverting each operator, to
   get the node's **desired semantics** -- a vector of required outputs;
3. replace the node with the library subtree whose output is closest.

Steps 2 and 3 are what standard mutation lacks. A blind mutation has to be
lucky twice, in structure and in constants; this one is told what it is aiming
at, so a structure the search would otherwise never assemble can be installed
in a single step.

Inversions are partial. ``log`` can only invert an output that is positive,
``sigmoid`` only one strictly inside (0, 1), and division only where the
denominator does not vanish. Positions where the inversion has no real answer
are returned as NaN and treated as *don't care* when matching -- the honest
reading, since the desired value there is genuinely unconstrained rather than
zero.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from coevo.surrogates.evolved import (
    Node,
    _deep_copy,
    _eval,
    _MAGNITUDE_LIMIT,
    _size,
    linear_scale,
)

#: Below this magnitude a denominator is treated as vanishing and the inversion
#: as undefined, rather than producing an enormous "desired" value that no
#: subtree could match and that would dominate the distance.
_EPS = 1e-9


def invert(op: str, arg_index: int, desired: np.ndarray, other: np.ndarray | None):
    """What argument ``arg_index`` must output for this node to output ``desired``.

    Returns NaN wherever no real value would do, and ``None`` when the operator
    cannot be inverted at all.
    """
    d = np.asarray(desired, dtype=float)
    with np.errstate(all="ignore"):
        if op == "+":
            return d - other
        if op == "-":
            return d + other if arg_index == 0 else other - d
        if op == "*":
            safe = np.where(np.abs(other) < _EPS, np.nan, other)
            return d / safe
        if op == "/":
            if arg_index == 0:
                return d * other
            safe = np.where(np.abs(d) < _EPS, np.nan, d)
            return other / safe
        if op == "neg":
            return -d
        if op == "exp":
            return np.where(d > 0, np.log(np.where(d > 0, d, 1.0)), np.nan)
        if op == "log":
            return np.exp(np.clip(d, -700.0, 700.0))
        if op == "expm1":
            return np.where(d > -1, np.log1p(np.where(d > -1, d, 0.0)), np.nan)
        if op == "log1p":
            return np.expm1(np.clip(d, -700.0, 700.0))
        if op == "sigmoid":
            inside = (d > 0.0) & (d < 1.0)
            safe = np.where(inside, d, 0.5)
            return np.where(inside, np.log(safe / (1.0 - safe)), np.nan)
        if op == "sin":
            inside = np.abs(d) <= 1.0
            return np.where(inside, np.arcsin(np.where(inside, d, 0.0)), np.nan)
        if op == "cos":
            inside = np.abs(d) <= 1.0
            return np.where(inside, np.arccos(np.where(inside, d, 0.0)), np.nan)
    # relu and pow are not invertible in any useful single-valued way.
    return None


def _paths(node: Node, prefix: tuple = ()) -> list[tuple]:
    """Every node's path from the root, as a tuple of argument indices."""
    out = [prefix]
    for i, arg in enumerate(node.args):
        out.extend(_paths(arg, prefix + (i,)))
    return out


def _at(node: Node, path: tuple) -> Node:
    for i in path:
        node = node.args[i]
    return node


def desired_semantics(
    tree: Node, path: tuple, X: np.ndarray, y: np.ndarray, functions: dict,
    linear_scaling: bool = True,
) -> np.ndarray | None:
    """The output the node at ``path`` must produce for ``tree`` to fit ``y``.

    With linear scaling the model is ``a + b·f(x)``, so the root's desired output
    is ``(y − a) / b`` rather than ``y`` itself.
    """
    raw = np.asarray(y, dtype=float)
    targets = [raw]
    if linear_scaling:
        # The scaling that will apply *after* substitution is unknown, so the
        # only estimate available is the one the current tree induces -- and that
        # tree is by assumption a poor fit, which is why we are replacing part of
        # it. Using it blindly is worse than useless: for a badly-scaled tree
        # (y - a)/b goes negative and every subsequent log or sigmoid inversion
        # returns NaN, precisely when the operator is most needed. So try the
        # scaled target first and keep the unscaled one as a fallback, taking
        # whichever leaves more positions defined. (Virgolin et al., GECCO 2019,
        # study this interaction between linear scaling and backpropagation.)
        try:
            root_pred = np.asarray(_eval(tree, X, functions), dtype=float)
            a, b = linear_scale(root_pred, raw)
            if abs(b) >= _EPS:
                targets.insert(0, (raw - a) / b)
        except Exception:
            pass

    best = None
    for target in targets:
        candidate = _propagate(tree, path, target, X, functions)
        if candidate is None:
            continue
        defined = int(np.sum(np.isfinite(candidate)))
        if best is None or defined > best[0]:
            best = (defined, candidate)
        if defined == len(raw):
            break
    return None if best is None else best[1]


def _propagate(tree: Node, path: tuple, target: np.ndarray, X: np.ndarray, functions: dict):
    """Invert ``target`` down ``path``; None if some operator cannot be inverted."""
    node = tree
    for index in path:
        other = None
        if len(node.args) == 2:
            try:
                other = np.asarray(
                    _eval(node.args[1 - index], X, functions), dtype=float
                )
            except Exception:
                return None
        target = invert(node.op, index, target, other)
        if target is None:
            return None
        target = np.asarray(target, dtype=float)
        if not np.any(np.isfinite(target)):
            return None
        node = node.args[index]
    return target


def _canonical_templates(
    functions: dict, const_values: np.ndarray
) -> list[Node]:
    """Curated depth-2/3 building-block subtrees, one per constant assignment.

    The random library at ``max_depth=2`` supplies the depth-1 pieces that
    Michaelis-Menten needs (``K + x``) but not the depth-3 pieces that logistic
    and Gompertz need: a sigmoid is ``c / (c + exp(c·x))`` and a Gompertz inner
    term is ``exp(c · exp(c·x))``, each several nodes deep. Building the random
    library at depth 3 supplies those shapes but *dilutes* matching — measured,
    Michaelis-Menten generation falls from 12/15 to 7/15 as the library grows.

    These templates supply the missing shapes directly, as a small number of
    meaningful entries rather than a large number of random ones. Every template
    is instantiated over ``const_values``; constants are refit after install, so
    the grid only has to land a shape close enough to be selected, not nail the
    parameters. Only templates whose operators are all present in ``functions``
    are produced. Templates are single-feature (built on ``x0``); the random
    library already supplies every feature as a terminal, and the growth/response
    laws these target are univariate.

    Measured on biosym's benchmark (plain operators, gen=25, pop=200, 4 seeds),
    comparing semantic backpropagation with and without templates, by how often
    the ground-truth structure is *generated* during search:

    ==================  ===========  ===========  ===========
    law                 plain       semantic     +templates
    ==================  ===========  ===========  ===========
    michaelis_menten    0/4          1/4          4/4
    logistic_growth     0/4          0/4          3/4
    gompertz_growth     0/4          0/4          2/4
    ==================  ===========  ===========  ===========

    Generation is no longer the wall. It is still not recovery: the structure
    reaches the returned Pareto front in ~1/4 runs and is sympy-confirmed as the
    law in 0/4, the same "inner optimizer loses correct skeletons" gap recorded
    elsewhere. Templates are a necessary building block, not the whole answer.
    """
    import itertools

    def C(v: float) -> Node:
        return Node("c", val=float(v))

    def X() -> Node:
        return Node("x", idx=0)

    def has(*ops: str) -> bool:
        return all(op in functions for op in ops)

    # (arity, builder) pairs. Each builder maps its constants to a subtree.
    specs: list[tuple[int, Callable[..., Node]]] = []
    if has("/", "+"):
        specs.append((1, lambda a: Node("/", (X(), Node("+", (C(a), X()))))))  # x/(c+x)
        specs.append(
            (2, lambda a, b: Node("/", (X(), Node("+", (C(a), Node("*", (C(b), X())))))))
        )  # x/(c+cx)
    if has("sq", "/", "+"):
        specs.append(
            (
                1,
                lambda a: Node(
                    "/", (Node("sq", (X(),)), Node("+", (C(a), Node("sq", (X(),)))))
                ),
            )
        )  # sq(x)/(c+sq(x))
    if has("exp", "*"):
        specs.append((1, lambda a: Node("exp", (Node("*", (C(a), X())),))))  # exp(cx)
        specs.append(
            (2, lambda a, b: Node("exp", (Node("*", (C(a), Node("exp", (Node("*", (C(b), X())),)))),)))
        )  # exp(c·exp(cx))
    if has("exp", "+", "*"):
        specs.append(
            (2, lambda a, b: Node("exp", (Node("+", (C(a), Node("*", (C(b), X())))),)))
        )  # exp(c+cx)
    if has("exp", "/", "+", "*"):
        specs.append(
            (
                2,
                lambda a, b: Node(
                    "/",
                    (
                        Node("c", val=1.0),
                        Node(
                            "+",
                            (
                                Node("c", val=1.0),
                                Node(
                                    "exp",
                                    (Node("+", (C(a), Node("*", (C(b), X())))),),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        )  # 1/(1+exp(c+cx)) — logistic (the two 1s are the sigmoid's, not fitted)
    if has("exp", "*", "-"):
        specs.append(
            (
                3,
                lambda a, b, d: Node(
                    "-", (C(a), Node("*", (C(b), Node("exp", (Node("*", (C(d), X())),)))))
                ),
            )
        )  # c - c·exp(cx) — exponential association
    if has("log", "+", "*"):
        specs.append(
            (2, lambda a, b: Node("log", (Node("+", (C(a), Node("*", (C(b), X())))),)))
        )  # log(c+cx)

    out: list[Node] = []
    for arity, builder in specs:
        for combo in itertools.product(const_values.tolist(), repeat=arity):
            out.append(builder(*combo))
    return out


#: How many grid values each template constant takes. Coarse on purpose: three
#: constants at five values each would be 125 near-identical entries that dilute
#: matching, and the operator refits constants after install anyway.
_TEMPLATE_GRID_SIZE = 4


def build_library(
    rng: np.random.Generator, X: np.ndarray, functions: dict,
    const_range: tuple[float, float] = (-5.0, 5.0), size: int = 400, max_depth: int = 2,
    templates: bool = False,
) -> list[tuple[Node, np.ndarray]]:
    """Small subtrees with their outputs precomputed on ``X``.

    Depth is the parameter that decides what the operator can and cannot fix, and
    it is not a free choice. At ``max_depth=2`` this operator raises the rate at
    which Michaelis-Menten's structure is generated from 2/15 runs to 12/15,
    because the piece it needs -- ``K + x`` -- is depth 1 and sits in the library.
    Logistic and Gompertz stay at 0/15, because the pieces *they* need are depth
    3 and no single substitution can supply them.

    Going deeper costs matching time linearly and risks installing whole
    solutions rather than missing pieces, which would turn the search into random
    sampling over the library.

    ``templates`` additionally installs the curated depth-2/3 building blocks
    from :func:`_canonical_templates` before the random fill, so the deep pieces
    a growth law needs are available without deepening the random library.
    """
    from coevo.surrogates.evolved import _random_node

    dim = X.shape[1]
    library: list[tuple[Node, np.ndarray]] = []
    seen: set[str] = set()

    def add(node: Node) -> None:
        try:
            out = np.asarray(_eval(node, X, functions), dtype=float)
        except Exception:
            return
        if out.shape != (len(X),) or not np.all(np.isfinite(out)):
            return
        # Reject entries that run away: a subtree reaching the float limit is
        # not a useful building block, and rounding it for the dedup key
        # overflows. Mirrors the GP's own magnitude limit.
        if np.any(np.abs(out) > _MAGNITUDE_LIMIT):
            return
        key = np.array2string(np.round(out, 8), threshold=64)
        if key in seen:
            return
        seen.add(key)
        library.append((node, out))

    for i in range(dim):
        add(Node("x", idx=i))
    for value in np.linspace(const_range[0], const_range[1], 21):
        add(Node("c", val=float(value)))
    if templates:
        grid = np.linspace(const_range[0], const_range[1], _TEMPLATE_GRID_SIZE)
        for node in _canonical_templates(functions, grid):
            add(node)
    guard = 0
    while len(library) < size and guard < size * 40:
        guard += 1
        add(_random_node(rng, dim, 0, max_depth, const_range, functions))
    return library


def _match(library: list[tuple[Node, np.ndarray]], desired: np.ndarray) -> Node | None:
    """The library entry closest to ``desired`` over the positions that are defined."""
    valid = np.isfinite(desired)
    if valid.sum() < 2:
        return None
    target = desired[valid]
    best, best_err = None, np.inf
    for node, out in library:
        diff = out[valid] - target
        with np.errstate(over="ignore", invalid="ignore"):
            err = float(np.mean(diff * diff))
        if not np.isfinite(err):
            continue
        if err < best_err:
            best, best_err = node, err
    return None if best is None else _deep_copy(best)


def random_desired_operator(
    rng: np.random.Generator,
    tree: Node,
    X: np.ndarray,
    y: np.ndarray,
    functions: dict,
    library: list[tuple[Node, np.ndarray]],
    max_size: int = 60,
    linear_scaling: bool = True,
    attempts: int = 4,
) -> Node:
    """Replace one node with the library subtree closest to its desired output.

    Returns ``tree`` unchanged when no node yields an invertible path -- which is
    common and not a failure, just a node whose desired semantics are undefined.
    """
    paths = _paths(tree)
    if len(paths) <= 1:
        return tree
    order = rng.permutation(len(paths))
    for k in order[:attempts]:
        path = paths[int(k)]
        desired = desired_semantics(tree, path, X, y, functions, linear_scaling)
        if desired is None:
            continue
        replacement = _match(library, desired)
        if replacement is None:
            continue
        candidate = _deep_copy(tree)
        if not path:
            candidate = replacement
        else:
            parent = _at(candidate, path[:-1])
            args = list(parent.args)
            args[path[-1]] = replacement
            parent.args = tuple(args)
        if _size(candidate) <= max_size:
            return candidate
    return tree
