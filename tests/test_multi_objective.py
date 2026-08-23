"""Multi-objective optimization: NSGA-II, metrics, benchmarks, and SAEA."""

import numpy as np
import pytest

from coevo import (
    NearestNeighborSurrogate,
    NSGA2,
    SurrogateMultiObjectiveEvaluator,
    TrueMultiObjectiveEvaluator,
    benchmarks,
)
from coevo.core.metrics import (
    crowding_distance,
    dominates,
    fast_non_dominated_sort,
    hypervolume,
    igd,
    nondominated_mask,
)


def test_dominance():
    assert dominates(np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    assert not dominates(np.array([1.0, 2.0]), np.array([2.0, 1.0]))
    assert not dominates(np.array([1.0, 1.0]), np.array([1.0, 1.0]))


def test_non_dominated_sort_fronts():
    obj = np.array(
        [
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0],
            [4.0, 4.0],
            [1.0, 1.0],
        ]
    )
    fronts = fast_non_dominated_sort(obj)
    assert [4] in fronts  # the origin is alone in the first front
    assert sum(len(f) for f in fronts) == 5


def test_crowding_distance_bounds_are_infinite():
    obj = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    cd = crowding_distance(obj, [0, 1, 2])
    assert cd[0] == np.inf and cd[2] == np.inf
    assert np.isfinite(cd[1])


def test_hypervolume_exact_2d():
    front = np.array([[1.0, 5.0], [3.0, 2.0]])
    # union of [1,11]x[5,11] and [3,11]x[2,11] = 12 + 72 = 84
    assert hypervolume(front, np.array([11.0, 11.0])) == pytest.approx(84.0)


def test_igd_zero_for_perfect_front():
    ref = benchmarks.zdt1_front()
    assert igd(ref, ref) == pytest.approx(0.0, abs=1e-12)


def test_zdt1_optimum_front():
    p = benchmarks.zdt1(dim=10)
    # x = [t, 0, 0, ...] is Pareto-optimal for ZDT1
    x = np.zeros((3, 10))
    x[:, 0] = [0.0, 0.5, 1.0]
    obj = p.evaluate(x)
    np.testing.assert_allclose(obj[:, 1], 1 - np.sqrt(x[:, 0]), rtol=1e-6)


def test_nsga2_converges_on_zdt1():
    problem = benchmarks.zdt1(dim=10)
    result = NSGA2(pop_size=100, generations=250, seed=0).optimize(problem)
    ref = benchmarks.zdt1_front()
    assert len(result.objectives) > 1
    assert igd(result.objectives, ref) < 0.05
    # every reported solution is non-dominated
    assert nondominated_mask(result.objectives).all()


def test_surrogate_mo_evaluator_reduces_evals_and_converges():
    problem = benchmarks.zdt1(dim=10)
    ev = SurrogateMultiObjectiveEvaluator(
        problem, lambda: NearestNeighborSurrogate(), eval_fraction=0.3, warmup=5
    )
    result = NSGA2(pop_size=100, generations=250, seed=1).optimize(problem, ev)
    assert result.true_evaluations < 100 + 100 * 250
    assert igd(result.objectives, benchmarks.zdt1_front()) < 0.1


def test_true_mo_evaluator_counts():
    p = benchmarks.zdt1(dim=5)
    ev = TrueMultiObjectiveEvaluator(p)
    x = np.random.default_rng(0).random((10, 5))
    out = ev(x)
    assert out.shape == (10, 2)
    assert ev.n_true == 10
