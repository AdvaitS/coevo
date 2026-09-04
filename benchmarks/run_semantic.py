"""Measure whether the search *generates* ground-truth structures.

The bottleneck coevo is chasing is structural, not a constant-fitting problem:
plain GP assembles Michaelis-Menten's ``x/(K+x)`` but never logistic's or
Gompertz's depth-3 forms, across ~18,600 evaluations per run. This harness scores
*generation* — did the target structure appear in any evaluated tree — which is
cheap (a signature string match, no symbolic solver) and is the metric semantic
operators move directly.

Usage
-----
    python benchmarks/run_semantic.py                     # 3 laws x 4 seeds, all arms
    python benchmarks/run_semantic.py --seeds 8 --laws logistic_growth,gompertz_growth
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from coevo import SymbolicRegressor
from coevo.surrogates.evolved import structure_signature

# Ground-truth functional forms, constants collapsed to ``C``. The affine terms
# that linear scaling supplies (scale/offset) are *not* part of the structure.
TARGETS: dict[str, str] = {
    "michaelis_menten": "/(x0,+(C,x0))",
    "logistic_growth": "/(C,+(C,exp(+(C,*(C,x0)))))",
    "gompertz_growth": "exp(*(C,exp(*(C,x0))))",
}


def _make_functions(operators: str) -> dict:
    from biosym.operators import BIOLOGICAL_FUNCTIONS, operator_set

    return dict(BIOLOGICAL_FUNCTIONS, **operator_set(operators))


def generated_during_search(
    target: str, X, y, functions, *, semantic_p, templates, population_size, generations,
    const_range, seed,
) -> bool:
    """Whether ``target`` appears in any tree evaluated during one fit."""
    import coevo.surrogates.evolved as ev

    seen: set[str] = set()
    original = ev._raw_eval

    def recording(tree, Xa, fns):
        seen.add(structure_signature(tree))
        return original(tree, Xa, fns)

    ev._raw_eval = recording
    try:
        SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            functions=functions,
            const_range=const_range,
            seed=seed,
            semantic_p=semantic_p,
            library_templates=templates,
        ).fit(X, y)
    finally:
        ev._raw_eval = original
    return target in seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--laws", type=str, default=None)
    parser.add_argument("--semantic-p", type=float, default=0.4)
    parser.add_argument("--operators", type=str, default="plain")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=200)
    args = parser.parse_args()

    from biosym import benchmark_datasets

    laws = {k: v for k, v in benchmark_datasets.DATASETS.items() if k in TARGETS}
    if args.laws:
        wanted = {s.strip() for s in args.laws.split(",")}
        laws = {k: v for k, v in laws.items() if k in wanted}
    seeds = list(range(args.seeds))

    arms = [
        ("plain", dict(semantic_p=0.0, templates=False)),
        ("semantic", dict(semantic_p=args.semantic_p, templates=False)),
        ("semantic+templates", dict(semantic_p=args.semantic_p, templates=True)),
    ]

    print(f"# structure generated during search (over {len(seeds)} seeds, "
          f"gen={args.generations}, pop={args.population}, operators={args.operators})")
    print()
    header = "| law | " + " | ".join(name for name, _ in arms) + " |"
    print(header)
    print("|" + "---|" * (1 + len(arms)))

    totals = {name: 0 for name, _ in arms}
    secs = {name: 0.0 for name, _ in arms}
    for law, loader in laws.items():
        target = TARGETS[law]
        cells = []
        for arm_name, kw in arms:
            t0 = time.perf_counter()
            hits = 0
            for seed in seeds:
                X, y, _ = loader(n=60, seed=seed)
                hits += int(
                    generated_during_search(
                        target, X, y, _make_functions(args.operators),
                        population_size=args.population, generations=args.generations,
                        const_range=(-5.0, 5.0), seed=seed, **kw,
                    )
                )
            totals[arm_name] += hits
            secs[arm_name] += time.perf_counter() - t0
            cells.append(f"{hits}/{len(seeds)}")
        print(f"| {law} | " + " | ".join(cells) + " |")

    print()
    n = len(laws) * len(seeds)
    agg = [
        f"{name}: {totals[name]}/{n} generated, {secs[name]:.0f}s" for name, _ in arms
    ]
    print(" ; ".join(agg))


if __name__ == "__main__":
    main()
