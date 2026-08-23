"""Demonstrate the evolved symbolic fitness predictor.

Evolves a compact, human-readable expression that predicts fitness, and prints
what it discovered. Run with::

    python examples/evolved_predictor.py
"""

import numpy as np

from coevo import EvolvedPredictor, SymbolicRegressor


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(80, 3))

    # Hidden ground truth: f(x) = x0^2 + sin(x1) - 0.5*x2
    y = X[:, 0] ** 2 + np.sin(X[:, 1]) - 0.5 * X[:, 2]

    predictor = EvolvedPredictor(
        population_size=300, generations=40, max_depth=5, seed=0
    ).fit(X, y)

    pred = predictor.predict(X)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    print(f"evolved expression:  f(x) = {predictor.expression}")
    print(f"RMSE on training data: {rmse:.4f}")


if __name__ == "__main__":
    main()
