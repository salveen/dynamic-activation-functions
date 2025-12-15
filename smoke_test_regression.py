"""Quick sanity checks for joint training (weights + activation params) on regression.

This is a lightweight script (no pytest) that verifies:
- `Neuron.train(..., loss='mse')` reduces MSE on a toy regression problem
- dynamic activation parameters update during training

Run: python smoke_test_regression.py
"""

from __future__ import annotations

import numpy as np

from models import Neuron


def mse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def main() -> None:
    rng = np.random.default_rng(0)

    # Toy regression: y = 2*x0 - 3*x1 + noise
    X = rng.normal(size=(400, 2))
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.1 * rng.normal(size=(400,))

    model = Neuron(input_dim=2, activation="dynamic_relu", learning_rate=0.05)

    y0 = model.predict(X)
    loss0 = mse(y, y0)
    a0 = model.activation_state.params["a"]
    b0 = model.activation_state.params["b"]

    model.train(X, y, epochs=200, batch_size=64, shuffle=True, seed=0, loss="mse")

    y1 = model.predict(X)
    loss1 = mse(y, y1)
    a1 = model.activation_state.params["a"]
    b1 = model.activation_state.params["b"]

    print(f"Initial MSE: {loss0:.4f}")
    print(f"Final MSE:   {loss1:.4f}")
    print(f"Activation a: {a0:.4f} -> {a1:.4f}")
    print(f"Activation b: {b0:.4f} -> {b1:.4f}")

    assert loss1 < loss0 * 0.5
    assert (abs(a1 - a0) + abs(b1 - b0)) > 1e-6


if __name__ == "__main__":
    main()
