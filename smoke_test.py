"""Quick sanity checks for the sigmoid (soft perceptron) training path.

Run this to verify:
- forward pass returns probabilities in [0, 1]
- mini-batch SGD reduces BCE on a toy linearly-separable set

This is intentionally lightweight and doesn't require pytest.
"""

from __future__ import annotations

import numpy as np

from models import Neuron


def bce(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main() -> None:
    rng = np.random.default_rng(0)

    # Toy data: label is 1 if x0 + x1 > 0 else 0
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = Neuron(input_dim=2, activation="sigmoid", learning_rate=0.5)

    p0 = model.predict_proba(X)
    assert p0.shape == (200,)
    assert np.all((p0 >= 0) & (p0 <= 1))

    loss0 = bce(y, p0)
    model.train_weights(X, y, epochs=200, batch_size=32, shuffle=True, seed=0)
    p1 = model.predict_proba(X)
    loss1 = bce(y, p1)

    print(f"Initial BCE: {loss0:.4f}")
    print(f"Final BCE:   {loss1:.4f}")

    # Not too strict, but should noticeably improve.
    assert loss1 < loss0 * 0.8

    acc = float(np.mean(model.predict(X) == y))
    print(f"Train accuracy: {acc:.3f}")
    assert acc > 0.85


if __name__ == "__main__":
    main()
