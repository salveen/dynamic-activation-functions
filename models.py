"""Single neuron implementation with configurable activations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


VALID_ACTIVATIONS = {
    "fixed_relu",
    "dynamic_relu",
    "sigmoid",
}


@dataclass
class ActivationState:
    """Holds mutable parameters for learnable activations."""
    name: str
    params: Dict[str, float]


class Neuron:
    """A single neuron with optional learnable activation parameters."""

    def __init__(self, input_dim: int, activation: str = "fixed_relu", learning_rate: float = 0.01):
        if activation not in VALID_ACTIVATIONS:
            raise ValueError(f"Unsupported activation '{activation}'. Choose from {sorted(VALID_ACTIVATIONS)}")
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.activation_state = ActivationState(activation, self._init_activation_params(activation))
        self.weights = np.random.randn(input_dim) * 0.01
        self.bias = 0.0
    
    def _compute_weighted_sum(self, X: np.ndarray) -> np.ndarray:
        """Compute weighted sum: w·x + b"""
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the **continuous** neuron output.

        This is regression-first and is the most reusable behavior when you
        later move to an MLP.
        """
        z = self._compute_weighted_sum(X)
        return self._activation_forward(z)

    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Legacy binary predictions by thresholding the continuous output."""
        y_hat = self.predict(X)
        return np.where(y_hat > threshold, 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Alias for `predict()`.

        Note: outputs are only true probabilities for sigmoid-like activations.
        """
        return self.predict(X)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int | None = None,
        loss: str = "mse",
        train_weights: bool = True,
        train_activation: bool = True,
    ) -> None:
        """Jointly train weights **and** learnable activation parameters.

        This is the most natural training path when you later scale this code
        to an MLP: one forward pass, one loss, one update step.

        Args:
            X: Features, shape (n_samples, n_features)
            y: Targets, shape (n_samples,)
            epochs: Passes over the dataset
            batch_size: Mini-batch size
            shuffle: Shuffle each epoch
            seed: RNG seed for shuffling
            loss: "mse" (regression) or "bce" (probabilistic outputs)
            train_weights: Update weights/bias
            train_activation: Update activation parameters (if any)
        """

        y = y.astype(float)
        n = len(X)
        if n == 0:
            return

        bs = int(batch_size)
        if bs <= 0:
            raise ValueError("batch_size must be a positive integer")
        if bs > n:
            bs = n

        if loss not in {"mse", "bce"}:
            raise ValueError("loss must be one of: {'mse', 'bce'}")

        rng = np.random.default_rng(seed)

        for _epoch in range(epochs):
            if shuffle:
                idx = rng.permutation(n)
                X_epoch = X[idx]
                y_epoch = y[idx]
            else:
                X_epoch = X
                y_epoch = y

            for start in range(0, n, bs):
                Xb = X_epoch[start : start + bs]
                yb = y_epoch[start : start + bs]

                z = self._compute_weighted_sum(Xb)
                p = self._activation_forward(z)

                # dL/dp
                dL_dp = self._loss_grad_wrt_output(yb, p, loss)

                # dL/dz = dL/dp * dp/dz
                dp_dz = self._activation_derivative(z)
                dL_dz = dL_dp * dp_dz

                if train_weights:
                    grad_w = (Xb.T @ dL_dz) / len(Xb)
                    grad_b = float(np.mean(dL_dz))
                    self.weights -= self.learning_rate * grad_w
                    self.bias -= self.learning_rate * grad_b

                if train_activation and self.activation_state.params:
                    self._train_activation_params_joint(z, dL_dp)

    def _loss_grad_wrt_output(self, y: np.ndarray, p: np.ndarray, loss: str) -> np.ndarray:
        """Gradient of loss w.r.t. model output p (not pre-activation z)."""
        if loss == "mse":
            # L = 1/2 * (p - y)^2, so dL/dp = (p - y)
            return p - y
        if loss == "bce":
            # L = -[y log p + (1-y) log(1-p)]
            # dL/dp = (p - y) / (p (1-p))
            eps = 1e-12
            pc = np.clip(p, eps, 1.0 - eps)
            return (pc - y) / (pc * (1.0 - pc))
        raise ValueError(f"Unknown loss: {loss}")

    def _train_activation_params_joint(self, z: np.ndarray, dL_dp: np.ndarray) -> None:
        """Update activation parameters using chain rule from dL/dp.

        This keeps activation learning consistent with the chosen loss.
        """
        if self.activation_state.name == "dynamic_relu":
            a = self.activation_state.params["a"]
            b = self.activation_state.params["b"]
            # p = a if a >= b z else b z
            mask_a_active = (a >= b * z).astype(float)
            mask_b_active = (b * z > a).astype(float)
            dp_da = mask_a_active
            dp_db = mask_b_active * z
            grad_a = float(np.mean(dL_dp * dp_da))
            grad_b = float(np.mean(dL_dp * dp_db))
            self.activation_state.params["a"] -= self.learning_rate * grad_a
            self.activation_state.params["b"] -= self.learning_rate * grad_b
        elif self.activation_state.name == "sigmoid":
            threshold = self.activation_state.params["threshold"]
            steepness = self.activation_state.params["steepness"]
            p = self._activation_forward(z)
            # p = sigmoid(s(z - t)); dp/dt = -s p (1-p)
            dp_dthreshold = -steepness * p * (1.0 - p)
            grad_threshold = float(np.mean(dL_dp * dp_dthreshold))
            self.activation_state.params["threshold"] -= self.learning_rate * grad_threshold

            # dp/ds = (z - t) p (1-p)
            dp_dsteepness = (z - threshold) * p * (1.0 - p)
            grad_steepness = float(np.mean(dL_dp * dp_dsteepness))
            self.activation_state.params["steepness"] -= self.learning_rate * grad_steepness
            self.activation_state.params["steepness"] = float(np.clip(self.activation_state.params["steepness"], 0.1, 50.0))

    @property
    def activation_info(self) -> str:
        """Return a human-readable summary of the activation configuration."""
        if self.activation_state.name == "fixed_relu":
            return "Fixed ReLU: max(0, x)"
        if self.activation_state.name == "dynamic_relu":
            a = self.activation_state.params["a"]
            b = self.activation_state.params["b"]
            return f"Dynamic ReLU: max({a:.4f}, {b:.4f}*x)"
        if self.activation_state.name == "sigmoid":
            threshold = self.activation_state.params["threshold"]
            steepness = self.activation_state.params["steepness"]
            return f"Sigmoid: sigmoid({steepness:.4f}*(x - {threshold:.4f}))"
        return self.activation_state.name

    def _init_activation_params(self, activation: str) -> Dict[str, float]:
        if activation == "dynamic_relu":
            return {"a": 0.0, "b": 1.0}
        if activation == "sigmoid":
            # Steepness controls how close we are to a hard step.
            # Keep it moderate so gradients don't vanish immediately.
            return {"threshold": 0.0, "steepness": 1.0}
        return {}

    def _activation_forward(self, z: np.ndarray) -> np.ndarray:
        if self.activation_state.name == "fixed_relu":
            return np.maximum(0, z)
        if self.activation_state.name == "dynamic_relu":
            a = self.activation_state.params["a"]
            b = self.activation_state.params["b"]
            return np.where(b * z > a, b * z, a)
        if self.activation_state.name == "sigmoid":
            threshold = self.activation_state.params["threshold"]
            steepness = self.activation_state.params["steepness"]
            t = steepness * (z - threshold)
            # Numerically stable sigmoid
            t = np.clip(t, -60.0, 60.0)
            return 1.0 / (1.0 + np.exp(-t))
        raise ValueError(f"Unknown activation: {self.activation_state.name}")

    def _activation_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivative of activation w.r.t z for continuous activations."""
        if self.activation_state.name == "fixed_relu":
            return (z > 0).astype(float)
        if self.activation_state.name == "dynamic_relu":
            a = self.activation_state.params["a"]
            b = self.activation_state.params["b"]
            return np.where(b * z > a, b, 0.0)
        if self.activation_state.name == "sigmoid":
            p = self._activation_forward(z)
            steepness = self.activation_state.params["steepness"]
            return steepness * p * (1.0 - p)
        return np.zeros_like(z, dtype=float)
