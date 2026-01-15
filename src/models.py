"""Single neuron implementation with configurable activations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


VALID_ACTIVATIONS = {
    "fixed_relu",
    "dynamic_relu",
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
        """
        Make binary predictions.
        Returns: Array of 0s and 1s.
        """
        z = self._compute_weighted_sum(X)
        activation_output = self._activation_forward(z)
        return np.where(activation_output > 0.5, 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return continuous outputs in [0, 1] for probabilistic models."""
        z = self._compute_weighted_sum(X)
        return self._activation_forward(z)
    
    def train_activation(self, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        """Train only activation function parameters (freeze weights)."""
        if self.activation_state.name in {"fixed_relu"}:
            print("--- Stage 1: Training Activation Function (skipped for fixed activations) ---")
            return

        print("--- Stage 1: Training Activation Function ---")
        print(f"    Initial: {self.activation_info}")
        
        for epoch in range(epochs):
            # Forward pass
            z = self._compute_weighted_sum(X)
            predictions = self._activation_forward(z)
            
            # Compute error (MSE)
            error = predictions - y
            
            # Update activation function parameters
            self._train_activation_params(z, error)
        
        print(f"    Final: {self.activation_info}")
    
    def train_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        """Train weights with **mini-batch SGD** (fully differentiable path).

        Args:
            X: Features, shape (n_samples, n_features)
            y: Binary labels (0/1), shape (n_samples,)
            epochs: Number of passes over the dataset
            batch_size: Mini-batch size. If >= n_samples, behaves like full-batch GD.
            shuffle: Shuffle samples each epoch
            seed: Optional RNG seed for deterministic shuffling
        """

        print("--- Stage 2: Training Weights (Mini-batch SGD) ---")
        print(f"    Using: {self.activation_info}")

        y = y.astype(float)
        n = len(X)
        if n == 0:
            return

        bs = int(batch_size)
        if bs <= 0:
            raise ValueError("batch_size must be a positive integer")
        if bs > n:
            bs = n

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

                d_act = self._activation_derivative(z)
                dz = (p - yb) * d_act

                grad_w = (Xb.T @ dz) / len(Xb)
                grad_b = float(np.mean(dz))
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

    @property
    def activation_info(self) -> str:
        """Return a human-readable summary of the activation configuration."""
        if self.activation_state.name == "fixed_relu":
            return "Fixed ReLU: max(0, x)"
        if self.activation_state.name == "dynamic_relu":
            a = self.activation_state.params["a"]
            b = self.activation_state.params["b"]
            return f"Dynamic ReLU: max({a:.4f}, {b:.4f}*x)"
        return self.activation_state.name

    def _init_activation_params(self, activation: str) -> Dict[str, float]:
        if activation == "dynamic_relu":
            return {"a": 0.0, "b": 1.0}
        return {}

    def _activation_forward(self, z: np.ndarray) -> np.ndarray:
        if self.activation_state.name == "fixed_relu":
            return np.maximum(0, z)
        if self.activation_state.name == "dynamic_relu":
            a = self.activation_state.params["a"]
            b = self.activation_state.params["b"]
            return np.where(b * z > a, b * z, a)
        raise ValueError(f"Unknown activation: {self.activation_state.name}")

    def _activation_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivative of activation w.r.t z for continuous activations."""
        if self.activation_state.name == "fixed_relu":
            return (z > 0).astype(float)
        if self.activation_state.name == "dynamic_relu":
            a = self.activation_state.params["a"]
            b = self.activation_state.params["b"]
            return np.where(b * z > a, b, 0.0)
        return np.zeros_like(z, dtype=float)

    def _train_activation_params(self, z: np.ndarray, error: np.ndarray) -> None:
        if self.activation_state.name == "dynamic_relu":
            a = self.activation_state.params["a"]
            b = self.activation_state.params["b"]
            mask_a_active = (a >= b * z).astype(float)
            mask_b_active = (b * z > a).astype(float)
            grad_a = np.mean(error * mask_a_active)
            grad_b = np.mean(error * mask_b_active * z)
            self.activation_state.params["a"] -= self.learning_rate * grad_a
            self.activation_state.params["b"] -= self.learning_rate * grad_b