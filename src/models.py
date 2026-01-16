"""Single neuron implementation with configurable activations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


VALID_ACTIVATIONS = {
    "fixed_relu",
    "dynamic_relu",
    "fixed_sigmoid",
}


@dataclass
class ActivationState:
    """Holds mutable parameters for learnable activations."""
    name: str
    params: Dict[str, float]


class Neuron:
    """A single neuron with optional learnable activation parameters."""

    def __init__(
        self,
        input_dim: int,
        activation: str = "fixed_relu",
        learning_rate: float = 0.01,
        weight_lr: float | None = None,
        activation_lr: float | None = None,
    ):
        if activation not in VALID_ACTIVATIONS:
            raise ValueError(f"Unsupported activation '{activation}'. Choose from {sorted(VALID_ACTIVATIONS)}")
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weight_lr = weight_lr if weight_lr is not None else learning_rate
        self.activation_lr = activation_lr if activation_lr is not None else learning_rate
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
    
    def train_activation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        patience: int = 20,
        min_delta: float = 1e-6,
    ) -> None:
        """Train only activation function parameters (freeze weights).
        
        Args:
            X: Features
            y: Binary labels
            epochs: Max number of epochs (early stopping may halt sooner)
            patience: Stop if loss doesn't improve for this many consecutive epochs
            min_delta: Minimum change in loss to count as improvement
        """
        if self.activation_state.name in {"fixed_relu", "fixed_sigmoid"}:
            return
        
        best_loss = float('inf')
        epochs_without_improvement = 0
        y = y.astype(float)
        
        for epoch in range(epochs):
            # Forward pass
            z = self._compute_weighted_sum(X)
            predictions = self._activation_forward(z)
            
            # Compute error (MSE)
            error = predictions - y
            
            # Update activation function parameters
            self._train_activation_params(z, error)
            
            # Compute loss for early stopping
            epoch_loss = float(np.mean(error ** 2))
            
            if epoch_loss < best_loss - min_delta:
                best_loss = epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break  # Early stopping
        
    
    def train_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int | None = None,
        patience: int = 10,
        min_delta: float = 1e-6,
    ) -> None:
        """Train weights with **mini-batch SGD** (fully differentiable path).

        Args:
            X: Features, shape (n_samples, n_features)
            y: Binary labels (0/1), shape (n_samples,)
            epochs: Max number of passes over the dataset (early stopping may halt sooner)
            batch_size: Mini-batch size. If >= n_samples, behaves like full-batch GD.
            shuffle: Shuffle samples each epoch
            seed: Optional RNG seed for deterministic shuffling
            patience: Stop if loss doesn't improve for this many consecutive epochs
            min_delta: Minimum change in loss to count as improvement
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

        rng = np.random.default_rng(seed)

        best_loss = float('inf')
        epochs_without_improvement = 0

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
                self.weights -= self.weight_lr * grad_w
                self.bias -= self.weight_lr * grad_b

            # Compute epoch loss for early stopping
            z_full = self._compute_weighted_sum(X)
            p_full = self._activation_forward(z_full)
            epoch_loss = float(np.mean((p_full - y) ** 2))

            if epoch_loss < best_loss - min_delta:
                best_loss = epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break  # Early stopping

    @property
    def activation_info(self) -> str:
        """Return a human-readable summary of the activation configuration."""
        if self.activation_state.name == "fixed_relu":
            return "Fixed ReLU: max(0, x)"
        if self.activation_state.name == "dynamic_relu":
            a = self.activation_state.params["a"]
            b = self.activation_state.params["b"]
            return f"Dynamic ReLU: max({a:.4f}, {b:.4f}*x)"
        if self.activation_state.name == "fixed_sigmoid":
            return "Fixed Sigmoid: 1 / (1 + exp(-x))"
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
        if self.activation_state.name == "fixed_sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        raise ValueError(f"Unknown activation: {self.activation_state.name}")

    def _activation_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivative of activation w.r.t z for continuous activations."""
        if self.activation_state.name == "fixed_relu":
            return (z > 0).astype(float)
        if self.activation_state.name == "dynamic_relu":
            a = self.activation_state.params["a"]
            b = self.activation_state.params["b"]
            return np.where(b * z > a, b, 0.0)
        if self.activation_state.name == "fixed_sigmoid":
            sig = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            return sig * (1.0 - sig)
        return np.zeros_like(z, dtype=float)

    def _train_activation_params(self, z: np.ndarray, error: np.ndarray) -> None:
        if self.activation_state.name == "dynamic_relu":
            a = self.activation_state.params["a"]
            b = self.activation_state.params["b"]
            mask_a_active = (a >= b * z).astype(float)
            mask_b_active = (b * z > a).astype(float)
            grad_a = np.mean(error * mask_a_active)
            grad_b = np.mean(error * mask_b_active * z)
            self.activation_state.params["a"] -= self.activation_lr * grad_a
            self.activation_state.params["b"] -= self.activation_lr * grad_b