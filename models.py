"""Single neuron implementation with configurable activations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


VALID_ACTIVATIONS = {"fixed_relu", "dynamic_relu", "fixed_step", "adaptive_step"}


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
    
    def train_activation(self, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        """Train only activation function parameters (freeze weights)."""
        if self.activation_state.name in {"fixed_relu", "fixed_step"}:
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
    
    def train_weights(self, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        """Train weights using Perceptron Learning Rule (freeze activation)."""
        print("--- Stage 2: Training Weights (Perceptron Rule) ---")
        print(f"    Using: {self.activation_info}")
        
        for epoch in range(epochs):
            for i in range(len(X)):
                # Forward pass
                prediction = self.predict(X[i:i+1])[0]
                
                # Update only on misclassification
                if prediction != y[i]:
                    error = y[i] - prediction
                    
                    # Perceptron update rule
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error

    @property
    def activation_info(self) -> str:
        """Return a human-readable summary of the activation configuration."""
        if self.activation_state.name == "fixed_relu":
            return "Fixed ReLU: max(0, x)"
        if self.activation_state.name == "dynamic_relu":
            a = self.activation_state.params["a"]
            b = self.activation_state.params["b"]
            return f"Dynamic ReLU: max({a:.4f}, {b:.4f}*x)"
        if self.activation_state.name == "fixed_step":
            return "Fixed Step: step(x)"
        if self.activation_state.name == "adaptive_step":
            threshold = self.activation_state.params["threshold"]
            return f"Adaptive Step: step(x - {threshold:.4f})"
        return self.activation_state.name

    def _init_activation_params(self, activation: str) -> Dict[str, float]:
        if activation == "dynamic_relu":
            return {"a": 0.0, "b": 1.0}
        if activation == "adaptive_step":
            return {"threshold": 0.0, "steepness": 10.0}
        return {}

    def _activation_forward(self, z: np.ndarray) -> np.ndarray:
        if self.activation_state.name == "fixed_relu":
            return np.maximum(0, z)
        if self.activation_state.name == "dynamic_relu":
            a = self.activation_state.params["a"]
            b = self.activation_state.params["b"]
            return np.where(b * z > a, b * z, a)
        if self.activation_state.name == "fixed_step":
            return np.where(z > 0, 1.0, 0.0)
        if self.activation_state.name == "adaptive_step":
            threshold = self.activation_state.params["threshold"]
            return np.where(z > threshold, 1.0, 0.0)
        raise ValueError(f"Unknown activation: {self.activation_state.name}")

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
        elif self.activation_state.name == "adaptive_step":
            threshold = self.activation_state.params["threshold"]
            steepness = self.activation_state.params["steepness"]
            sigmoid_approx = 1.0 / (1.0 + np.exp(-steepness * (z - threshold)))
            sigmoid_deriv = sigmoid_approx * (1.0 - sigmoid_approx)
            grad_threshold = -np.mean(error * sigmoid_deriv * steepness)
            self.activation_state.params["threshold"] -= self.learning_rate * grad_threshold