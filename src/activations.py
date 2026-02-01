"""
Activation functions for neural network layers.

Supports both fixed (standard) and dynamic (learnable) activations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class ActivationParams:
    """Holds learnable parameters for dynamic activations."""
    params: Dict[str, float] = field(default_factory=dict)


class Activation(ABC):
    """Base class for all activation functions."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.params = ActivationParams()
    
    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        """Compute activation output."""
        pass
    
    @abstractmethod
    def derivative(self, z: np.ndarray) -> np.ndarray:
        """Compute derivative w.r.t. pre-activation z."""
        pass
    
    def update_params(self, z: np.ndarray, error: np.ndarray) -> None:
        """Update learnable parameters (no-op for fixed activations)."""
        pass
    
    @property
    def is_learnable(self) -> bool:
        """Whether this activation has learnable parameters."""
        return False
    
    @property
    def info(self) -> str:
        """Human-readable description."""
        return self.__class__.__name__


class ReLU(Activation):
    """Standard ReLU: max(0, x)"""
    
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)
    
    def derivative(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)
    
    @property
    def info(self) -> str:
        return "ReLU: max(0, x)"


class DynamicReLU(Activation):
    """
    Dynamic ReLU with learnable parameters: max(a, b*x)
    
    Supports both per-layer (scalar a, b) and per-neuron (vector a, b) modes.
    
    Parameters:
        a: learnable threshold/floor value(s)
        b: learnable slope(s) for positive region
        num_neurons: If provided, creates per-neuron parameters
    """
    
    def __init__(self, learning_rate: float = 0.01, num_neurons: int = None):
        super().__init__(learning_rate)
        self.num_neurons = num_neurons
        self.per_neuron = num_neurons is not None
        
        if self.per_neuron:
            # Per-neuron parameters: each neuron has its own a, b
            self._a = np.zeros(num_neurons)
            self._b = np.ones(num_neurons)
        else:
            # Scalar parameters (shared across all neurons)
            self._a = 0.0
            self._b = 1.0
    
    @property
    def a(self) -> np.ndarray:
        return self._a
    
    @a.setter
    def a(self, value):
        self._a = value
    
    @property
    def b(self) -> np.ndarray:
        return self._b
    
    @b.setter
    def b(self, value):
        self._b = value
    
    def forward(self, z: np.ndarray) -> np.ndarray:
        # Works for both scalar and per-neuron (broadcasting handles it)
        return np.where(self.b * z > self.a, self.b * z, self.a)
    
    def derivative(self, z: np.ndarray) -> np.ndarray:
        return np.where(self.b * z > self.a, self.b, 0.0)
    
    def update_params(self, z: np.ndarray, error: np.ndarray) -> None:
        """Update a and b using gradient descent."""
        mask_a_active = (self.a >= self.b * z).astype(float)
        mask_b_active = (self.b * z > self.a).astype(float)
        
        if self.per_neuron:
            # Per-neuron gradients: average over batch, keep neuron dimension
            grad_a = np.mean(error * mask_a_active, axis=0)
            grad_b = np.mean(error * mask_b_active * z, axis=0)
        else:
            # Scalar gradients: average over everything
            grad_a = np.mean(error * mask_a_active)
            grad_b = np.mean(error * mask_b_active * z)
        
        self._a -= self.learning_rate * grad_a
        self._b -= self.learning_rate * grad_b
    
    @property
    def is_learnable(self) -> bool:
        return True
    
    @property
    def num_activation_params(self) -> int:
        """Return number of learnable activation parameters."""
        if self.per_neuron:
            return 2 * self.num_neurons
        return 2
    
    @property
    def info(self) -> str:
        if self.per_neuron:
            a_mean, a_std = np.mean(self._a), np.std(self._a)
            b_mean, b_std = np.mean(self._b), np.std(self._b)
            return f"DynamicReLU[{self.num_neurons}]: a={a_mean:.4f}±{a_std:.4f}, b={b_mean:.4f}±{b_std:.4f}"
        return f"DynamicReLU: max({self._a:.4f}, {self._b:.4f}*x)"


class Sigmoid(Activation):
    """Sigmoid activation: 1 / (1 + exp(-x))"""
    
    def forward(self, z: np.ndarray) -> np.ndarray:
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))
    
    def derivative(self, z: np.ndarray) -> np.ndarray:
        sig = self.forward(z)
        return sig * (1.0 - sig)
    
    @property
    def info(self) -> str:
        return "Sigmoid: 1 / (1 + exp(-x))"


class Softmax(Activation):
    """
    Softmax activation for multi-class output layer.
    
    Note: Derivative is the Jacobian, but for cross-entropy loss
    the combined gradient simplifies to (pred - target).
    """
    
    def forward(self, z: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    def derivative(self, z: np.ndarray) -> np.ndarray:
        # For softmax + cross-entropy, we handle this in the loss
        # Return ones as placeholder (actual gradient computed in loss)
        return np.ones_like(z)
    
    @property
    def info(self) -> str:
        return "Softmax"


class LeakyReLU(Activation):
    """Leaky ReLU: max(alpha*x, x) where alpha is small (e.g., 0.01)"""
    
    def __init__(self, alpha: float = 0.01, learning_rate: float = 0.01):
        super().__init__(learning_rate)
        self.alpha = alpha
    
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, z, self.alpha * z)
    
    def derivative(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1.0, self.alpha)
    
    @property
    def info(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"


# Factory function for creating activations
def create_activation(name: str, learning_rate: float = 0.01, **kwargs) -> Activation:
    """
    Factory function to create activation by name.
    
    Args:
        name: One of 'relu', 'dynamic_relu', 'sigmoid', 'softmax', 'leaky_relu'
        learning_rate: Learning rate for learnable activations
        **kwargs: Additional arguments (e.g., alpha for LeakyReLU)
    
    Returns:
        Activation instance
    """
    activations = {
        "relu": ReLU,
        "dynamic_relu": DynamicReLU,
        "sigmoid": Sigmoid,
        "softmax": Softmax,
        "leaky_relu": LeakyReLU,
    }
    
    name_lower = name.lower()
    if name_lower not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(activations.keys())}")
    
    if name_lower == "leaky_relu":
        return LeakyReLU(alpha=kwargs.get("alpha", 0.01), learning_rate=learning_rate)
    
    if name_lower == "dynamic_relu":
        num_neurons = kwargs.get("num_neurons", None)
        return DynamicReLU(learning_rate=learning_rate, num_neurons=num_neurons)
    
    return activations[name_lower](learning_rate=learning_rate)
