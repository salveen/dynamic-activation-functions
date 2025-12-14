"""
Activation Function Classes

This module contains all activation function implementations following the Strategy Pattern.
"""

import numpy as np
from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    """
    Abstract base class for activation functions.
    Follows the Strategy Pattern and Open/Closed Principle.
    """
    
    @abstractmethod
    def compute(self, z: np.ndarray) -> np.ndarray:
        """Compute the activation function output."""
        pass
    
    @abstractmethod
    def train(self, z: np.ndarray, error: np.ndarray, learning_rate: float) -> None:
        """Train activation function parameters if applicable."""
        pass
    
    @abstractmethod
    def get_info(self) -> str:
        """Get information about the activation function."""
        pass


class FixedReLUActivation(ActivationFunction):
    """Standard ReLU activation: max(0, x)"""
    
    def compute(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)
    
    def train(self, z: np.ndarray, error: np.ndarray, learning_rate: float) -> None:
        """ReLU has no learnable parameters."""
        pass
    
    def get_info(self) -> str:
        return "Fixed ReLU: max(0, x)"


class DynamicReLUActivation(ActivationFunction):
    """Learnable ReLU activation: max(a, b*x)"""
    
    def __init__(self):
        self.a = 0.0  # threshold parameter
        self.b = 1.0  # slope parameter
    
    def compute(self, z: np.ndarray) -> np.ndarray:
        return np.where(self.b * z > self.a, self.b * z, self.a)
    
    def train(self, z: np.ndarray, error: np.ndarray, learning_rate: float) -> None:
        """Train a and b parameters using gradient descent."""
        # Compute masks for which part of max is active
        mask_a_active = (self.a >= self.b * z).astype(float)
        mask_b_active = (self.b * z > self.a).astype(float)
        
        # Compute gradients
        grad_a = np.mean(error * mask_a_active)
        grad_b = np.mean(error * mask_b_active * z)
        
        # Update parameters
        self.a -= learning_rate * grad_a
        self.b -= learning_rate * grad_b
    
    def get_info(self) -> str:
        return f"Dynamic ReLU: max({self.a:.4f}, {self.b:.4f}*x)"


class FixedStepActivation(ActivationFunction):
    """Fixed step activation with threshold at 0."""
    
    def compute(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1.0, 0.0)
    
    def train(self, z: np.ndarray, error: np.ndarray, learning_rate: float) -> None:
        """Step function has no learnable parameters."""
        pass
    
    def get_info(self) -> str:
        return "Fixed Step: step(x)"


class AdaptiveStepActivation(ActivationFunction):
    """Learnable step activation with adaptive threshold."""
    
    def __init__(self):
        self.threshold = 0.0
        self._steepness = 10.0  # For smooth gradient approximation
    
    def compute(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > self.threshold, 1.0, 0.0)
    
    def train(self, z: np.ndarray, error: np.ndarray, learning_rate: float) -> None:
        """Train threshold using sigmoid approximation for gradient."""
        # Use sigmoid approximation for smooth gradients
        sigmoid_approx = 1.0 / (1.0 + np.exp(-self._steepness * (z - self.threshold)))
        sigmoid_deriv = sigmoid_approx * (1.0 - sigmoid_approx)
        
        # Compute gradient w.r.t. threshold
        grad_threshold = -np.mean(error * sigmoid_deriv * self._steepness)
        
        # Update threshold
        self.threshold -= learning_rate * grad_threshold
    
    def get_info(self) -> str:
        return f"Adaptive Step: step(x - {self.threshold:.4f})"
