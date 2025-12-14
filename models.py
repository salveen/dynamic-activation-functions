"""
Neuron Model

This module contains the core neuron implementation.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from activation_functions import ActivationFunction


class Neuron:
    """
    A single neuron with configurable activation function.
    """
    
    def __init__(self, input_dim: int, activation_fn: 'ActivationFunction', learning_rate: float = 0.01):
        self.input_dim = input_dim
        self.activation_fn = activation_fn
        self.learning_rate = learning_rate
        
        # Initialize weights and bias
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
        activation_output = self.activation_fn.compute(z)
        return np.where(activation_output > 0.5, 1, 0)
    
    def train_activation(self, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        """Train only activation function parameters (freeze weights)."""
        print(f"--- Stage 1: Training Activation Function ---")
        print(f"    Initial: {self.activation_fn.get_info()}")
        
        for epoch in range(epochs):
            # Forward pass
            z = self._compute_weighted_sum(X)
            predictions = self.activation_fn.compute(z)
            
            # Compute error (MSE)
            error = predictions - y
            
            # Update activation function parameters
            self.activation_fn.train(z, error, self.learning_rate)
        
        print(f"    Final: {self.activation_fn.get_info()}")
    
    def train_weights(self, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        """Train weights using Perceptron Learning Rule (freeze activation)."""
        print(f"--- Stage 2: Training Weights (Perceptron Rule) ---")
        print(f"    Using: {self.activation_fn.get_info()}")
        
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