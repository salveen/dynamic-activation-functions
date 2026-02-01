"""
Neural network layers for MLP architecture.

Provides Dense (fully connected) layers with configurable activations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from activations import Activation, ReLU, create_activation


class Dense:
    """
    Fully connected (dense) layer.
    
    Computes: output = activation(X @ W + b)
    
    Attributes:
        input_dim: Number of input features
        output_dim: Number of output neurons
        activation: Activation function instance
        weights: Weight matrix of shape (input_dim, output_dim)
        bias: Bias vector of shape (output_dim,)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Optional[Activation] = None,
        weight_init: str = "he",
    ):
        """
        Initialize Dense layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output neurons  
            activation: Activation function (default: ReLU)
            weight_init: Weight initialization strategy ('he', 'xavier', 'normal')
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation or ReLU()
        
        # Initialize weights
        self.weights, self.bias = self._init_weights(weight_init)
        
        # Cache for backpropagation
        self._input: Optional[np.ndarray] = None
        self._z: Optional[np.ndarray] = None  # Pre-activation
        self._output: Optional[np.ndarray] = None
    
    def _init_weights(self, strategy: str) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize weights using specified strategy."""
        if strategy == "he":
            # He initialization (good for ReLU)
            std = np.sqrt(2.0 / self.input_dim)
        elif strategy == "xavier":
            # Xavier/Glorot initialization (good for tanh/sigmoid)
            std = np.sqrt(2.0 / (self.input_dim + self.output_dim))
        else:
            # Simple normal initialization
            std = 0.01
        
        weights = np.random.randn(self.input_dim, self.output_dim) * std
        bias = np.zeros(self.output_dim)
        
        return weights, bias
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        
        Args:
            X: Input of shape (batch_size, input_dim)
            
        Returns:
            Output of shape (batch_size, output_dim)
        """
        self._input = X
        self._z = X @ self.weights + self.bias
        self._output = self.activation.forward(self._z)
        return self._output
    
    def backward(
        self,
        grad_output: np.ndarray,
        learning_rate: float,
        update_activation: bool = True,
        update_weights: bool = True,
    ) -> np.ndarray:
        """
        Backward pass: compute gradients and update weights.
        
        Args:
            grad_output: Gradient from next layer, shape (batch_size, output_dim)
            learning_rate: Learning rate for weight updates
            update_activation: Whether to update activation parameters
            update_weights: Whether to update weight parameters (False = frozen)
            
        Returns:
            Gradient to pass to previous layer, shape (batch_size, input_dim)
        """
        batch_size = grad_output.shape[0]
        
        # Update learnable activation parameters BEFORE computing gradient through it
        if update_activation and self.activation.is_learnable:
            self.activation.update_params(self._z, grad_output)
        
        # Gradient through activation
        grad_z = grad_output * self.activation.derivative(self._z)
        
        # Gradient to pass to previous layer (always compute for backprop)
        grad_input = grad_z @ self.weights.T
        
        # Update weights only if not frozen
        if update_weights:
            grad_weights = (self._input.T @ grad_z) / batch_size
            grad_bias = np.mean(grad_z, axis=0)
            self.weights -= learning_rate * grad_weights
            self.bias -= learning_rate * grad_bias
        
        return grad_input
    
    def update_activation_params(self, error: np.ndarray) -> None:
        """Update learnable activation parameters if applicable."""
        if self.activation.is_learnable:
            self.activation.update_params(self._z, error)
    
    @property
    def info(self) -> str:
        """Layer description."""
        return f"Dense({self.input_dim} -> {self.output_dim}, {self.activation.info})"
    
    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return self.weights.size + self.bias.size


class Dropout:
    """
    Dropout regularization layer.
    
    Randomly sets a fraction of inputs to zero during training.
    """
    
    def __init__(self, rate: float = 0.5):
        """
        Args:
            rate: Fraction of inputs to drop (0 to 1)
        """
        if not 0 <= rate < 1:
            raise ValueError("Dropout rate must be in [0, 1)")
        self.rate = rate
        self._mask: Optional[np.ndarray] = None
        self.training = True
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply dropout during training."""
        if self.training and self.rate > 0:
            self._mask = np.random.binomial(1, 1 - self.rate, X.shape) / (1 - self.rate)
            return X * self._mask
        return X
    
    def backward(self, grad_output: np.ndarray, learning_rate: float, update_activation: bool = True, update_weights: bool = True) -> np.ndarray:
        """Pass gradient through dropout mask."""
        if self.training and self.rate > 0:
            return grad_output * self._mask
        return grad_output
    
    @property
    def info(self) -> str:
        return f"Dropout(rate={self.rate})"


class BatchNorm:
    """
    Batch Normalization layer.
    
    Normalizes activations to have zero mean and unit variance.
    """
    
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        self.training = True
        
        # Cache for backprop
        self._input_norm: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._input: Optional[np.ndarray] = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Normalize and scale input."""
        self._input = X
        
        if self.training:
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        self._std = np.sqrt(var + self.eps)
        self._input_norm = (X - mean) / self._std
        
        return self.gamma * self._input_norm + self.beta
    
    def backward(self, grad_output: np.ndarray, learning_rate: float, update_activation: bool = True) -> np.ndarray:
        """Backprop through batch norm."""
        batch_size = grad_output.shape[0]
        
        # Gradients for gamma and beta
        grad_gamma = np.sum(grad_output * self._input_norm, axis=0)
        grad_beta = np.sum(grad_output, axis=0)
        
        # Gradient through normalization
        grad_norm = grad_output * self.gamma
        grad_var = np.sum(grad_norm * (self._input - np.mean(self._input, axis=0)) * -0.5 * (self._std ** -3), axis=0)
        grad_mean = np.sum(grad_norm * -1 / self._std, axis=0)
        
        grad_input = grad_norm / self._std + grad_var * 2 * (self._input - np.mean(self._input, axis=0)) / batch_size + grad_mean / batch_size
        
        # Update parameters
        self.gamma -= learning_rate * grad_gamma
        self.beta -= learning_rate * grad_beta
        
        return grad_input
    
    @property
    def info(self) -> str:
        return f"BatchNorm({self.num_features})"
