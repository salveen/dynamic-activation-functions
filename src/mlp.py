"""
Multi-Layer Perceptron (MLP) implementation.

Provides a flexible, configurable MLP architecture for classification tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

from activations import Activation, ReLU, DynamicReLU, Sigmoid, Softmax, create_activation
from layers import Dense, Dropout


@dataclass
class MLPConfig:
    """Configuration for MLP architecture."""
    input_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    output_dim: int = 10
    hidden_activation: str = "relu"  # 'relu' or 'dynamic_relu'
    output_activation: str = "softmax"  # 'softmax' for multi-class, 'sigmoid' for binary
    dropout_rate: float = 0.0
    weight_init: str = "he"
    learning_rate: float = 0.01
    activation_lr: Optional[float] = None  # LR for dynamic activation params
    per_neuron_activation: bool = False  # If True, each neuron has its own a, b
    
    def __post_init__(self):
        if self.activation_lr is None:
            self.activation_lr = self.learning_rate


class MLP:
    """
    Multi-Layer Perceptron for classification.
    
    Architecture: Input -> [Hidden + Activation + Dropout]* -> Output
    
    Supports both fixed (ReLU) and dynamic (learnable) activations.
    """
    
    def __init__(self, config: MLPConfig):
        """
        Initialize MLP from config.
        
        Args:
            config: MLPConfig specifying architecture
        """
        self.config = config
        self.layers: List[Union[Dense, Dropout]] = []
        self._build_network()
        self.training = True
    
    def _build_network(self) -> None:
        """Construct the network layers."""
        dims = [self.config.input_dim] + self.config.hidden_dims + [self.config.output_dim]
        
        for i in range(len(dims) - 1):
            is_output_layer = (i == len(dims) - 2)
            output_neurons = dims[i + 1]
            
            # Choose activation
            if is_output_layer:
                activation = create_activation(
                    self.config.output_activation,
                    learning_rate=self.config.activation_lr
                )
            else:
                # For hidden layers, optionally use per-neuron dynamic activations
                num_neurons = output_neurons if self.config.per_neuron_activation else None
                activation = create_activation(
                    self.config.hidden_activation,
                    learning_rate=self.config.activation_lr,
                    num_neurons=num_neurons,
                )
            
            # Create dense layer
            layer = Dense(
                input_dim=dims[i],
                output_dim=dims[i + 1],
                activation=activation,
                weight_init=self.config.weight_init,
            )
            self.layers.append(layer)
            
            # Add dropout after hidden layers (not output)
            if not is_output_layer and self.config.dropout_rate > 0:
                self.layers.append(Dropout(rate=self.config.dropout_rate))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            X: Input of shape (batch_size, input_dim)
            
        Returns:
            Output of shape (batch_size, output_dim)
        """
        output = X
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = self.training
            output = layer.forward(output)
        return output
    
    def backward(
        self, 
        grad_output: np.ndarray, 
        update_activation: bool = True,
        update_weights: bool = True,
    ) -> None:
        """
        Backward pass: update layer parameters.
        
        Args:
            grad_output: Gradient of loss w.r.t. output
            update_activation: Whether to update learnable activation params
            update_weights: Whether to update weight params (False = frozen weights)
        """
        grad = grad_output
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad, self.config.learning_rate, update_activation, update_weights)
    
    def train_activation_params(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> None:
        """
        Train only the learnable activation parameters (freeze weights).
        
        Only affects layers with dynamic activations.
        """
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute error
            error = output - y
            
            # Update activation params in each layer
            grad = error
            for layer in reversed(self.layers):
                if isinstance(layer, Dense):
                    layer.update_activation_params(grad)
                    # Backprop error for next layer
                    grad = grad @ layer.weights.T
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions (class labels).
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        self.training = False
        probs = self.forward(X)
        self.training = True
        
        if self.config.output_activation == "sigmoid":
            return (probs > 0.5).astype(int).flatten()
        else:
            return np.argmax(probs, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability predictions."""
        self.training = False
        probs = self.forward(X)
        self.training = True
        return probs
    
    def eval(self) -> None:
        """Set network to evaluation mode."""
        self.training = False
    
    def train(self) -> None:
        """Set network to training mode."""
        self.training = True
    
    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(
            layer.num_params for layer in self.layers 
            if isinstance(layer, Dense)
        )
    
    def summary(self) -> str:
        """Return a summary of the network architecture."""
        lines = ["MLP Architecture:"]
        lines.append("-" * 50)
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                lines.append(f"  {i}: {layer.info}")
                lines.append(f"      Parameters: {layer.num_params:,}")
                total_params += layer.num_params
            else:
                lines.append(f"  {i}: {layer.info}")
        
        lines.append("-" * 50)
        lines.append(f"Total Parameters: {total_params:,}")
        
        return "\n".join(lines)


def create_baseline_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    learning_rate: float = 0.01,
    dropout_rate: float = 0.0,
) -> MLP:
    """
    Create baseline MLP with standard ReLU activations.
    
    Architecture: ReLU hidden layers + Softmax output
    """
    config = MLPConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        hidden_activation="relu",
        output_activation="softmax",
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )
    return MLP(config)


def create_dynamic_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    learning_rate: float = 0.01,
    activation_lr: float = 0.01,
    dropout_rate: float = 0.0,
) -> MLP:
    """
    Create MLP with dynamic (learnable) ReLU activations.
    
    Architecture: Dynamic ReLU hidden layers + Softmax output
    """
    config = MLPConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        hidden_activation="dynamic_relu",
        output_activation="softmax",
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        activation_lr=activation_lr,
    )
    return MLP(config)
