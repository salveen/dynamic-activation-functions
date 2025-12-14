"""
Neuron Factory

This module provides factory methods for creating different neuron configurations.
"""

from models import Neuron
from activation_functions import (
    FixedReLUActivation,
    DynamicReLUActivation,
    FixedStepActivation,
    AdaptiveStepActivation
)


class NeuronFactory:
    """
    Factory for creating neurons with different configurations.
    Follows Factory Pattern and Open/Closed Principle.
    """
    
    @staticmethod
    def create_fixed_relu_neuron(input_dim: int, learning_rate: float = 0.01) -> Neuron:
        """Create a neuron with fixed ReLU activation."""
        return Neuron(input_dim, FixedReLUActivation(), learning_rate)
    
    @staticmethod
    def create_dynamic_relu_neuron(input_dim: int, learning_rate: float = 0.01) -> Neuron:
        """Create a neuron with learnable dynamic ReLU activation."""
        return Neuron(input_dim, DynamicReLUActivation(), learning_rate)
    
    @staticmethod
    def create_fixed_step_neuron(input_dim: int, learning_rate: float = 0.01) -> Neuron:
        """Create a neuron with fixed step activation (threshold at 0)."""
        return Neuron(input_dim, FixedStepActivation(), learning_rate)
    
    @staticmethod
    def create_adaptive_step_neuron(input_dim: int, learning_rate: float = 0.01) -> Neuron:
        """Create a neuron with learnable step activation."""
        return Neuron(input_dim, AdaptiveStepActivation(), learning_rate)
