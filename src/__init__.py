"""
Dynamic Activation Functions - Neural Network Library

This package provides neural network components with support for
both fixed and learnable (dynamic) activation functions.

Modules:
    - activations: Activation functions (ReLU, DynamicReLU, Sigmoid, DynamicSigmoid, DynamicReLUSigmoid, Softmax)
    - layers: Neural network layers (Dense, Dropout, BatchNorm)
    - mlp: Multi-Layer Perceptron implementation
    - mlp_trainer: Training utilities for MLPs
    - data_utils: Dataset loading and preprocessing
"""

from activations import (
    Activation,
    ReLU,
    DynamicReLU,
    Sigmoid,
    Softmax,
    LeakyReLU,
    create_activation,
)

from layers import Dense, Dropout, BatchNorm

from mlp import MLP, MLPConfig, create_baseline_mlp, create_dynamic_mlp

from mlp_trainer import MLPTrainer, MLPExperiment, TrainingHistory, ExperimentResult

__version__ = "0.2.0"
