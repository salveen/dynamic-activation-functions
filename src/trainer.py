"""
Model Training and Evaluation

This module handles training orchestration and model evaluation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from models import Neuron


class ModelTrainer:
    """Orchestrates the training of different neuron models."""
    
    def __init__(
        self,
        activation_epochs: int = 1000,
        weight_epochs: int = 1000,
        seed: int = 42,
        weight_lr: float = 0.01,
        activation_lr: Optional[float] = None,
    ):
        self.activation_epochs = activation_epochs
        self.weight_epochs = weight_epochs
        self.seed = seed
        self.weight_lr = weight_lr
        self.activation_lr = activation_lr if activation_lr is not None else weight_lr
    
    def train_baseline_sklearn(self, X_train: np.ndarray, y_train: np.ndarray) -> Perceptron:
        """Train sklearn's perceptron as baseline."""
        model = Perceptron(max_iter=self.weight_epochs, eta0=0.01, random_state=self.seed)
        model.fit(X_train, y_train)
        return model

    def train_sklearn_sigmoid(self, X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
        """Train sklearn's MLPClassifier with sigmoid (logistic) activation as single neuron."""
        model = MLPClassifier(
            hidden_layer_sizes=(),  # No hidden layers = single neuron
            activation='logistic',  # Sigmoid activation
            max_iter=self.weight_epochs,
            learning_rate_init=0.01,
            random_state=self.seed
        )
        model.fit(X_train, y_train)
        return model
    
    def train_fixed_neuron(self, X_train: np.ndarray, y_train: np.ndarray, 
                          input_dim: int) -> Neuron:
        """Train neuron with fixed ReLU (no activation training)."""
        neuron = Neuron(
            input_dim,
            activation="fixed_relu",
            learning_rate=self.weight_lr,
            weight_lr=self.weight_lr,
        )
        neuron.train_weights(X_train, y_train, self.weight_epochs)
        return neuron
    
    def train_dynamic_neuron(self, X_train: np.ndarray, y_train: np.ndarray, 
                            input_dim: int) -> Neuron:
        """Train neuron with dynamic learnable activation."""
        neuron = Neuron(
            input_dim,
            activation="dynamic_relu",
            learning_rate=self.weight_lr,
            weight_lr=self.weight_lr,
            activation_lr=self.activation_lr,
        )
        neuron.train_weights(X_train, y_train, self.weight_epochs)
        neuron.train_activation(X_train, y_train, self.activation_epochs)
        return neuron

    def train_sigmoid_neuron(self, X_train: np.ndarray, y_train: np.ndarray,
                             input_dim: int) -> Neuron:
        """Train neuron with fixed sigmoid activation."""
        neuron = Neuron(
            input_dim,
            activation="fixed_sigmoid",
            learning_rate=self.weight_lr,
            weight_lr=self.weight_lr,
        )
        neuron.train_weights(X_train, y_train, self.weight_epochs)
        return neuron


# ==========================================
# EVALUATION
# ==========================================

@dataclass
class ModelResult:
    """Data class for storing model evaluation results."""
    dataset: str = ""
    model_name: str = ""
    accuracy: float = 0.0
    model_info: str = ""
    seed: int = 0


class ModelEvaluator:
    """
    Evaluates and compares different models.
    Follows Single Responsibility Principle.
    """
    
    def __init__(self, dataset_name: str = "unknown", seed: int = 0):
        self.results: List[ModelResult] = []
        self.dataset_name = dataset_name
        self.seed = seed
    
    def evaluate_sklearn_model(self, model: Perceptron, X_test: np.ndarray, 
                               y_test: np.ndarray, name: str) -> ModelResult:
        """Evaluate sklearn model."""
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        result = ModelResult(
            dataset=self.dataset_name,
            model_name=name,
            accuracy=accuracy,
            seed=self.seed
        )
        self.results.append(result)
        return result
    
    def evaluate_neuron(self, neuron: Neuron, X_test: np.ndarray, 
                       y_test: np.ndarray, name: str) -> ModelResult:
        """Evaluate custom neuron."""
        predictions = neuron.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        model_info = neuron.activation_info
        result = ModelResult(
            dataset=self.dataset_name,
            model_name=name,
            accuracy=accuracy,
            model_info=model_info,
            seed=self.seed
        )
        self.results.append(result)
        return result
    
    def print_comparison(self) -> None:
        """No-op for silent mode."""
        return
    
    def save_to_csv(self, filepath: str = "results.csv", append: bool = True) -> None:
        """
        Save results to CSV file.
        
        Args:
            filepath: Path to save CSV file
            append: If True, append to existing file. If False, overwrite.
        """
        if not self.results:
            return
        
        # Convert results to DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Check if file exists and append mode is on
        file_path = Path(filepath)
        if append and file_path.exists():
            # Load existing data and append
            existing_df = pd.read_csv(filepath)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        return
    
    def _analyze_improvements(self) -> None:
        """No-op for silent mode."""
        return
