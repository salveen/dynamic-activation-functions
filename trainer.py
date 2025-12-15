"""
Model Training and Evaluation

This module handles training orchestration and model evaluation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

from models import Neuron


class ModelTrainer:
    """Orchestrates the training of different neuron models."""
    
    def __init__(self, activation_epochs: int = 50, weight_epochs: int = 100):
        self.activation_epochs = activation_epochs
        self.weight_epochs = weight_epochs
    
    def train_baseline_sklearn(self, X_train: np.ndarray, y_train: np.ndarray) -> Perceptron:
        """Train sklearn's perceptron as baseline."""
        print("1. Training Sklearn Perceptron (Baseline)")
        model = Perceptron(max_iter=self.weight_epochs, eta0=0.01, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    def train_fixed_neuron(self, X_train: np.ndarray, y_train: np.ndarray, 
                          input_dim: int) -> Neuron:
        """Train neuron with fixed ReLU (no activation training)."""
        print("\n2. Training Fixed ReLU Neuron")
        neuron = Neuron(input_dim, activation="fixed_relu")
        # Fixed activation has no learnable params; just train weights.
        neuron.train(X_train, y_train, epochs=self.weight_epochs, loss="mse", train_activation=False)
        return neuron
    
    def train_dynamic_neuron(self, X_train: np.ndarray, y_train: np.ndarray, 
                            input_dim: int) -> Neuron:
        """Train neuron with dynamic learnable activation."""
        print("\n3. Training Dynamic ReLU Neuron")
        neuron = Neuron(input_dim, activation="dynamic_relu")
        # Seamless joint training: weights + activation together.
        neuron.train(X_train, y_train, epochs=self.weight_epochs, loss="mse", train_activation=True)
        return neuron
    
    def train_sigmoid_neuron(self, X_train: np.ndarray, y_train: np.ndarray,
                             input_dim: int) -> Neuron:
        """Train a sigmoid neuron (soft perceptron) with fully differentiable learning."""
        print("\n4. Training Sigmoid Neuron (Soft Perceptron)")
        neuron = Neuron(input_dim, activation="sigmoid")
        # Seamless joint training: weights + activation together.
        neuron.train(X_train, y_train, epochs=self.weight_epochs, loss="bce", train_activation=True)
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
    timestamp: str = ""


class ModelEvaluator:
    """
    Evaluates and compares different models.
    Follows Single Responsibility Principle.
    """
    
    def __init__(self, dataset_name: str = "unknown"):
        self.results: List[ModelResult] = []
        self.dataset_name = dataset_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def evaluate_sklearn_model(self, model: Perceptron, X_test: np.ndarray, 
                               y_test: np.ndarray, name: str) -> ModelResult:
        """Evaluate sklearn model."""
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        result = ModelResult(
            dataset=self.dataset_name,
            model_name=name,
            accuracy=accuracy,
            timestamp=self.timestamp
        )
        self.results.append(result)
        return result
    
    def evaluate_neuron(self, neuron: Neuron, X_test: np.ndarray, 
                       y_test: np.ndarray, name: str) -> ModelResult:
        """Evaluate custom neuron."""
        # `Neuron.predict()` is regression-first (continuous). For classification
        # experiments we use the legacy thresholding helper.
        predictions = neuron.predict_class(X_test)
        accuracy = accuracy_score(y_test, predictions)
        model_info = neuron.activation_info
        result = ModelResult(
            dataset=self.dataset_name,
            model_name=name,
            accuracy=accuracy,
            model_info=model_info,
            timestamp=self.timestamp
        )
        self.results.append(result)
        return result
    
    def print_comparison(self) -> None:
        """Print comprehensive comparison of all models."""
        print("\n" + "=" * 60)
        print("FINAL RESULTS COMPARISON")
        print("=" * 60)
        
        for result in self.results:
            print(f"\n{result.model_name}")
            print(f"  Accuracy: {result.accuracy:.4f}")
            if result.model_info:
                print(f"  Config: {result.model_info}")
        
        # Find best performer
        best_result = max(self.results, key=lambda r: r.accuracy)
        print("\n" + "-" * 60)
        print(f"Best Performer: {best_result.model_name} ({best_result.accuracy:.4f})")
        
        # Check if adaptive methods improved
        self._analyze_improvements()
    
    def save_to_csv(self, filepath: str = "results.csv", append: bool = True) -> None:
        """
        Save results to CSV file.
        
        Args:
            filepath: Path to save CSV file
            append: If True, append to existing file. If False, overwrite.
        """
        if not self.results:
            print("No results to save.")
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
        print(f"\n✓ Results saved to: {filepath}")
        print(f"  Total records: {len(df)}")
    
    def _analyze_improvements(self) -> None:
        """Analyze if adaptive activation functions showed improvement."""
        fixed_result = next((r for r in self.results if "Fixed" in r.model_name), None)
        dynamic_result = next((r for r in self.results if "Dynamic" in r.model_name), None)
        step_result = next((r for r in self.results if "Adaptive Step" in r.model_name), None)
        
        if not fixed_result:
            return
        
        improvements = []
        if dynamic_result and dynamic_result.accuracy > fixed_result.accuracy:
            improvements.append(f"  ✓ Dynamic ReLU improved by {(dynamic_result.accuracy - fixed_result.accuracy):.4f}")
        
        if step_result and step_result.accuracy > fixed_result.accuracy:
            improvements.append(f"  ✓ Adaptive Step improved by {(step_result.accuracy - fixed_result.accuracy):.4f}")
        
        if improvements:
            print("\nAdaptive Activation Success:")
            for imp in improvements:
                print(imp)
        else:
            print("\nNote: Adaptive activations did not outperform fixed ReLU.")
            print("Performance depends on data complexity and separability.")
