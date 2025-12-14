"""
Perceptron Implementation with SOLID Principles and OOP Best Practices

This module demonstrates various perceptron implementations with different
activation functions, following SOLID principles for maintainability and extensibility.
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================
# ACTIVATION FUNCTIONS (Strategy Pattern)
# ==========================================

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


# ==========================================
# TRAINING STRATEGIES (Strategy Pattern)
# ==========================================

class TrainingStrategy(ABC):
    """Abstract base class for training strategies."""
    
    @abstractmethod
    def train(self, neuron: 'Neuron', X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        """Execute the training strategy."""
        pass


class ActivationTrainingStrategy(TrainingStrategy):
    """Strategy for training only activation function parameters."""
    
    def train(self, neuron: 'Neuron', X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        print(f"--- Stage 1: Training Activation Function ---")
        print(f"    Initial: {neuron.activation_fn.get_info()}")
        
        for epoch in range(epochs):
            # Forward pass
            z = neuron._compute_weighted_sum(X)
            predictions = neuron.activation_fn.compute(z)
            
            # Compute error (MSE)
            error = predictions - y
            
            # Update activation function parameters
            neuron.activation_fn.train(z, error, neuron.learning_rate)
        
        print(f"    Final: {neuron.activation_fn.get_info()}")


class WeightTrainingStrategy(TrainingStrategy):
    """Strategy for training weights using Perceptron Learning Rule."""
    
    def train(self, neuron: 'Neuron', X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        print(f"--- Stage 2: Training Weights (Perceptron Rule) ---")
        print(f"    Using: {neuron.activation_fn.get_info()}")
        
        for epoch in range(epochs):
            misclassifications = 0
            
            for i in range(len(X)):
                # Forward pass
                prediction = neuron.predict(X[i:i+1])[0]
                
                # Update only on misclassification
                if prediction != y[i]:
                    misclassifications += 1
                    error = y[i] - prediction
                    
                    # Perceptron update rule
                    neuron.weights += neuron.learning_rate * error * X[i]
                    neuron.bias += neuron.learning_rate * error


# ==========================================
# NEURON CLASS (Single Responsibility)
# ==========================================

class Neuron:
    """
    A single neuron with configurable activation function.
    Follows Single Responsibility Principle - handles only neuron computation.
    """
    
    def __init__(self, input_dim: int, activation_fn: ActivationFunction, learning_rate: float = 0.01):
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
    
    def train_with_strategy(self, strategy: TrainingStrategy, X: np.ndarray, 
                           y: np.ndarray, epochs: int) -> None:
        """Train the neuron using a specific training strategy."""
        strategy.train(self, X, y, epochs)



# ==========================================
# MODEL FACTORY (Factory Pattern)
# ==========================================

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


# ==========================================
# DATA MANAGER (Single Responsibility)
# ==========================================

@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    n_samples: int = 1000
    n_features: int = 20
    n_informative: int = 6
    n_redundant: int = 8
    n_repeated: int = 4
    n_clusters_per_class: int = 2
    class_sep: float = 0.5
    flip_y: float = 0.15
    test_size: float = 0.2
    random_state: int = 42


class DataManager:
    """
    Handles data generation, preprocessing, and splitting.
    Follows Single Responsibility Principle.
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.scaler = StandardScaler()
    
    def generate_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate and preprocess classification dataset.
        Returns: X_train, X_test, y_train, y_test
        """
        # Generate synthetic dataset
        X, y = make_classification(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            n_informative=self.config.n_informative,
            n_redundant=self.config.n_redundant,
            n_repeated=self.config.n_repeated,
            n_clusters_per_class=self.config.n_clusters_per_class,
            class_sep=self.config.class_sep,
            flip_y=self.config.flip_y,
            random_state=self.config.random_state
        )
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state
        )
        
        print(f"Dataset Generated: X_train {X_train.shape}, y_train {y_train.shape}")
        print("-" * 60)
        
        return X_train, X_test, y_train, y_test


# ==========================================
# MODEL TRAINER (Single Responsibility)
# ==========================================

class ModelTrainer:
    """
    Orchestrates the training of different neuron models.
    Follows Single Responsibility Principle.
    """
    
    def __init__(self, activation_epochs: int = 50, weight_epochs: int = 100):
        self.activation_epochs = activation_epochs
        self.weight_epochs = weight_epochs
        self.activation_strategy = ActivationTrainingStrategy()
        self.weight_strategy = WeightTrainingStrategy()
    
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
        neuron = NeuronFactory.create_fixed_relu_neuron(input_dim)
        neuron.train_with_strategy(self.weight_strategy, X_train, y_train, self.weight_epochs)
        return neuron
    
    def train_dynamic_neuron(self, X_train: np.ndarray, y_train: np.ndarray, 
                            input_dim: int) -> Neuron:
        """Train neuron with dynamic learnable activation."""
        print("\n3. Training Dynamic ReLU Neuron")
        neuron = NeuronFactory.create_dynamic_relu_neuron(input_dim)
        neuron.train_with_strategy(self.activation_strategy, X_train, y_train, self.activation_epochs)
        neuron.train_with_strategy(self.weight_strategy, X_train, y_train, self.weight_epochs)
        return neuron
    
    def train_fixed_step_neuron(self, X_train: np.ndarray, y_train: np.ndarray, 
                                input_dim: int) -> Neuron:
        """Train neuron with fixed step function (no activation training)."""
        print("\n4. Training Fixed Step Neuron")
        neuron = NeuronFactory.create_fixed_step_neuron(input_dim)
        neuron.train_with_strategy(self.weight_strategy, X_train, y_train, self.weight_epochs)
        return neuron
    
    def train_adaptive_step_neuron(self, X_train: np.ndarray, y_train: np.ndarray, 
                                   input_dim: int) -> Neuron:
        """Train neuron with adaptive step function."""
        print("\n5. Training Adaptive Step Neuron")
        neuron = NeuronFactory.create_adaptive_step_neuron(input_dim)
        neuron.train_with_strategy(self.activation_strategy, X_train, y_train, self.activation_epochs)
        neuron.train_with_strategy(self.weight_strategy, X_train, y_train, self.weight_epochs)
        return neuron


# ==========================================
# EVALUATION (Single Responsibility)
# ==========================================

@dataclass
class ModelResult:
    """Data class for storing model evaluation results."""
    name: str
    accuracy: float
    model_info: str = ""


class ModelEvaluator:
    """
    Evaluates and compares different models.
    Follows Single Responsibility Principle.
    """
    
    def __init__(self):
        self.results = []
    
    def evaluate_sklearn_model(self, model: Perceptron, X_test: np.ndarray, 
                               y_test: np.ndarray, name: str) -> ModelResult:
        """Evaluate sklearn model."""
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        result = ModelResult(name=name, accuracy=accuracy)
        self.results.append(result)
        return result
    
    def evaluate_neuron(self, neuron: Neuron, X_test: np.ndarray, 
                       y_test: np.ndarray, name: str) -> ModelResult:
        """Evaluate custom neuron."""
        predictions = neuron.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        model_info = neuron.activation_fn.get_info()
        result = ModelResult(name=name, accuracy=accuracy, model_info=model_info)
        self.results.append(result)
        return result
    
    def print_comparison(self) -> None:
        """Print comprehensive comparison of all models."""
        print("\n" + "=" * 60)
        print("FINAL RESULTS COMPARISON")
        print("=" * 60)
        
        for result in self.results:
            print(f"\n{result.name}")
            print(f"  Accuracy: {result.accuracy:.4f}")
            if result.model_info:
                print(f"  Config: {result.model_info}")
        
        # Find best performer
        best_result = max(self.results, key=lambda r: r.accuracy)
        print("\n" + "-" * 60)
        print(f"Best Performer: {best_result.name} ({best_result.accuracy:.4f})")
        
        # Check if adaptive methods improved
        self._analyze_improvements()
    
    def _analyze_improvements(self) -> None:
        """Analyze if adaptive activation functions showed improvement."""
        fixed_result = next((r for r in self.results if "Fixed" in r.name), None)
        dynamic_result = next((r for r in self.results if "Dynamic" in r.name), None)
        step_result = next((r for r in self.results if "Adaptive Step" in r.name), None)
        
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


# ==========================================
# MAIN ORCHESTRATOR (Facade Pattern)
# ==========================================

class PerceptronExperiment:
    """
    Main orchestrator for running perceptron experiments.
    Follows Facade Pattern to provide simple interface.
    """
    
    def __init__(self, dataset_config: Optional[DatasetConfig] = None):
        self.dataset_config = dataset_config or DatasetConfig()
        self.data_manager = DataManager(self.dataset_config)
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
    
    def run(self) -> None:
        """Execute the complete experiment pipeline."""
        print("=" * 60)
        print("PERCEPTRON EXPERIMENT WITH ADAPTIVE ACTIVATIONS")
        print("=" * 60)
        print()
        
        # 1. Generate and prepare data
        X_train, X_test, y_train, y_test = self.data_manager.generate_dataset()
        input_dim = X_train.shape[1]
        
        # 2. Train all models
        sklearn_model = self.trainer.train_baseline_sklearn(X_train, y_train)
        fixed_neuron = self.trainer.train_fixed_neuron(X_train, y_train, input_dim)
        dynamic_neuron = self.trainer.train_dynamic_neuron(X_train, y_train, input_dim)
        fixed_step_neuron = self.trainer.train_fixed_step_neuron(X_train, y_train, input_dim)
        adaptive_step_neuron = self.trainer.train_adaptive_step_neuron(X_train, y_train, input_dim)
        
        print("\n" + "-" * 60)
        print("EVALUATION ON TEST SET")
        print("-" * 60)
        
        # 3. Evaluate all models
        self.evaluator.evaluate_sklearn_model(sklearn_model, X_test, y_test, 
                                              "Sklearn Perceptron (Baseline)")
        self.evaluator.evaluate_neuron(fixed_neuron, X_test, y_test, 
                                       "Fixed ReLU Neuron")
        self.evaluator.evaluate_neuron(dynamic_neuron, X_test, y_test, 
                                       "Dynamic ReLU Neuron")
        self.evaluator.evaluate_neuron(fixed_step_neuron, X_test, y_test, 
                                       "Fixed Step Neuron")
        self.evaluator.evaluate_neuron(adaptive_step_neuron, X_test, y_test, 
                                       "Adaptive Step Neuron")
        
        # 4. Display comprehensive results
        self.evaluator.print_comparison()


# ==========================================
# ENTRY POINT
# ==========================================

def main():
    """Main entry point for the experiment."""
    # Configure experiment
    config = DatasetConfig(
        n_samples=1000,
        n_features=20,
        n_informative=6,
        n_redundant=8,
        n_repeated=4,
        class_sep=0.5,
        flip_y=0.15
    )
    
    # Run experiment
    experiment = PerceptronExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()

