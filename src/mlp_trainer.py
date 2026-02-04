"""
MLP Training and Evaluation Module.

Provides training utilities for MLP models with support for:
- Mini-batch gradient descent
- Cross-entropy loss for multi-class classification
- Learning rate scheduling
- Early stopping
- Progress tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
import time

import numpy as np
from sklearn.metrics import accuracy_score

from mlp import MLP


@dataclass
class TrainingHistory:
    """Records training metrics over epochs."""
    train_losses: List[float]
    train_accuracies: List[float]
    val_losses: List[float]
    val_accuracies: List[float]
    epochs_trained: int
    training_time: float
    
    def best_val_accuracy(self) -> float:
        """Return the best validation accuracy achieved."""
        return max(self.val_accuracies) if self.val_accuracies else 0.0


class CrossEntropyLoss:
    """Cross-entropy loss for multi-class classification with softmax."""
    
    @staticmethod
    def compute(predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1e-15) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            predictions: Softmax outputs, shape (batch_size, num_classes)
            targets: One-hot encoded targets, shape (batch_size, num_classes)
            epsilon: Small value to avoid log(0)
            
        Returns:
            Mean cross-entropy loss
        """
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.mean(np.sum(targets * np.log(predictions), axis=1))
    
    @staticmethod
    def gradient(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute gradient of cross-entropy loss w.r.t. softmax output.
        
        For softmax + cross-entropy, the gradient simplifies to: pred - target
        """
        return predictions - targets


class MLPTrainer:
    """
    Trainer for MLP models.
    
    Handles training loop, validation, and early stopping.
    """
    
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.01,
        early_stopping_patience: int = 10,
        verbose: bool = True,
        print_every: int = 10,
    ):
        """
        Initialize trainer.
        
        Args:
            epochs: Maximum number of training epochs
            batch_size: Mini-batch size
            learning_rate: Learning rate (overrides model's if set)
            early_stopping_patience: Stop if val loss doesn't improve for this many epochs
            verbose: Print progress during training
            print_every: Print every N epochs
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.print_every = print_every
    
    def train(
        self,
        model: MLP,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TrainingHistory:
        """
        Train the MLP model.
        
        Args:
            model: MLP model to train
            X_train: Training features
            y_train: Training labels (integer class labels)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            TrainingHistory with metrics over training
        """
        start_time = time.time()
        
        # Convert labels to one-hot encoding
        num_classes = model.config.output_dim
        y_train_onehot = self._to_onehot(y_train, num_classes)
        y_val_onehot = self._to_onehot(y_val, num_classes) if y_val is not None else None
        
        # Override learning rate if specified
        if self.learning_rate is not None:
            model.config.learning_rate = self.learning_rate
        
        # Training history
        history = TrainingHistory(
            train_losses=[],
            train_accuracies=[],
            val_losses=[],
            val_accuracies=[],
            epochs_trained=0,
            training_time=0.0,
        )
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        n_samples = len(X_train)
        
        for epoch in range(self.epochs):
            model.train()
            
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_onehot[indices]
            
            # Mini-batch training
            epoch_losses = []
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward pass
                predictions = model.forward(X_batch)
                
                # Compute loss
                loss = CrossEntropyLoss.compute(predictions, y_batch)
                epoch_losses.append(loss)
                
                # Backward pass
                grad = CrossEntropyLoss.gradient(predictions, y_batch)
                model.backward(grad)
            
            # Epoch metrics
            train_loss = np.mean(epoch_losses)
            train_acc = self._compute_accuracy(model, X_train, y_train)
            
            history.train_losses.append(train_loss)
            history.train_accuracies.append(train_acc)
            
            # Validation
            if X_val is not None and y_val is not None:
                model.eval()
                val_predictions = model.forward(X_val)
                val_loss = CrossEntropyLoss.compute(val_predictions, y_val_onehot)
                val_acc = self._compute_accuracy(model, X_val, y_val)
                
                history.val_losses.append(val_loss)
                history.val_accuracies.append(val_acc)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.early_stopping_patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break
            
            # Print progress
            if self.verbose and (epoch + 1) % self.print_every == 0:
                msg = f"Epoch {epoch + 1}/{self.epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f}"
                if X_val is not None:
                    msg += f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
                print(msg)
            
            history.epochs_trained = epoch + 1
        
        history.training_time = time.time() - start_time
        return history
    
    def train_activation_params(
        self,
        model: MLP,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TrainingHistory:
        """
        Train only the learnable activation parameters (phase 2) with frozen weights.
        
        Uses mini-batch SGD to tune activation parameters while keeping weights fixed.
        
        Args:
            model: MLP model with dynamic activations
            X_train: Training features
            y_train: Training labels
            epochs: Number of epochs for activation training
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            TrainingHistory for activation training phase
        """
        start_time = time.time()
        
        num_classes = model.config.output_dim
        y_train_onehot = self._to_onehot(y_train, num_classes)
        y_val_onehot = self._to_onehot(y_val, num_classes) if y_val is not None else None
        
        history = TrainingHistory(
            train_losses=[],
            train_accuracies=[],
            val_losses=[],
            val_accuracies=[],
            epochs_trained=0,
            training_time=0.0,
        )
        
        n_samples = len(X_train)
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            model.train()
            
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_onehot[indices]
            
            # Mini-batch training with frozen weights
            epoch_losses = []
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward pass
                predictions = model.forward(X_batch)
                
                # Compute loss
                loss = CrossEntropyLoss.compute(predictions, y_batch)
                epoch_losses.append(loss)
                
                # Backward pass: update ONLY activation params (freeze weights)
                grad = CrossEntropyLoss.gradient(predictions, y_batch)
                model.backward(grad, update_activation=True, update_weights=False)
            
            # Epoch metrics
            train_loss = np.mean(epoch_losses)
            train_acc = self._compute_accuracy(model, X_train, y_train)
            
            history.train_losses.append(train_loss)
            history.train_accuracies.append(train_acc)
            
            # Validation
            if X_val is not None and y_val is not None:
                model.eval()
                val_predictions = model.forward(X_val)
                val_loss = CrossEntropyLoss.compute(val_predictions, y_val_onehot)
                val_acc = self._compute_accuracy(model, X_val, y_val)
                
                history.val_losses.append(val_loss)
                history.val_accuracies.append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.early_stopping_patience:
                        if self.verbose:
                            print(f"  [Activation] Early stopping at epoch {epoch + 1}")
                        break
            
            # Print progress
            if self.verbose and (epoch + 1) % self.print_every == 0:
                msg = f"  [Activation] Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f}"
                if X_val is not None:
                    msg += f" - Val Acc: {val_acc:.4f}"
                print(msg)
            
            history.epochs_trained = epoch + 1
        
        history.training_time = time.time() - start_time
        return history
    
    def _to_onehot(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        """Convert integer labels to one-hot encoding."""
        onehot = np.zeros((len(y), num_classes))
        onehot[np.arange(len(y)), y] = 1
        return onehot
    
    def _compute_accuracy(self, model: MLP, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        predictions = model.predict(X)
        return accuracy_score(y, predictions)


@dataclass
class ExperimentResult:
    """Results from an experiment run."""
    model_name: str
    train_accuracy: float
    test_accuracy: float
    training_time: float
    epochs_trained: int
    model_params: int
    activation_info: str = ""


class MLPExperiment:
    """
    Run MLP experiments comparing different architectures.
    """
    
    def __init__(
        self,
        hidden_dims: List[int] = [256, 128],
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        activation_epochs: int = 50,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.activation_epochs = activation_epochs
        self.seed = seed
        self.verbose = verbose
    
    def run_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        input_dim: int,
        output_dim: int,
    ) -> ExperimentResult:
        """Run baseline MLP with fixed ReLU activations."""
        from mlp import create_baseline_mlp
        
        np.random.seed(self.seed)
        
        model = create_baseline_mlp(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            learning_rate=self.learning_rate,
        )
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("BASELINE MLP (ReLU)")
            print("=" * 60)
            print(model.summary())
        
        trainer = MLPTrainer(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
        )
        
        history = trainer.train(model, X_train, y_train, X_test, y_test)
        
        train_acc = trainer._compute_accuracy(model, X_train, y_train)
        test_acc = trainer._compute_accuracy(model, X_test, y_test)
        
        return ExperimentResult(
            model_name="Baseline MLP (ReLU)",
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            training_time=history.training_time,
            epochs_trained=history.epochs_trained,
            model_params=model.num_params,
        )
    
    def run_dynamic(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        input_dim: int,
        output_dim: int,
    ) -> ExperimentResult:
        """Run MLP with dynamic (learnable) ReLU activations."""
        from mlp import create_dynamic_mlp
        
        np.random.seed(self.seed)
        
        model = create_dynamic_mlp(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            learning_rate=self.learning_rate,
            activation_lr=self.learning_rate,
        )
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("DYNAMIC MLP (Learnable ReLU)")
            print("=" * 60)
            print(model.summary())
        
        trainer = MLPTrainer(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
        )
        
        # Joint training: weights and activation parameters are trained together
        if self.verbose:
            print("\nTraining weights and activation parameters jointly...")
        history = trainer.train(model, X_train, y_train, X_test, y_test)
        
        train_acc = trainer._compute_accuracy(model, X_train, y_train)
        test_acc = trainer._compute_accuracy(model, X_test, y_test)
        
        # Collect activation info
        activation_info = []
        for layer in model.layers:
            if hasattr(layer, 'activation') and layer.activation.is_learnable:
                activation_info.append(layer.activation.info)
        
        return ExperimentResult(
            model_name="Dynamic MLP (Learnable ReLU)",
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            training_time=history.training_time,
            epochs_trained=history.epochs_trained,
            model_params=model.num_params,
            activation_info="; ".join(activation_info),
        )

    def run_dynamic_per_neuron(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        input_dim: int,
        output_dim: int,
    ) -> ExperimentResult:
        """Run MLP with per-neuron dynamic ReLU activations (each neuron has its own a, b)."""
        from mlp import MLPConfig, MLP
        
        np.random.seed(self.seed)
        
        config = MLPConfig(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            hidden_activation="dynamic_relu",
            output_activation="softmax",
            learning_rate=self.learning_rate,
            activation_lr=self.learning_rate,
            per_neuron_activation=True,  # Enable per-neuron params
        )
        model = MLP(config)
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("DYNAMIC MLP (Per-Neuron Learnable ReLU)")
            print("=" * 60)
            print(model.summary())
        
        trainer = MLPTrainer(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
        )
        
        if self.verbose:
            print("\nTraining weights and per-neuron activations jointly...")
        history = trainer.train(model, X_train, y_train, X_test, y_test)
        
        train_acc = trainer._compute_accuracy(model, X_train, y_train)
        test_acc = trainer._compute_accuracy(model, X_test, y_test)
        
        # Collect activation info
        activation_info = []
        for layer in model.layers:
            if hasattr(layer, 'activation') and layer.activation.is_learnable:
                activation_info.append(layer.activation.info)
        
        return ExperimentResult(
            model_name="Dynamic MLP (Per-Neuron)",
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            training_time=history.training_time,
            epochs_trained=history.epochs_trained,
            model_params=model.num_params,
            activation_info="; ".join(activation_info),
        )

    def run_dynamic_two_phase(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        input_dim: int,
        output_dim: int,
        per_neuron: bool = True,
    ) -> ExperimentResult:
        """
        Run MLP with two-phase training:
        Phase 1: Train weights only (freeze activations)
        Phase 2: Freeze weights, tune activation parameters
        """
        from mlp import MLPConfig, MLP
        
        np.random.seed(self.seed)
        
        config = MLPConfig(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            hidden_activation="dynamic_relu",
            output_activation="softmax",
            learning_rate=self.learning_rate,
            activation_lr=self.learning_rate,
            per_neuron_activation=per_neuron,
        )
        model = MLP(config)
        
        mode_str = "Per-Neuron" if per_neuron else "Per-Layer"
        
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"DYNAMIC MLP Two-Phase ({mode_str})")
            print("=" * 60)
            print(model.summary())
        
        trainer = MLPTrainer(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
        )
        
        # Phase 1: Train weights only (don't update activation params)
        if self.verbose:
            print("\n--- Phase 1: Training weights (activations frozen) ---")
        
        # Temporarily disable activation updates by using update_activation=False
        start_time = time.time()
        num_classes = model.config.output_dim
        y_train_onehot = trainer._to_onehot(y_train, num_classes)
        y_test_onehot = trainer._to_onehot(y_test, num_classes) if y_test is not None else None
        
        n_samples = len(X_train)
        for epoch in range(self.epochs):
            model.train()
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_onehot[indices]
            
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                predictions = model.forward(X_batch)
                grad = CrossEntropyLoss.gradient(predictions, y_batch)
                # Only update weights, NOT activations
                model.backward(grad, update_activation=False, update_weights=True)
            
            if self.verbose and (epoch + 1) % trainer.print_every == 0:
                train_acc = trainer._compute_accuracy(model, X_train, y_train)
                test_acc = trainer._compute_accuracy(model, X_test, y_test)
                print(f"Epoch {epoch + 1}/{self.epochs} - Train Acc: {train_acc:.4f} - Val Acc: {test_acc:.4f}")
        
        phase1_time = time.time() - start_time
        phase1_acc = trainer._compute_accuracy(model, X_test, y_test)
        
        if self.verbose:
            print(f"\nPhase 1 complete. Test accuracy: {phase1_acc:.4f}")
        
        # Phase 2: Freeze weights, tune activations
        if self.verbose:
            print(f"\n--- Phase 2: Tuning activations (weights frozen) ---")
        
        activation_history = trainer.train_activation_params(
            model, X_train, y_train, 
            epochs=self.activation_epochs,
            X_val=X_test, y_val=y_test
        )
        
        total_time = phase1_time + activation_history.training_time
        
        train_acc = trainer._compute_accuracy(model, X_train, y_train)
        test_acc = trainer._compute_accuracy(model, X_test, y_test)
        
        if self.verbose:
            print(f"\nPhase 2 complete. Test accuracy: {test_acc:.4f} (was {phase1_acc:.4f})")
        
        # Collect activation info
        activation_info = []
        for layer in model.layers:
            if hasattr(layer, 'activation') and layer.activation.is_learnable:
                activation_info.append(layer.activation.info)
        
        return ExperimentResult(
            model_name=f"Dynamic MLP Two-Phase ({mode_str})",
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            training_time=total_time,
            epochs_trained=self.epochs + activation_history.epochs_trained,
            model_params=model.num_params,
            activation_info="; ".join(activation_info),
        )

    def run_activation_finetuning(
        self,
        baseline_model: 'MLP',
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        per_neuron: bool = True,
    ) -> ExperimentResult:
        """
        Take a trained baseline model, copy it with dynamic activations,
        and train only the activation parameters.
        
        This avoids training two separate models - we reuse the baseline weights.
        
        Args:
            baseline_model: A trained baseline MLP model
            X_train, y_train: Training data
            X_test, y_test: Test data
            per_neuron: If True, each neuron gets its own a,b parameters
            
        Returns:
            ExperimentResult for the activation-finetuned model
        """
        # Copy the baseline model with dynamic activations
        dynamic_model = baseline_model.copy_with_dynamic_activations(
            per_neuron=per_neuron,
            activation_lr=self.learning_rate,
        )
        
        mode_str = "Per-Neuron" if per_neuron else "Per-Layer"
        
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"ACTIVATION FINETUNING ({mode_str})")
            print("=" * 60)
            print("Starting from trained baseline weights...")
            print(dynamic_model.summary())
        
        trainer = MLPTrainer(
            epochs=self.activation_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
        )
        
        # Check accuracy before activation training
        pre_train_acc = trainer._compute_accuracy(dynamic_model, X_test, y_test)
        if self.verbose:
            print(f"\nTest accuracy before activation training: {pre_train_acc:.4f}")
            print(f"\n--- Training activation parameters (weights frozen) ---")
        
        # Train only activation parameters
        activation_history = trainer.train_activation_params(
            dynamic_model, X_train, y_train,
            epochs=self.activation_epochs,
            X_val=X_test, y_val=y_test
        )
        
        train_acc = trainer._compute_accuracy(dynamic_model, X_train, y_train)
        test_acc = trainer._compute_accuracy(dynamic_model, X_test, y_test)
        
        if self.verbose:
            print(f"\nActivation training complete.")
            print(f"Test accuracy: {test_acc:.4f} (was {pre_train_acc:.4f}, improvement: {test_acc - pre_train_acc:+.4f})")
        
        # Collect activation info
        activation_info = []
        for layer in dynamic_model.layers:
            if hasattr(layer, 'activation') and layer.activation.is_learnable:
                activation_info.append(layer.activation.info)
        
        return ExperimentResult(
            model_name=f"Activation Finetuning ({mode_str})",
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            training_time=activation_history.training_time,
            epochs_trained=activation_history.epochs_trained,
            model_params=dynamic_model.num_params,
            activation_info="; ".join(activation_info),
        )
