#!/usr/bin/env python3
"""
MLP Experiment: Comprehensive Comparison of Activation Strategies

Compares multiple MLP architectures on challenging datasets:
1. Baseline: ReLU hidden layers + Softmax output
2. Dynamic (Per-Layer): Learnable ReLU with shared a,b per layer
3. Dynamic (Per-Neuron): Each neuron has its own learnable a,b
4. Two-Phase Training: Train weights first, then tune activations

Datasets tested:
- Fashion-MNIST: Clothing images (harder than MNIST digits)
- MNIST: Handwritten digits (baseline)
- CIFAR-10: Color images (hardest)

Now with multiple seeds for statistical significance!
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from data_utils import DatasetConfig, DataManager
from mlp_trainer import MLPExperiment, ExperimentResult


@dataclass
class AggregatedResult:
    """Aggregated results from multiple seeds."""
    model_name: str
    train_accs: List[float] = field(default_factory=list)
    test_accs: List[float] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)
    
    @property
    def train_mean(self) -> float:
        return np.mean(self.train_accs)
    
    @property
    def train_std(self) -> float:
        return np.std(self.train_accs)
    
    @property
    def test_mean(self) -> float:
        return np.mean(self.test_accs)
    
    @property
    def test_std(self) -> float:
        return np.std(self.test_accs)
    
    @property
    def time_mean(self) -> float:
        return np.mean(self.times)
    
    @property
    def epochs_mean(self) -> float:
        return np.mean(self.epochs)


def load_dataset(dataset_type: str, test_size: float = 0.2, random_state: int = 42):
    """Load and prepare dataset."""
    print(f"Loading {dataset_type} dataset...")
    config = DatasetConfig(
        dataset_type=dataset_type,
        test_size=test_size,
        random_state=random_state,
    )
    dm = DataManager(config)
    X_train, X_test, y_train, y_test = dm.generate_dataset()
    
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Input features: {X_train.shape[1]}")
    print(f"  Classes: {len(np.unique(y_train))}")
    
    return X_train, X_test, y_train, y_test


def print_results(agg_results: Dict[str, AggregatedResult], dataset_name: str, n_seeds: int) -> None:
    """Print comparison of aggregated experiment results."""
    print("\n" + "=" * 100)
    print(f"EXPERIMENT RESULTS: {dataset_name.upper()} ({n_seeds} seeds)")
    print("=" * 100)
    
    headers = ["Model", "Train Acc", "Test Acc", "Time (s)", "Epochs"]
    row_format = "{:<45} {:>18} {:>18} {:>10} {:>8}"
    
    print(row_format.format(*headers))
    print("-" * 100)
    
    for name, r in agg_results.items():
        print(row_format.format(
            name[:45],
            f"{r.train_mean:.4f} ± {r.train_std:.4f}",
            f"{r.test_mean:.4f} ± {r.test_std:.4f}",
            f"{r.time_mean:.2f}",
            f"{r.epochs_mean:.1f}",
        ))
    
    print("-" * 100)
    
    # Highlight winner
    best_name = max(agg_results.keys(), key=lambda x: agg_results[x].test_mean)
    best = agg_results[best_name]
    baseline = agg_results.get("Baseline (ReLU)", list(agg_results.values())[0])
    improvement = best.test_mean - baseline.test_mean
    
    print(f"\n🏆 Best: {best_name}")
    print(f"   Test Accuracy: {best.test_mean:.4f} ± {best.test_std:.4f}")
    print(f"   Improvement over baseline: {improvement:+.4f} ({improvement*100:+.2f}%)")


def save_results(all_results: dict, filepath: str, n_seeds: int) -> None:
    """Save aggregated results to CSV file."""
    rows = []
    for dataset_name, agg_results in all_results.items():
        for model_name, r in agg_results.items():
            rows.append({
                "dataset": dataset_name,
                "model_name": model_name,
                "n_seeds": n_seeds,
                "train_acc_mean": r.train_mean,
                "train_acc_std": r.train_std,
                "test_acc_mean": r.test_mean,
                "test_acc_std": r.test_std,
                "time_mean_seconds": r.time_mean,
                "epochs_mean": r.epochs_mean,
                "train_accs": str(r.train_accs),
                "test_accs": str(r.test_accs),
                "timestamp": datetime.now().isoformat(),
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"\nResults saved to: {filepath}")


def run_single_seed(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    hidden_dims: List[int],
    epochs: int,
    activation_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    verbose: bool = False,
) -> Dict[str, ExperimentResult]:
    """Run all experiment variants for a single seed."""
    
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    
    # Create experiment runner
    experiment = MLPExperiment(
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        activation_epochs=activation_epochs,
        seed=seed,
        verbose=verbose,
    )
    
    results = {}
    
    # 1. Baseline MLP (ReLU)
    if verbose:
        print("\n" + "🔷" * 30)
    baseline_result = experiment.run_baseline(
        X_train, y_train, X_test, y_test,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    results["Baseline (ReLU)"] = baseline_result
    
    # # 2. Dynamic MLP (Per-Layer, Joint Training)
    # if verbose:
    #     print("\n" + "🔷" * 30)
    # dynamic_result = experiment.run_dynamic(
    #     X_train, y_train, X_test, y_test,
    #     input_dim=input_dim,
    #     output_dim=output_dim,
    # )
    # results["Dynamic (Per-Layer)"] = dynamic_result
    
    # # 3. Dynamic MLP (Per-Neuron, Joint Training)
    # if verbose:
    #     print("\n" + "🔷" * 30)
    # per_neuron_result = experiment.run_dynamic_per_neuron(
    #     X_train, y_train, X_test, y_test,
    #     input_dim=input_dim,
    #     output_dim=output_dim,
    # )
    # results["Dynamic (Per-Neuron)"] = per_neuron_result
    
    # 4. Two-Phase Training (Per-Neuron)
    if verbose:
        print("\n" + "🔷" * 30)
    two_phase_result = experiment.run_dynamic_two_phase(
        X_train, y_train, X_test, y_test,
        input_dim=input_dim,
        output_dim=output_dim,
        per_neuron=True,
    )
    results["Two-Phase (Per-Neuron)"] = two_phase_result
    
    return results


def run_experiment_suite(
    dataset_type: str,
    hidden_dims: List[int],
    epochs: int,
    activation_epochs: int,
    batch_size: int,
    learning_rate: float,
    seeds: List[int],
) -> Dict[str, AggregatedResult]:
    """Run experiments with multiple seeds and aggregate results."""
    
    # Load data once (use first seed for train/test split)
    X_train, X_test, y_train, y_test = load_dataset(dataset_type, random_state=seeds[0])
    
    # Initialize aggregated results
    model_names = [
        "Baseline (ReLU)",
        "Dynamic (Per-Layer)",
        "Dynamic (Per-Neuron)",
        "Two-Phase (Per-Neuron)",
    ]
    agg_results = {name: AggregatedResult(name) for name in model_names}
    
    # Run experiments for each seed
    for i, seed in enumerate(seeds):
        print(f"\n  Seed {i+1}/{len(seeds)} (seed={seed})...")
        
        # Run all models with this seed
        seed_results = run_single_seed(
            X_train, X_test, y_train, y_test,
            hidden_dims=hidden_dims,
            epochs=epochs,
            activation_epochs=activation_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            verbose=False,
        )
        
        # Aggregate results
        for name, result in seed_results.items():
            agg_results[name].train_accs.append(result.train_accuracy)
            agg_results[name].test_accs.append(result.test_accuracy)
            agg_results[name].times.append(result.training_time)
            agg_results[name].epochs.append(result.epochs_trained)
        
        # Print progress
        baseline_acc = seed_results["Baseline (ReLU)"].test_accuracy
        best_acc = max(r.test_accuracy for r in seed_results.values())
        print(f"    Baseline: {baseline_acc:.4f}, Best: {best_acc:.4f}")
    
    return agg_results


def main():
    """Run the comprehensive MLP comparison experiment."""
    print("=" * 100)
    print("MLP EXPERIMENT: Baseline vs Dynamic Activations (Multi-Seed)")
    print("Comparing: ReLU | Dynamic Per-Layer | Dynamic Per-Neuron | Two-Phase")
    print("=" * 100)
    
    # Configuration
    N_SEEDS = 20
    SEEDS = [42, 123, 456, 789, 1001, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1010, 2020, 3030, 4040, 5050, 6060]
    HIDDEN_DIMS = [256, 128]  # Two hidden layers
    EPOCHS = 30
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01
    ACTIVATION_EPOCHS = 30
    
    print(f"\nConfiguration:")
    print(f"  Number of seeds: {N_SEEDS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Hidden layers: {HIDDEN_DIMS}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Activation training epochs: {ACTIVATION_EPOCHS}")
    print(f"  Activation training epochs: {ACTIVATION_EPOCHS}")
    
    all_results = {}
    
    # Test on Fashion-MNIST (primary test)
    print("\n" + "🌟" * 40)
    print("DATASET: FASHION-MNIST")
    print("🌟" * 40)
    
    fashion_results = run_experiment_suite(
        dataset_type='fashion_mnist',
        hidden_dims=HIDDEN_DIMS,
        epochs=EPOCHS,
        activation_epochs=ACTIVATION_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seeds=SEEDS,
    )
    all_results['fashion_mnist'] = fashion_results
    print_results(fashion_results, "Fashion-MNIST", N_SEEDS)
    
    # Test on MNIST (digits - easier baseline)
    print("\n" + "🌟" * 40)
    print("DATASET: MNIST (Digits)")
    print("🌟" * 40)
    
    try:
        mnist_results = run_experiment_suite(
            dataset_type='mnist',
            hidden_dims=HIDDEN_DIMS,
            epochs=EPOCHS,
            activation_epochs=ACTIVATION_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            seeds=SEEDS,
        )
        all_results['mnist'] = mnist_results
        print_results(mnist_results, "MNIST", N_SEEDS)
    except Exception as e:
        print(f"Skipping MNIST: {e}")
    
    # Final summary
    print("\n" + "=" * 100)
    print("FINAL SUMMARY (Mean ± Std over {} seeds)".format(N_SEEDS))
    print("=" * 100)
    
    for dataset_name, agg_results in all_results.items():
        print(f"\n📊 {dataset_name.upper()}:")
        
        baseline = agg_results.get("Baseline (ReLU)")
        best_name = max(agg_results.keys(), key=lambda x: agg_results[x].test_mean)
        best = agg_results[best_name]
        improvement = best.test_mean - baseline.test_mean
        
        print(f"   Baseline: {baseline.test_mean:.4f} ± {baseline.test_std:.4f}")
        print(f"   Best ({best_name}): {best.test_mean:.4f} ± {best.test_std:.4f}")
        print(f"   Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    # Save results
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(output_dir, "mlp_experiment_results.csv")
    save_results(all_results, output_path, N_SEEDS)
    
    return all_results


if __name__ == "__main__":
    main()
