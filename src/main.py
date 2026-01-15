"""
Perceptron Experiment - Main Entry Point

This module orchestrates the complete perceptron experiment with various activation functions.
Runs multiple seeds to analyze variance and convergence.
"""

from typing import Optional, Dict, List
from collections import defaultdict

import numpy as np

from data_utils import DatasetConfig, DataManager
from trainer import ModelTrainer, ModelEvaluator


class PerceptronExperiment:
    """
    Main orchestrator for running perceptron experiments.
    Follows Facade Pattern to provide simple interface.
    """
    
    def __init__(self, dataset_config: Optional[DatasetConfig] = None, dataset_name: str = "unknown", seed: int = 42):
        self.dataset_config = dataset_config or DatasetConfig()
        self.dataset_name = dataset_name
        self.seed = seed
        self.data_manager = DataManager(self.dataset_config)
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator(dataset_name=dataset_name, seed=seed)
    
    def run(self, verbose: bool = True) -> Dict[str, float]:
        """Execute the complete experiment pipeline."""
        if verbose:
            print("=" * 60)
            print("PERCEPTRON EXPERIMENT WITH ADAPTIVE ACTIVATIONS")
            print("=" * 60)
            print()
        
        # Set numpy seed for reproducibility
        np.random.seed(self.seed)
        
        # 1. Generate and prepare data
        X_train, X_test, y_train, y_test = self.data_manager.generate_dataset()
        input_dim = X_train.shape[1]
        
        # 2. Train all models
        sklearn_model = self.trainer.train_baseline_sklearn(X_train, y_train)
        fixed_neuron = self.trainer.train_fixed_neuron(X_train, y_train, input_dim)
        dynamic_neuron = self.trainer.train_dynamic_neuron(X_train, y_train, input_dim)
        
        if verbose:
            print("\n" + "-" * 60)
            print("EVALUATION ON TEST SET")
            print("-" * 60)
        
        # 3. Evaluate all models
        results = {}
        
        result = self.evaluator.evaluate_sklearn_model(
            sklearn_model, X_test, y_test, "Sklearn Perceptron (Baseline)"
        )
        results["Sklearn Perceptron (Baseline)"] = result.accuracy
        
        result = self.evaluator.evaluate_neuron(fixed_neuron, X_test, y_test, "Fixed ReLU Neuron")
        results["Fixed ReLU Neuron"] = result.accuracy
        
        result = self.evaluator.evaluate_neuron(dynamic_neuron, X_test, y_test, "Dynamic ReLU Neuron")
        results["Dynamic ReLU Neuron"] = result.accuracy
        
        if verbose:
            # 4. Display comprehensive results
            self.evaluator.print_comparison()
        
        return results


def compute_statistics(accuracies: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of accuracy values."""
    arr = np.array(accuracies)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
    }


def print_statistics_table(dataset_name: str, model_results: Dict[str, List[float]]) -> None:
    """Print a formatted statistics table for all models."""
    print("\n" + "=" * 80)
    print(f"STATISTICS SUMMARY FOR: {dataset_name.upper()}")
    print("=" * 80)
    
    # Header
    print(f"\n{'Model':<35} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Range':>8}")
    print("-" * 80)
    
    for model_name, accuracies in model_results.items():
        stats = compute_statistics(accuracies)
        print(f"{model_name:<35} {stats['mean']:>8.4f} {stats['std']:>8.4f} "
              f"{stats['min']:>8.4f} {stats['max']:>8.4f} {stats['range']:>8.4f}")
    
    # Print individual runs
    print("\n" + "-" * 80)
    print("Individual Run Results:")
    print("-" * 80)
    
    n_seeds = len(next(iter(model_results.values())))
    print(f"\n{'Seed':<6}", end="")
    for model_name in model_results.keys():
        short_name = model_name[:20] + "..." if len(model_name) > 20 else model_name
        print(f"{short_name:>25}", end="")
    print()
    print("-" * 80)
    
    for i in range(n_seeds):
        print(f"{i:<6}", end="")
        for model_name in model_results.keys():
            print(f"{model_results[model_name][i]:>25.4f}", end="")
        print()


def main():
    """Main entry point for the experiment."""
    print("=" * 80)
    print("MULTI-SEED EXPERIMENT: Analyzing Model Variance")
    print("=" * 80)
    print()
    print("Available datasets:")
    print("  1. breast_cancer - Breast Cancer Wisconsin (569 samples, 30 features)")
    print("  2. titanic - Passenger survival (with engineered categorical features)")
    print("  3. heart_disease - UCI Heart Disease (13 clinical measurements)")
    print()
    
    # Configuration
    NUM_SEEDS = 10
    BASE_SEED = 42
    seeds = [BASE_SEED + i for i in range(NUM_SEEDS)]
    
    print(f"Running {NUM_SEEDS} seeds: {seeds}")
    print()
    
    datasets = ['breast_cancer', 'titanic', 'heart_disease']
    
    # Store all results: {dataset: {model: [accuracies]}}
    all_dataset_results: Dict[str, Dict[str, List[float]]] = {}
    all_evaluators = []
    
    for dataset_name in datasets:
        print("\n" + "=" * 80)
        print(f"RUNNING EXPERIMENTS ON: {dataset_name.upper()}")
        print("=" * 80)
        
        model_results: Dict[str, List[float]] = defaultdict(list)
        
        for seed_idx, seed in enumerate(seeds):
            print(f"\n--- Seed {seed_idx + 1}/{NUM_SEEDS} (seed={seed}) ---")
            
            config = DatasetConfig(dataset_type=dataset_name, random_state=seed)
            experiment = PerceptronExperiment(config, dataset_name=dataset_name, seed=seed)
            
            # Run with reduced verbosity after first run
            verbose = (seed_idx == 0)
            results = experiment.run(verbose=verbose)
            
            # Collect results
            for model_name, accuracy in results.items():
                model_results[model_name].append(accuracy)
            
            all_evaluators.append(experiment.evaluator)
            
            if not verbose:
                # Print brief summary for non-verbose runs
                summary = " | ".join([f"{name[:15]}:{acc:.4f}" for name, acc in results.items()])
                print(f"  Results: {summary}")
        
        all_dataset_results[dataset_name] = dict(model_results)
        
        # Print statistics for this dataset
        print_statistics_table(dataset_name, model_results)
    
    # Save all results to CSV
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    if all_evaluators:
        # Collect all results
        all_results = []
        for evaluator in all_evaluators:
            all_results.extend(evaluator.results)
        
        # Create evaluator to use save method
        final_evaluator = ModelEvaluator()
        final_evaluator.results = all_results
        
        # Save to a single file (overwrite mode)
        final_evaluator.save_to_csv("experiment_results.csv", append=False)
        
        print(f"\n✓ All results saved to: experiment_results.csv")
        print(f"  Total experiments: {len(all_results)}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)
    
    for dataset_name, model_results in all_dataset_results.items():
        print(f"\n{dataset_name.upper()}:")
        for model_name, accuracies in model_results.items():
            stats = compute_statistics(accuracies)
            convergence = "STABLE" if stats['std'] < 0.02 else "VARIABLE" if stats['std'] < 0.05 else "DIVERGENT"
            print(f"  {model_name}: {convergence} (std={stats['std']:.4f}, range={stats['range']:.4f})")
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
    