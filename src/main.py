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
    
    def run(self, verbose: bool = False) -> Dict[str, float]:
        """Execute the complete experiment pipeline."""
        
        # Set numpy seed for reproducibility
        np.random.seed(self.seed)
        
        # 1. Generate and prepare data
        X_train, X_test, y_train, y_test = self.data_manager.generate_dataset()
        input_dim = X_train.shape[1]
        
        # 2. Train all models
        sklearn_model = self.trainer.train_baseline_sklearn(X_train, y_train)
        fixed_neuron = self.trainer.train_fixed_neuron(X_train, y_train, input_dim)
        dynamic_neuron = self.trainer.train_dynamic_neuron(X_train, y_train, input_dim)
        sigmoid_neuron = self.trainer.train_sigmoid_neuron(X_train, y_train, input_dim)
        
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
        
        result = self.evaluator.evaluate_neuron(sigmoid_neuron, X_test, y_test, "Fixed Sigmoid Neuron")
        results["Fixed Sigmoid Neuron"] = result.accuracy
        
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
        "n_runs": len(arr),
    }


def save_statistics_to_csv(
    all_dataset_results: Dict[str, Dict[str, List[float]]], 
    filepath: str = "experiment_statistics.csv"
) -> None:
    """Save statistics summary to CSV file."""
    import pandas as pd
    
    rows = []
    for dataset_name, model_results in all_dataset_results.items():
        for model_name, accuracies in model_results.items():
            stats = compute_statistics(accuracies)
            # Classify convergence
            convergence = "STABLE" if stats['std'] < 0.02 else "VARIABLE" if stats['std'] < 0.05 else "DIVERGENT"
            rows.append({
                "dataset": dataset_name,
                "model_name": model_name,
                "mean": stats["mean"],
                "std": stats["std"],
                "convergence": convergence,
                "min": stats["min"],
                "max": stats["max"],
                "range": stats["range"],
                "n_runs": stats["n_runs"],
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    return


def print_statistics_table(dataset_name: str, model_results: Dict[str, List[float]]) -> None:
    """Print a formatted statistics table for all models."""
    return


def main():
    """Main entry point for the experiment."""
    # Configuration
    NUM_SEEDS = 100
    BASE_SEED = 42
    seeds = [BASE_SEED + i for i in range(NUM_SEEDS)]
    
    datasets = ['breast_cancer', 'titanic', 'heart_disease']
    
    # Store all results: {dataset: {model: [accuracies]}}
    all_dataset_results: Dict[str, Dict[str, List[float]]] = {}
    all_evaluators = []
    
    for dataset_name in datasets:
        model_results: Dict[str, List[float]] = defaultdict(list)
        
        for seed_idx, seed in enumerate(seeds):
            config = DatasetConfig(dataset_type=dataset_name, random_state=seed)
            experiment = PerceptronExperiment(config, dataset_name=dataset_name, seed=seed)
            
            results = experiment.run(verbose=False)
            
            # Collect results
            for model_name, accuracy in results.items():
                model_results[model_name].append(accuracy)
            
            all_evaluators.append(experiment.evaluator)
        
        all_dataset_results[dataset_name] = dict(model_results)
        
        # Statistics computed later for CSV only
    
    # Save all results to CSV
    if all_evaluators:
        # Collect all results
        all_results = []
        for evaluator in all_evaluators:
            all_results.extend(evaluator.results)
        
        # Create evaluator to use save method
        final_evaluator = ModelEvaluator()
        final_evaluator.results = all_results
        
        # Save individual runs to CSV
        final_evaluator.save_to_csv("experiment_results.csv", append=False)
        
        # Save statistics summary to CSV
        save_statistics_to_csv(all_dataset_results, "experiment_statistics.csv")
    
    # No console output; results are saved to CSV files only.


if __name__ == "__main__":
    main()
    