"""
Perceptron Experiment - Main Entry Point

This module orchestrates the complete perceptron experiment with various activation functions.
Follows SOLID principles and clean architecture patterns.
"""

from typing import Optional

from data_utils import DatasetConfig, DataManager
from trainer import ModelTrainer, ModelEvaluator


class PerceptronExperiment:
    """
    Main orchestrator for running perceptron experiments.
    Follows Facade Pattern to provide simple interface.
    """
    
    def __init__(self, dataset_config: Optional[DatasetConfig] = None, dataset_name: str = "unknown"):
        self.dataset_config = dataset_config or DatasetConfig()
        self.dataset_name = dataset_name
        self.data_manager = DataManager(self.dataset_config)
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator(dataset_name=dataset_name)
    
    def run(self) -> None:
        """Execute the complete experiment pipeline."""
        print("=" * 60)
        print("PERCEPTRON EXPERIMENT WITH ADAPTIVE + SOFT ACTIVATIONS")
        print("=" * 60)
        print()
        
        # 1. Generate and prepare data
        X_train, X_test, y_train, y_test = self.data_manager.generate_dataset()
        input_dim = X_train.shape[1]
        
        # 2. Train all models
        sklearn_model = self.trainer.train_baseline_sklearn(X_train, y_train)
        fixed_neuron = self.trainer.train_fixed_neuron(X_train, y_train, input_dim)
        dynamic_neuron = self.trainer.train_dynamic_neuron(X_train, y_train, input_dim)
        sigmoid_neuron = self.trainer.train_sigmoid_neuron(X_train, y_train, input_dim)
        
        print("\n" + "-" * 60)
        print("EVALUATION ON TEST SET")
        print("-" * 60)
        
        # 3. Evaluate all models
        self.evaluator.evaluate_sklearn_model(
            sklearn_model, X_test, y_test, "Sklearn Perceptron (Baseline)"
        )
        self.evaluator.evaluate_neuron(fixed_neuron, X_test, y_test, "Fixed ReLU Neuron")
        self.evaluator.evaluate_neuron(dynamic_neuron, X_test, y_test, "Dynamic ReLU Neuron")
        self.evaluator.evaluate_neuron(sigmoid_neuron, X_test, y_test, "Sigmoid Neuron")
        
        # 4. Display comprehensive results
        self.evaluator.print_comparison()
        
        # 5. Return evaluator for access to results
        return self.evaluator


def main():
    """Main entry point for the experiment."""
    print("Available datasets:")
    print("  1. breast_cancer - Breast Cancer Wisconsin (569 samples, 30 features)")
    print("  2. titanic - Passenger survival (with engineered categorical features)")
    print("  3. heart_disease - UCI Heart Disease (13 clinical measurements)")
    print()
    
    # Fixed seed for reproducibility
    SEED = 42
    
    # You can easily switch between datasets by changing dataset_type
    datasets_to_test = [
        ('breast_cancer', DatasetConfig(dataset_type='breast_cancer', random_state=SEED)),
        ('titanic', DatasetConfig(dataset_type='titanic', random_state=SEED)),
        ('heart_disease', DatasetConfig(dataset_type='heart_disease', random_state=SEED))
    ]
    
    # Run experiments on all datasets
    all_evaluators = []
    
    for dataset_name, config in datasets_to_test:
        print("\n" + "=" * 70)
        print(f"RUNNING EXPERIMENT ON: {dataset_name.upper()}")
        print("=" * 70)
        
        # Run experiment
        experiment = PerceptronExperiment(config, dataset_name=dataset_name)
        evaluator = experiment.run()
        all_evaluators.append(evaluator)
        
        print("\n" * 2)
    
    # Save all results to single CSV file
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
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
        
        print("\n✓ All results saved to: experiment_results.csv")
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
    