"""Test to see if different activation_lr values produce meaningfully different results."""
import sys
sys.path.insert(0, 'src')

from data_utils import DatasetConfig, DataManager
from trainer import ModelTrainer
from sklearn.metrics import accuracy_score
import numpy as np

config = DatasetConfig(dataset_type='breast_cancer', random_state=42)
dm = DataManager(config)
X_train, X_test, y_train, y_test = dm.generate_dataset()

print("Comparing different activation_lr values:")
print("=" * 70)
print(f"{'activation_lr':>15} | {'a':>12} | {'b':>12} | {'accuracy':>10}")
print("-" * 70)

for act_lr in [0.0, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0]:
    np.random.seed(42)  # Reset seed for fair comparison
    trainer = ModelTrainer(seed=42, weight_lr=0.01, activation_lr=act_lr)
    neuron = trainer.train_dynamic_neuron(X_train, y_train, X_train.shape[1])
    
    predictions = neuron.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    a = neuron.activation_state.params['a']
    b = neuron.activation_state.params['b']
    
    print(f"{act_lr:>15.3f} | {a:>12.6f} | {b:>12.6f} | {acc:>10.4f}")
