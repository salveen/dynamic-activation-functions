"""Final test to verify activation_lr is correctly passed through the entire system."""
import sys
sys.path.insert(0, 'src')

from data_utils import DatasetConfig, DataManager
from trainer import ModelTrainer
import numpy as np

# Test 1: Verify ModelTrainer stores activation_lr correctly
print("=== Test 1: ModelTrainer stores activation_lr ===")
trainer1 = ModelTrainer(seed=42, weight_lr=0.01, activation_lr=1.0)
trainer2 = ModelTrainer(seed=42, weight_lr=0.01, activation_lr=0.001)
trainer3 = ModelTrainer(seed=42, weight_lr=0.01, activation_lr=None)  # Should default to weight_lr

print(f"trainer1.activation_lr (set to 1.0): {trainer1.activation_lr}")
print(f"trainer2.activation_lr (set to 0.001): {trainer2.activation_lr}")
print(f"trainer3.activation_lr (set to None, should be 0.01): {trainer3.activation_lr}")

# Test 2: Verify Neuron receives activation_lr from trainer
print("\n=== Test 2: Neuron receives activation_lr from trainer ===")
config = DatasetConfig(dataset_type='breast_cancer', random_state=42)
dm = DataManager(config)
X_train, X_test, y_train, y_test = dm.generate_dataset()

np.random.seed(42)
neuron1 = trainer1.train_dynamic_neuron(X_train, y_train, X_train.shape[1])
print(f"neuron1.activation_lr (from trainer1 with 1.0): {neuron1.activation_lr}")

np.random.seed(42)
neuron2 = trainer2.train_dynamic_neuron(X_train, y_train, X_train.shape[1])
print(f"neuron2.activation_lr (from trainer2 with 0.001): {neuron2.activation_lr}")

# Test 3: Final activation params should differ
print("\n=== Test 3: Final activation params comparison ===")
print(f"neuron1 params: {neuron1.activation_state.params}")
print(f"neuron2 params: {neuron2.activation_state.params}")

# Test 4: Direct comparison - are they different?
print("\n=== Test 4: Are params different? ===")
a1, b1 = neuron1.activation_state.params['a'], neuron1.activation_state.params['b']
a2, b2 = neuron2.activation_state.params['a'], neuron2.activation_state.params['b']
print(f"neuron1: a={a1:.10f}, b={b1:.10f}")
print(f"neuron2: a={a2:.10f}, b={b2:.10f}")
print(f"Difference in a: {abs(a1-a2):.10f}")
print(f"Difference in b: {abs(b1-b2):.10f}")
print(f"Are they different? {a1 != a2 or b1 != b2}")
