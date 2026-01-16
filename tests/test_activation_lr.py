"""Test to debug activation_lr flow."""
import sys
sys.path.insert(0, 'src')

from data_utils import DatasetConfig, DataManager
from trainer import ModelTrainer

# Test 1: Trainer directly
print("=== Test 1: Trainer directly ===")
trainer = ModelTrainer(seed=42, weight_lr=0.01, activation_lr=1.0)
print(f'Trainer activation_lr: {trainer.activation_lr}')

trainer2 = ModelTrainer(seed=42, weight_lr=0.01, activation_lr=None)
print(f'Trainer2 activation_lr (None passed): {trainer2.activation_lr}')

# Test 2: Full training flow
print("\n=== Test 2: Full training flow ===")
config = DatasetConfig(dataset_type='breast_cancer', random_state=42)
dm = DataManager(config)
X_train, X_test, y_train, y_test = dm.generate_dataset()

trainer = ModelTrainer(seed=42, weight_lr=0.01, activation_lr=1.0)
print(f'Trainer activation_lr: {trainer.activation_lr}')

neuron = trainer.train_dynamic_neuron(X_train, y_train, X_train.shape[1])
print(f'Neuron activation_lr: {neuron.activation_lr}')
print(f'Final activation params: {neuron.activation_state.params}')

# Test 3: Compare with different activation_lr values
print("\n=== Test 3: Compare different activation_lr values ===")
for act_lr in [0.001, 0.1, 1.0, 10.0]:
    trainer = ModelTrainer(seed=42, weight_lr=0.01, activation_lr=act_lr)
    neuron = trainer.train_dynamic_neuron(X_train, y_train, X_train.shape[1])
    a, b = neuron.activation_state.params['a'], neuron.activation_state.params['b']
    print(f'activation_lr={act_lr}: a={a:.6f}, b={b:.6f}')
