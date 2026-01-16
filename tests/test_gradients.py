"""Test to see gradient magnitudes."""
import sys
sys.path.insert(0, 'src')

from data_utils import DatasetConfig, DataManager
from models import Neuron
import numpy as np

np.random.seed(42)

config = DatasetConfig(dataset_type='breast_cancer', random_state=42)
dm = DataManager(config)
X_train, X_test, y_train, y_test = dm.generate_dataset()

neuron = Neuron(X_train.shape[1], activation='dynamic_relu', weight_lr=0.01, activation_lr=1.0)

# Train weights first
neuron.train_weights(X_train, y_train, epochs=100)

# Look at gradients
y = y_train.astype(float)
z = np.dot(X_train, neuron.weights) + neuron.bias
predictions = neuron._activation_forward(z)
error = predictions - y

a = neuron.activation_state.params['a']
b = neuron.activation_state.params['b']

mask_a_active = (a >= b * z).astype(float)
mask_b_active = (b * z > a).astype(float)

grad_a = np.mean(error * mask_a_active)
grad_b = np.mean(error * mask_b_active * z)

print(f"Error stats: mean={np.mean(error):.6f}, std={np.std(error):.6f}")
print(f"z stats: mean={np.mean(z):.6f}, std={np.std(z):.6f}")
print(f"mask_a_active: {np.sum(mask_a_active)} samples (where a >= b*z)")
print(f"mask_b_active: {np.sum(mask_b_active)} samples (where b*z > a)")
print(f"grad_a: {grad_a:.6f}")
print(f"grad_b: {grad_b:.6f}")
print(f"activation_lr * grad_a = {neuron.activation_lr * grad_a:.6f}")
print(f"activation_lr * grad_b = {neuron.activation_lr * grad_b:.6f}")

print(f"\nPredictions range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
print(f"Unique predictions: {np.unique(predictions)[:5]}...")
