"""Test to understand why grad_a is 0."""
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

# For a=0, b=1: mask_a_active when 0 >= z (z <= 0)
print(f"a={a}, b={b}")
print(f"Condition for mask_a_active: a >= b*z means {a} >= {b}*z means z <= {a/b}")
print(f"Number of samples where z <= 0: {np.sum(z <= 0)}")

mask_a_active = (a >= b * z).astype(float)
print(f"mask_a_active sum: {np.sum(mask_a_active)}")

# Check the error for those samples
error_for_a_active = error[mask_a_active.astype(bool)]
print(f"Error for a-active samples: mean={np.mean(error_for_a_active):.6f}")
print(f"error * mask_a_active: {np.mean(error * mask_a_active):.10f}")

# But wait - predictions for these samples should be 'a' (which is 0)
# So prediction=0 for z <= 0
# error = prediction - y = 0 - y = -y
# For class 0: error = 0 - 0 = 0
# For class 1: error = 0 - 1 = -1
print(f"\nFor samples where z <= 0:")
active_mask = z <= 0
print(f"  y values: {np.unique(y[active_mask], return_counts=True)}")
print(f"  prediction values: {np.unique(predictions[active_mask])}")
print(f"  error values: {np.unique(error[active_mask])}")
