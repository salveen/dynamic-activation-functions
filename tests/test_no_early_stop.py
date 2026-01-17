"""Test to compare activation params with different activation_lr values."""
import sys
sys.path.insert(0, 'src')

from data_utils import DatasetConfig, DataManager
from models import Neuron
import numpy as np

# Use same seed for reproducibility
np.random.seed(42)

config = DatasetConfig(dataset_type='breast_cancer', random_state=42)
dm = DataManager(config)
X_train, X_test, y_train, y_test = dm.generate_dataset()

print("Testing different activation_lr values with NO early stopping")
print("=" * 60)

for act_lr in [0.0, 0.001, 0.01, 0.1, 1.0]:
    np.random.seed(42)  # Reset seed for each test
    
    neuron = Neuron(X_train.shape[1], activation='dynamic_relu', weight_lr=0.01, activation_lr=act_lr)
    
    # Train weights first
    neuron.train_weights(X_train, y_train, epochs=100)
    
    # Now train activation WITHOUT early stopping
    y = y_train.astype(float)
    for epoch in range(1000):  # Many epochs, no early stopping
        z = np.dot(X_train, neuron.weights) + neuron.bias
        predictions = neuron._activation_forward(z)
        error = predictions - y
        neuron._train_activation_params(z, error)
    
    a, b = neuron.activation_state.params['a'], neuron.activation_state.params['b']
    print(f'activation_lr={act_lr}: a={a:.6f}, b={b:.6f}')
