"""Test to debug activation_lr flow with verbose output."""
import sys
sys.path.insert(0, 'src')

from data_utils import DatasetConfig, DataManager
from models import Neuron
import numpy as np

config = DatasetConfig(dataset_type='breast_cancer', random_state=42)
dm = DataManager(config)
X_train, X_test, y_train, y_test = dm.generate_dataset()

# Create neuron with activation_lr=1.0
neuron = Neuron(X_train.shape[1], activation='dynamic_relu', weight_lr=0.01, activation_lr=1.0)
print(f'Initial params: {neuron.activation_state.params}')

# Train weights first
neuron.train_weights(X_train, y_train, epochs=100)
print(f'After weight training: {neuron.activation_state.params}')

# Now manually do what train_activation does, but with verbose output
print("\n=== Manual activation training (verbose) ===")
y = y_train.astype(float)
best_loss = float('inf')
epochs_without_improvement = 0
patience = 10
min_delta = 1e-6

for epoch in range(100):
    z = np.dot(X_train, neuron.weights) + neuron.bias
    predictions = neuron._activation_forward(z)
    error = predictions - y
    
    # Get params before update
    a_before = neuron.activation_state.params['a']
    b_before = neuron.activation_state.params['b']
    
    # Update activation params
    neuron._train_activation_params(z, error)
    
    a_after = neuron.activation_state.params['a']
    b_after = neuron.activation_state.params['b']
    
    epoch_loss = float(np.mean(error ** 2))
    
    if epoch < 5 or epoch % 20 == 0:
        print(f'Epoch {epoch}: loss={epoch_loss:.6f}, a: {a_before:.6f} -> {a_after:.6f}, b: {b_before:.6f} -> {b_after:.6f}')
    
    if epoch_loss < best_loss - min_delta:
        best_loss = epoch_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch}!')
            break

print(f'\nFinal params: {neuron.activation_state.params}')
