# Dynamic Activation Functions

A from-scratch neural network library in NumPy that explores **learnable activation functions** — where the activation shape itself is trained alongside (or separately from) network weights.

## Research Questions

1. **Can finetuning activation parameters improve a trained baseline?**  
   Train a standard MLP, then freeze weights and learn activation shape parameters — does accuracy improve?

2. **Which learnable activation works best?**  
   Compare `DynamicReLU(a,b)`, `DynamicSigmoid(a,b)`, and a linear combination `a·ReLU + b·Sigmoid`.

3. **Can pretrained activations speed up training?**  
   Learn activation parameters on good weights, then train fresh random weights with those activations — do weights converge faster?

## Activation Functions

| Activation | Formula | Learnable Params |
|---|---|---|
| ReLU | $\max(0, x)$ | — |
| DynamicReLU | $\max(a, b \cdot x)$ | $a, b$ per neuron |
| Sigmoid | $\frac{1}{1 + e^{-x}}$ | — |
| DynamicSigmoid | $\frac{1}{1 + e^{-(b \cdot x + a)}}$ | $a, b$ per neuron |
| DynamicReLUSigmoid | $a \cdot \text{ReLU}(x) + b \cdot \sigma(x)$ | $a, b$ per neuron |

## Project Structure

```
├── run_mlp_relu_sigmoid_experiment.ipynb   # ReLU vs DynamicReLUSigmoid finetuning
├── run_mlp_sigmoid_experiment.ipynb        # Sigmoid vs DynamicSigmoid finetuning
├── run_pretrained_activation_experiment.ipynb  # 3-phase pretrained activation experiment
├── dataset_analysis.ipynb                  # Dataset exploration & visualization
├── src/
│   ├── activations.py      # All activation functions (fixed & learnable)
│   ├── layers.py            # Dense, Dropout, BatchNorm layers
│   ├── mlp.py               # MLP model + factory functions
│   ├── mlp_trainer.py       # Training loop, experiments, early stopping
│   └── data_utils.py        # Dataset loading (MNIST, Fashion-MNIST, CIFAR-10, etc.)
├── csvs/                    # Archived experiment results
├── requirements.txt
└── results.txt
```

## Experiments

### ReLU + Sigmoid Finetuning (`run_mlp_relu_sigmoid_experiment.ipynb`)
Trains a baseline ReLU MLP, then finetunes with $f(x) = a \cdot \text{ReLU}(x) + b \cdot \sigma(x)$ where only $a, b$ are updated (weights frozen). Tested on MNIST, Fashion-MNIST, and CIFAR-10 with 20 seeds.

### Sigmoid Finetuning (`run_mlp_sigmoid_experiment.ipynb`)
Same pipeline but replaces Sigmoid baseline with DynamicSigmoid finetuning.

### Pretrained Activations (`run_pretrained_activation_experiment.ipynb`)
Three-phase experiment:
1. Train weights with fixed ReLU → $W^*$
2. Freeze $W^*$, learn DynamicReLU$(a, b)$ → $a^*, b^*$  
3. Train fresh random weights with pretrained $a^*, b^*$ — compare convergence to Phase 1

## Results

**Finetuning activation functions does not produce a statistically significant improvement** over fixed activations for the MLP architecture tested (2 hidden layers, 256→128 neurons). Across all three experiments — DynamicReLU, DynamicSigmoid, and DynamicReLUSigmoid — the learned activation parameters converge to values very close to their fixed counterparts, and accuracy gains are negligible.

Similarly, pretrained activation parameters do not meaningfully reduce weight training convergence time compared to standard fixed ReLU.

### Why this is expected

In small-to-medium MLPs, the network already has enough capacity to approximate the target function with fixed activations like ReLU. The learnable $a, b$ parameters add very few degrees of freedom relative to the thousands of weight parameters, so the optimization landscape doesn't change meaningfully.

For **larger architectures** (deeper networks, wider layers), this effect is likely even more pronounced. Deeper networks with fixed ReLU activations can approximate increasingly complex functions through composition — each layer can "bend" the piecewise-linear mapping further. The added expressiveness of learnable activation shapes becomes redundant when the network already has sufficient depth and width to achieve the same effect through its weights alone.

## Setup

```bash
pip install -r requirements.txt
```

Then open any notebook and run all cells. Datasets are downloaded automatically via scikit-learn / OpenML.