# Dynamic Activation Functions in Neural Networks

> **Can neural networks improve by learning the shape of their activation functions?**
>
> A from-scratch investigation — built entirely in NumPy — into whether per-neuron learnable activation parameters can boost accuracy or accelerate training beyond what fixed activations like ReLU and Sigmoid achieve.

---

## Table of Contents

- [Project Proposal](#project-proposal)
  - [Motivation](#motivation)
  - [Hypothesis](#hypothesis)
  - [Theoretical Background](#theoretical-background)
- [Implementation](#implementation)
  - [Architecture](#architecture)
  - [Dynamic Activation Functions](#dynamic-activation-functions)
  - [Project Structure](#project-structure)
- [Experiments & Results](#experiments--results)
  - [Experiment 1 — Dynamic ReLU Finetuning](#experiment-1--dynamic-relu-finetuning)
  - [Experiment 2 — ReLU–Sigmoid Blend Finetuning](#experiment-2--relusigmoid-blend-finetuning)
  - [Experiment 3 — Dynamic Sigmoid Finetuning](#experiment-3--dynamic-sigmoid-finetuning)
  - [Experiment 4 — Pretrained Activation Transfer](#experiment-4--pretrained-activation-transfer)
  - [Summary of Results](#summary-of-results)
- [Analysis & Conclusion](#analysis--conclusion)
- [Setup & Reproduction](#setup--reproduction)

---

## Project Proposal

### Motivation

Standard neural networks rely on fixed activation functions — ReLU, Sigmoid, Tanh — whose shapes are chosen before training and never change. If the activation shape could **adapt during training**, a network might converge faster, generalise better, or reach higher accuracy with fewer parameters.

### Hypothesis

**Learnable, per-neuron activation function parameters can improve the performance of feedforward neural networks beyond what fixed activations achieve.** Specifically:

1. Given a trained network with fixed activations, **freezing the weights and fine-tuning only the activation shape** should recover additional accuracy that the fixed shape left on the table.
2. Activation shapes learned on one set of trained weights should **transfer to fresh random weights**, accelerating convergence — evidence that a task-optimal activation shape exists independently of a particular weight initialisation.

### Theoretical Background

#### Why Activation Functions Matter

A neural network without nonlinear activations collapses to a single affine transformation regardless of depth. The activation function is what gives depth its power: each additional layer can represent an exponentially richer class of functions. The choice of activation therefore controls both the **expressiveness** of the network and the **geometry of its loss landscape**.

- **ReLU** ( $\max(0, x)$ ) is piecewise linear. Compositions of piecewise-linear functions partition the input space into an exponentially growing number of linear regions, giving deep ReLU networks enormous capacity. However, the fixed zero threshold and unit slope are arbitrary.
- **Sigmoid** ( $\sigma(x) = \frac{1}{1+e^{-x}}$ ) is smooth and bounded, but suffers from vanishing gradients at saturation. Its inflection point is fixed at $x = 0$ and its steepness is fixed at 1.

#### Parameterising the Shape

A natural extension is to make the activation itself learnable. For each neuron $j$, we introduce a small parameter vector $(a_j, b_j)$ that controls the shape:

| Activation | Definition | Learnable Parameters |
|---|---|---|
| DynamicReLU | $f(x) = \max(a_j,\; b_j \cdot x)$ | $a_j$: leak floor · $b_j$: positive slope |
| DynamicSigmoid | $f(x) = \frac{1}{1 + e^{-b_j(x - a_j)}}$ | $a_j$: horizontal shift · $b_j$: steepness |
| ReLU–Sigmoid Blend | $f(x) = a_j \cdot \text{ReLU}(x) + b_j \cdot \sigma(x)$ | $a_j, b_j$: mixing coefficients |

When $a = 0, b = 1$ the dynamic variants reduce exactly to their fixed counterparts — so the standard activation is always a reachable special case. Training can only help, never hurt, in principle.

#### The Capacity Argument

A two-hidden-layer MLP with 256 and 128 neurons has roughly **200,000 weight parameters**. Adding per-neuron $(a, b)$ pairs introduces only **768 activation parameters** — less than 0.4 % of the total. The question is whether this thin parametric slice captures a degree of freedom that weights alone cannot efficiently reach, or whether the weight parameters already subsume the effect.

Universal approximation theorems guarantee that a sufficiently wide network with *any* reasonable fixed activation can approximate any continuous function. In practice, however, networks are finite, and the right activation shape could reduce the width or depth required. The potential benefit is therefore most likely to appear in **capacity-constrained** regimes — small networks, limited data, or hard tasks where every parameter must pull its weight.

---

## Implementation

Everything is implemented from scratch in **NumPy** — no PyTorch, no TensorFlow. This includes forward/backward passes, mini-batch SGD, per-neuron gradient computation for learnable activation parameters, early stopping, and cross-entropy loss.

### Architecture

<table>
<tr><td><b>Component</b></td><td><b>Details</b></td></tr>
<tr><td>Network</td><td>2-hidden-layer MLP: Input → 256 → 128 → 10 (Softmax)</td></tr>
<tr><td>Loss</td><td>Cross-entropy with numerically stable softmax</td></tr>
<tr><td>Optimiser</td><td>Mini-batch SGD (batch size 128)</td></tr>
<tr><td>Regularisation</td><td>Early stopping (patience = 10), optional Dropout</td></tr>
<tr><td>Init</td><td>He (ReLU-based) or Xavier (Sigmoid-based)</td></tr>
<tr><td>Datasets</td><td>MNIST · Fashion-MNIST · CIFAR-10</td></tr>
<tr><td>Seeds</td><td>20 random seeds per experiment for statistical robustness</td></tr>
</table>

### Dynamic Activation Functions

Each dynamic activation is parameterised **per neuron**, meaning every hidden unit in the network has its own learnable $(a, b)$. Gradients for these parameters are derived analytically and flow through standard backpropagation.

The key implementation detail: during the **activation finetuning phase**, weight gradients are computed (for backpropagation to earlier layers) but **not applied** — only the $(a, b)$ gradients update. This is controlled by a single flag in the backward pass.

### Project Structure

```
dynamic-activation-functions/
│
├── src/
│   ├── activations.py          # Activation functions (ReLU, DynamicReLU, Sigmoid, DynamicSigmoid, DynamicReLUSigmoid, Softmax)
│   ├── layers.py               # Dense, Dropout, BatchNorm layers
│   ├── mlp.py                  # MLP model, config dataclass, factory functions
│   ├── mlp_trainer.py          # Training loop, early stopping, experiment runner
│   └── data_utils.py           # Dataset loading & preprocessing (12 datasets)
│
├── run_mlp_relu_experiment.ipynb              # Experiment 1
├── run_mlp_relu_sigmoid_experiment.ipynb      # Experiment 2
├── run_mlp_sigmoid_experiment.ipynb           # Experiment 3
├── run_pretrained_activation_experiment.ipynb # Experiment 4
│
├── results.txt                 # Raw experiment output
├── requirements.txt
└── README.md
```

---

## Experiments & Results

### Experiment 1 — Dynamic ReLU Finetuning

**Protocol:** Train a baseline ReLU MLP to convergence → copy weights → replace ReLU with DynamicReLU $f(x) = \max(a_j, b_j \cdot x)$ → freeze weights → finetune only $(a, b)$.

Initialised at $a=0, b=1$ so the network starts identically to baseline ReLU.

| Dataset | Baseline (ReLU) | + DynamicReLU | Δ Accuracy |
|---|---|---|---|
| **MNIST** | 0.9639 ± 0.0006 | 0.9639 ± 0.0007 | +0.0000 (0.00 %) |
| **Fashion-MNIST** | 0.8845 ± 0.0020 | 0.8858 ± 0.0016 | +0.0013 (+0.13 %) |
| **CIFAR-10** | 0.5155 ± 0.0038 | 0.5201 ± 0.0030 | +0.0046 (+0.46 %) |

### Experiment 2 — ReLU–Sigmoid Blend Finetuning

**Protocol:** Train a baseline ReLU MLP to convergence → copy weights → replace ReLU with $f(x) = a \cdot \text{ReLU}(x) + b \cdot \sigma(x)$ → freeze weights → finetune only $(a, b)$.

| Dataset | Baseline (ReLU) | + Finetuning | Δ Accuracy |
|---|---|---|---|
| **MNIST** | 0.9639 ± 0.0006 | 0.9639 ± 0.0006 | +0.0000 (0.00 %) |
| **Fashion-MNIST** | 0.8845 ± 0.0020 | 0.8845 ± 0.0020 | +0.0000 (0.00 %) |
| **CIFAR-10** | 0.5155 ± 0.0038 | 0.5213 ± 0.0036 | **+0.0058 (+0.58 %)** |

CIFAR-10 shows the largest gain — but +0.58 % on a 51 % baseline is not practically meaningful.

### Experiment 3 — Dynamic Sigmoid Finetuning

**Protocol:** Baseline uses Sigmoid activations; finetuning replaces them with DynamicSigmoid: $f(x) = \frac{1}{1 + e^{-b(x-a)}}$.

| Dataset | Baseline (Sigmoid) | + Finetuning | Δ Accuracy |
|---|---|---|---|
| **MNIST** | 0.9128 ± 0.0009 | 0.9131 ± 0.0010 | +0.0003 (+0.03 %) |
| **Fashion-MNIST** | 0.8339 ± 0.0012 | 0.8346 ± 0.0013 | +0.0007 (+0.07 %) |
| **CIFAR-10** | 0.4189 ± 0.0019 | 0.4197 ± 0.0019 | +0.0008 (+0.08 %) |

Improvements are negligible and not statistically significant across all three datasets.

### Experiment 4 — Pretrained Activation Transfer

**Protocol (3 phases):**
1. Train weights with fixed ReLU → $W^*$
2. Freeze $W^*$, learn DynamicReLU $(a, b)$ → $a^*, b^*$
3. Train **fresh random weights** with frozen $a^*, b^*$ — compare convergence to Phase 1

| Dataset | Phase 1 (Fixed ReLU) | Phase 3 (Pretrained Act.) | Δ Accuracy | Epochs Saved |
|---|---|---|---|---|
| **MNIST** | 0.9639 ± 0.0006 (30 ep) | 0.9640 ± 0.0006 (30 ep) | +0.0001 | 0.0 |
| **Fashion-MNIST** | 0.8845 ± 0.0020 (30 ep) | 0.8847 ± 0.0022 (30 ep) | +0.0002 | 0.0 |
| **CIFAR-10** | 0.5155 ± 0.0038 (21.6 ep) | 0.5147 ± 0.0033 (21.1 ep) | −0.0008 | 0.5 |

No significant accuracy improvement or convergence speedup. Paired t-tests confirm none of the differences are statistically significant at $p < 0.05$.

### Summary of Results

| Experiment | Best Δ Accuracy | Statistically Significant? |
|---|---|---|
| Dynamic ReLU Finetuning | +0.46 % (CIFAR-10) | No |
| ReLU–Sigmoid Blend Finetuning | +0.58 % (CIFAR-10) | No |
| Dynamic Sigmoid Finetuning | +0.08 % (CIFAR-10) | No |
| Pretrained Activation Transfer | +0.02 % (Fashion-MNIST) | No |

---

## Analysis & Conclusion

**The hypothesis is not supported for the architecture tested.** Learnable activation parameters do not produce meaningful accuracy gains or convergence speedups on two-hidden-layer MLPs with 256 → 128 neurons.

### Why?

The explanation lies in the **parameter ratio**. The network has ~200,000 weight parameters versus only 768 activation parameters (< 0.4 %). The weight matrices already provide enough degrees of freedom to approximate the target function — the activation shape adds a negligible new dimension to the optimisation landscape.

This effect is likely **even more pronounced** in deeper or wider architectures. Deeper networks with fixed ReLU can approximate increasingly complex functions through composition — each layer "bends" the piecewise-linear mapping further. The expressiveness gained from learnable activation shapes becomes redundant when the network already has sufficient depth and width.

### When might dynamic activations help?

The results suggest dynamic activations are most likely to matter in **capacity-constrained regimes**:
- Very small or shallow networks where every parameter counts
- Extremely hard tasks relative to model size
- Settings where architectural changes (more layers/neurons) are not possible

These are open directions worth exploring in future work.

---

## Setup & Reproduction

**Requirements:** Python 3.10+, NumPy, pandas, scikit-learn.

```bash
pip install -r requirements.txt
```

Run any experiment notebook. Datasets download automatically via scikit-learn / OpenML.

| Notebook | Experiment |
|---|---|
| `run_mlp_relu_experiment.ipynb` | Baseline ReLU vs Dynamic ReLU Finetuning |
| `run_mlp_relu_sigmoid_experiment.ipynb` | ReLU–Sigmoid Blend Finetuning |
| `run_mlp_sigmoid_experiment.ipynb` | Dynamic Sigmoid Finetuning |
| `run_pretrained_activation_experiment.ipynb` | 3-Phase Pretrained Activation Transfer |