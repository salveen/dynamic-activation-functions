#!/usr/bin/env python3
"""
Pretrained Activation Experiment
================================

Research question: If we train activation functions from scratch based on good
weights, and then train random weights with those pretrained activations — does
it reduce training time for weights?

Experimental pipeline (3 phases):

Phase 1 – Baseline: Train weights with fixed ReLU activation.
    min_W  L(f(X; W, ReLU))
    → produces W* (trained weights) and records epochs/time to converge.

Phase 2 – Learn activations on top of good weights:
    Take W* from Phase 1, replace ReLU with DynamicReLU(a, b),
    freeze weights, and solve:
        min_{a,b}  L(f(X; W*, DynamicReLU(a, b)))
    → produces a*, b* (pretrained activation parameters).

Phase 3 – Train random weights with pretrained activations:
    Initialize fresh random weights W', plug in a*, b* from Phase 2,
    freeze activations, and solve:
        min_{W'}  L(f(X; W', DynamicReLU(a*, b*)))
    → records epochs/time to converge and compares with Phase 1.

Outputs
-------
• Console report comparing Phase 1 vs Phase 3:
    – Epochs to converge
    – Wall-clock training time
    – Final accuracy
    – Weight similarity (cosine similarity, L2 distance)
• Answers to:
    1. Will weights be different?  (almost certainly yes — different loss landscape)
    2. Did weight training finish earlier with pretrained activations?
"""

import sys
import os
import copy
import time

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from activations import DynamicReLU
from layers import Dense
from mlp import MLP, MLPConfig
from mlp_trainer import MLPTrainer, CrossEntropyLoss, TrainingHistory
from data_utils import DatasetConfig, DataManager


# ============================================================================
# Helper utilities
# ============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened weight vectors."""
    a_flat = a.ravel()
    b_flat = b.ravel()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 (Euclidean) distance between two flattened weight vectors."""
    return float(np.linalg.norm(a.ravel() - b.ravel()))


def extract_all_weights(model: MLP) -> np.ndarray:
    """Concatenate all layer weights and biases into a single flat vector."""
    parts = []
    for layer in model.layers:
        if isinstance(layer, Dense):
            parts.append(layer.weights.ravel())
            parts.append(layer.bias.ravel())
    return np.concatenate(parts)


def extract_activation_params(model: MLP) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract (a, b) activation parameters from each Dense layer
    that has a DynamicReLU activation.
    """
    params = []
    for layer in model.layers:
        if isinstance(layer, Dense) and isinstance(layer.activation, DynamicReLU):
            a = np.copy(layer.activation._a) if isinstance(layer.activation._a, np.ndarray) else np.array([layer.activation._a])
            b = np.copy(layer.activation._b) if isinstance(layer.activation._b, np.ndarray) else np.array([layer.activation._b])
            params.append((a, b))
    return params


def inject_activation_params(model: MLP, params: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    """
    Inject pretrained (a, b) activation parameters into each Dense layer
    that has a DynamicReLU activation.
    """
    idx = 0
    for layer in model.layers:
        if isinstance(layer, Dense) and isinstance(layer.activation, DynamicReLU):
            a_val, b_val = params[idx]
            if layer.activation.per_neuron:
                layer.activation._a = np.copy(a_val)
                layer.activation._b = np.copy(b_val)
            else:
                layer.activation._a = float(a_val.ravel()[0])
                layer.activation._b = float(b_val.ravel()[0])
            idx += 1


# ============================================================================
# Training helpers (return rich history)
# ============================================================================

def train_weights_only(
    model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    trainer: MLPTrainer,
) -> TrainingHistory:
    """
    Train only the network weights; activation parameters are frozen.

    For fixed activations (ReLU) this is normal training.
    For dynamic activations this freezes a, b and only updates W, bias.
    """
    start_time = time.time()
    num_classes = model.config.output_dim
    y_train_oh = trainer._to_onehot(y_train, num_classes)
    y_val_oh = trainer._to_onehot(y_val, num_classes)

    history = TrainingHistory(
        train_losses=[], train_accuracies=[],
        val_losses=[], val_accuracies=[],
        epochs_trained=0, training_time=0.0,
    )

    best_val_loss = float('inf')
    patience_counter = 0
    n = len(X_train)

    for epoch in range(trainer.epochs):
        model.train()
        idx = np.random.permutation(n)
        X_shuf = X_train[idx]
        y_shuf = y_train_oh[idx]

        epoch_losses = []
        for start in range(0, n, trainer.batch_size):
            end = min(start + trainer.batch_size, n)
            Xb = X_shuf[start:end]
            yb = y_shuf[start:end]

            preds = model.forward(Xb)
            loss = CrossEntropyLoss.compute(preds, yb)
            epoch_losses.append(loss)

            grad = CrossEntropyLoss.gradient(preds, yb)
            # update_weights=True, update_activation=False
            model.backward(grad, update_activation=False, update_weights=True)

        train_loss = float(np.mean(epoch_losses))
        train_acc = float(trainer._compute_accuracy(model, X_train, y_train))
        history.train_losses.append(train_loss)
        history.train_accuracies.append(train_acc)

        model.eval()
        val_preds = model.forward(X_val)
        val_loss = float(CrossEntropyLoss.compute(val_preds, y_val_oh))
        val_acc = float(trainer._compute_accuracy(model, X_val, y_val))
        history.val_losses.append(val_loss)
        history.val_accuracies.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= trainer.early_stopping_patience:
                if trainer.verbose:
                    print(f"    Early stopping at epoch {epoch + 1}")
                break

        if trainer.verbose and (epoch + 1) % trainer.print_every == 0:
            print(
                f"    Epoch {epoch+1:>3}/{trainer.epochs} | "
                f"Loss {train_loss:.4f} | Train {train_acc:.4f} | Val {val_acc:.4f}"
            )

        history.epochs_trained = epoch + 1

    history.training_time = time.time() - start_time
    return history


def train_activations_only(
    model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    trainer: MLPTrainer,
    epochs: int = 100,
) -> TrainingHistory:
    """
    Train only the learnable activation parameters; weights are frozen.

    This is Phase 2: given good weights W*, learn a*, b* for DynamicReLU.
    """
    start_time = time.time()
    num_classes = model.config.output_dim
    y_train_oh = trainer._to_onehot(y_train, num_classes)
    y_val_oh = trainer._to_onehot(y_val, num_classes)

    history = TrainingHistory(
        train_losses=[], train_accuracies=[],
        val_losses=[], val_accuracies=[],
        epochs_trained=0, training_time=0.0,
    )

    best_val_loss = float('inf')
    patience_counter = 0
    n = len(X_train)

    for epoch in range(epochs):
        model.train()
        idx = np.random.permutation(n)
        X_shuf = X_train[idx]
        y_shuf = y_train_oh[idx]

        epoch_losses = []
        for start in range(0, n, trainer.batch_size):
            end = min(start + trainer.batch_size, n)
            Xb = X_shuf[start:end]
            yb = y_shuf[start:end]

            preds = model.forward(Xb)
            loss = CrossEntropyLoss.compute(preds, yb)
            epoch_losses.append(loss)

            grad = CrossEntropyLoss.gradient(preds, yb)
            # update_weights=False, update_activation=True
            model.backward(grad, update_activation=True, update_weights=False)

        train_loss = float(np.mean(epoch_losses))
        train_acc = float(trainer._compute_accuracy(model, X_train, y_train))
        history.train_losses.append(train_loss)
        history.train_accuracies.append(train_acc)

        model.eval()
        val_preds = model.forward(X_val)
        val_loss = float(CrossEntropyLoss.compute(val_preds, y_val_oh))
        val_acc = float(trainer._compute_accuracy(model, X_val, y_val))
        history.val_losses.append(val_loss)
        history.val_accuracies.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= trainer.early_stopping_patience:
                if trainer.verbose:
                    print(f"    [Activation] Early stopping at epoch {epoch + 1}")
                break

        if trainer.verbose and (epoch + 1) % trainer.print_every == 0:
            print(
                f"    [Act] Epoch {epoch+1:>3}/{epochs} | "
                f"Loss {train_loss:.4f} | Train {train_acc:.4f} | Val {val_acc:.4f}"
            )

        history.epochs_trained = epoch + 1

    history.training_time = time.time() - start_time
    return history


# ============================================================================
# Phase builders
# ============================================================================

def build_fixed_relu_model(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    lr: float,
) -> MLP:
    """Build an MLP with standard (fixed) ReLU hidden activations."""
    config = MLPConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        hidden_activation="relu",
        output_activation="softmax",
        learning_rate=lr,
        weight_init="he",
    )
    return MLP(config)


def build_dynamic_relu_model(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    lr: float,
    activation_lr: float,
    per_neuron: bool = True,
) -> MLP:
    """Build an MLP with DynamicReLU hidden activations."""
    config = MLPConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        hidden_activation="dynamic_relu",
        output_activation="softmax",
        learning_rate=lr,
        activation_lr=activation_lr,
        per_neuron_activation=per_neuron,
        weight_init="he",
    )
    return MLP(config)


def copy_weights_into(dst: MLP, src: MLP) -> None:
    """Copy weights and biases from src into dst (layer-by-layer)."""
    for d_layer, s_layer in zip(dst.layers, src.layers):
        if isinstance(d_layer, Dense) and isinstance(s_layer, Dense):
            d_layer.weights = s_layer.weights.copy()
            d_layer.bias = s_layer.bias.copy()


# ============================================================================
# Main experiment
# ============================================================================

@dataclass
class PhaseResult:
    """Stores per-phase metrics."""
    name: str
    epochs_trained: int
    training_time: float
    final_train_acc: float
    final_val_acc: float
    best_val_acc: float
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)


def run_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: Optional[List[int]] = None,
    weight_lr: float = 0.01,
    activation_lr: float = 0.01,
    weight_epochs: int = 30,
    activation_epochs: int = 30,
    batch_size: int = 128,
    patience: int = 10,
    per_neuron: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, PhaseResult]:
    """
    Run the full 3-phase experiment on pre-loaded data.

    Returns dict with keys 'phase1', 'phase2', 'phase3'.
    """
    if hidden_dims is None:
        hidden_dims = [256, 128]

    np.random.seed(seed)

    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    if verbose:
        print(f"  Architecture: {[input_dim] + hidden_dims + [output_dim]}")

    trainer = MLPTrainer(
        epochs=weight_epochs,
        batch_size=batch_size,
        learning_rate=weight_lr,
        early_stopping_patience=patience,
        verbose=verbose,
        print_every=10,
    )

    results: Dict[str, PhaseResult] = {}

    # ==================================================================
    # PHASE 1: Train weights with fixed ReLU
    # ==================================================================
    if verbose:
        print(f"\n{'─'*70}")
        print("  PHASE 1: Train weights with fixed ReLU activation")
        print(f"{'─'*70}")

    np.random.seed(seed)
    phase1_model = build_fixed_relu_model(input_dim, hidden_dims, output_dim, weight_lr)
    # Save the random weight init state right after building the model
    phase1_init_weights = extract_all_weights(phase1_model)

    h1 = train_weights_only(phase1_model, X_train, y_train, X_val, y_val, trainer)

    results['phase1'] = PhaseResult(
        name="Phase 1: Weights on Fixed ReLU",
        epochs_trained=h1.epochs_trained,
        training_time=h1.training_time,
        final_train_acc=h1.train_accuracies[-1],
        final_val_acc=h1.val_accuracies[-1],
        best_val_acc=max(h1.val_accuracies),
        train_losses=h1.train_losses,
        val_losses=h1.val_losses,
    )

    if verbose:
        print(f"  → Epochs: {h1.epochs_trained} | Time: {h1.training_time:.2f}s")
        print(f"  → Train acc: {h1.train_accuracies[-1]:.4f} | Val acc: {h1.val_accuracies[-1]:.4f}")

    # ==================================================================
    # PHASE 2: Learn activations on pretrained weights (weights frozen)
    # ==================================================================
    if verbose:
        print(f"\n{'─'*70}")
        print("  PHASE 2: Train DynamicReLU(a,b) on pretrained weights (W frozen)")
        print(f"{'─'*70}")

    # Build a dynamic model and copy the trained weights from Phase 1
    phase2_model = build_dynamic_relu_model(
        input_dim, hidden_dims, output_dim,
        lr=weight_lr, activation_lr=activation_lr, per_neuron=per_neuron,
    )
    copy_weights_into(phase2_model, phase1_model)

    # Verify accuracy matches Phase 1 before activation training
    # (DynamicReLU initialises to a=0, b=1 which is standard ReLU)
    pre_acc = float(trainer._compute_accuracy(phase2_model, X_val, y_val))
    if verbose:
        print(f"  Accuracy before activation training (should ≈ Phase 1): {pre_acc:.4f}")

    h2 = train_activations_only(
        phase2_model, X_train, y_train, X_val, y_val,
        trainer, epochs=activation_epochs,
    )

    results['phase2'] = PhaseResult(
        name="Phase 2: Learn activations (weights frozen)",
        epochs_trained=h2.epochs_trained,
        training_time=h2.training_time,
        final_train_acc=h2.train_accuracies[-1],
        final_val_acc=h2.val_accuracies[-1],
        best_val_acc=max(h2.val_accuracies),
        train_losses=h2.train_losses,
        val_losses=h2.val_losses,
    )

    if verbose:
        print(f"  → Epochs: {h2.epochs_trained} | Time: {h2.training_time:.2f}s")
        print(f"  → Val acc after activation training: {h2.val_accuracies[-1]:.4f}")

    # Extract the learned activation parameters
    pretrained_act_params = extract_activation_params(phase2_model)
    if verbose:
        for i, (a, b) in enumerate(pretrained_act_params):
            if a.size > 1:
                print(f"  Layer {i}: a ∈ [{a.min():.4f}, {a.max():.4f}] (mean {a.mean():.4f}), "
                      f"b ∈ [{b.min():.4f}, {b.max():.4f}] (mean {b.mean():.4f})")
            else:
                print(f"  Layer {i}: a={float(a):.4f}, b={float(b):.4f}")

    # ==================================================================
    # PHASE 3: Train random weights with pretrained activations (a,b frozen)
    # ==================================================================
    if verbose:
        print(f"\n{'─'*70}")
        print("  PHASE 3: Train random weights with pretrained DynamicReLU(a*,b*)")
        print(f"{'─'*70}")

    # Use the SAME seed so random weight init matches Phase 1's init
    np.random.seed(seed)
    phase3_model = build_dynamic_relu_model(
        input_dim, hidden_dims, output_dim,
        lr=weight_lr, activation_lr=activation_lr, per_neuron=per_neuron,
    )
    # Inject pretrained activation parameters
    inject_activation_params(phase3_model, pretrained_act_params)

    # Confirm initial weights match Phase 1 init
    phase3_init_weights = extract_all_weights(phase3_model)
    init_cos = cosine_similarity(phase1_init_weights, phase3_init_weights)
    if verbose:
        print(f"  Initial weight cosine similarity with Phase 1 init: {init_cos:.6f}")
        print(f"  (Should be 1.0 — same seed, same architecture)")

    h3 = train_weights_only(phase3_model, X_train, y_train, X_val, y_val, trainer)

    results['phase3'] = PhaseResult(
        name="Phase 3: Random weights + pretrained activations",
        epochs_trained=h3.epochs_trained,
        training_time=h3.training_time,
        final_train_acc=h3.train_accuracies[-1],
        final_val_acc=h3.val_accuracies[-1],
        best_val_acc=max(h3.val_accuracies),
        train_losses=h3.train_losses,
        val_losses=h3.val_losses,
    )

    if verbose:
        print(f"  → Epochs: {h3.epochs_trained} | Time: {h3.training_time:.2f}s")
        print(f"  → Train acc: {h3.train_accuracies[-1]:.4f} | Val acc: {h3.val_accuracies[-1]:.4f}")

    # ==================================================================
    # ANALYSIS
    # ==================================================================
    if verbose:
        print_analysis(results, phase1_model, phase3_model, phase1_init_weights)

    return results


# ============================================================================
# Analysis and reporting
# ============================================================================

def print_analysis(
    results: Dict[str, PhaseResult],
    phase1_model: MLP,
    phase3_model: MLP,
    phase1_init_weights: np.ndarray,
) -> None:
    """Print a comprehensive comparison of Phase 1 vs Phase 3."""
    p1 = results['phase1']
    p2 = results['phase2']
    p3 = results['phase3']

    # Weight comparison
    w1 = extract_all_weights(phase1_model)
    w3 = extract_all_weights(phase3_model)
    cos_sim = cosine_similarity(w1, w3)
    l2_dist = l2_distance(w1, w3)
    w1_norm = float(np.linalg.norm(w1))
    w3_norm = float(np.linalg.norm(w3))
    # Relative L2 distance (normalised by average weight norm)
    rel_l2 = l2_dist / ((w1_norm + w3_norm) / 2) if (w1_norm + w3_norm) > 0 else 0.0

    # Per-layer weight comparison
    layer_comparisons = []
    idx = 0
    for l1, l3 in zip(phase1_model.layers, phase3_model.layers):
        if isinstance(l1, Dense) and isinstance(l3, Dense):
            lc = cosine_similarity(l1.weights, l3.weights)
            ld = l2_distance(l1.weights, l3.weights)
            layer_comparisons.append((idx, lc, ld))
            idx += 1

    print(f"\n{'='*70}")
    print("  ANALYSIS: Phase 1 (Fixed ReLU) vs Phase 3 (Pretrained DynamicReLU)")
    print(f"{'='*70}")

    # ---- Side-by-side comparison table ----
    print(f"\n  {'Metric':<40} {'Phase 1':>12} {'Phase 3':>12} {'Δ':>10}")
    print(f"  {'─'*74}")
    print(f"  {'Epochs to converge':<40} {p1.epochs_trained:>12} {p3.epochs_trained:>12} {p3.epochs_trained - p1.epochs_trained:>+10}")
    print(f"  {'Training time (s)':<40} {p1.training_time:>12.2f} {p3.training_time:>12.2f} {p3.training_time - p1.training_time:>+10.2f}")
    print(f"  {'Final train accuracy':<40} {p1.final_train_acc:>12.4f} {p3.final_train_acc:>12.4f} {p3.final_train_acc - p1.final_train_acc:>+10.4f}")
    print(f"  {'Final val accuracy':<40} {p1.final_val_acc:>12.4f} {p3.final_val_acc:>12.4f} {p3.final_val_acc - p1.final_val_acc:>+10.4f}")
    print(f"  {'Best val accuracy':<40} {p1.best_val_acc:>12.4f} {p3.best_val_acc:>12.4f} {p3.best_val_acc - p1.best_val_acc:>+10.4f}")

    # ---- Weight similarity ----
    print(f"\n  Weight Comparison (Phase 1 final W* vs Phase 3 final W'):")
    print(f"    Cosine similarity:    {cos_sim:.6f}")
    print(f"    L2 distance:          {l2_dist:.4f}")
    print(f"    Relative L2 distance: {rel_l2:.4f}")
    print(f"    ‖W*‖ = {w1_norm:.4f},  ‖W'‖ = {w3_norm:.4f}")

    if layer_comparisons:
        print(f"\n  Per-layer weight cosine similarity:")
        for layer_idx, lc, ld in layer_comparisons:
            print(f"    Layer {layer_idx}: cos = {lc:.6f}, L2 = {ld:.4f}")

    # ---- Activation training impact ----
    print(f"\n  Activation Training Impact (Phase 2):")
    print(f"    Epochs used: {p2.epochs_trained}")
    print(f"    Val accuracy before: {p1.final_val_acc:.4f} → after: {p2.final_val_acc:.4f} (Δ = {p2.final_val_acc - p1.final_val_acc:+.4f})")

    # ---- Loss trajectory comparison ----
    # Compare loss at epoch 5, 10, 20 (if available)
    print(f"\n  Training Loss Trajectory Comparison:")
    print(f"    {'Epoch':<10} {'Phase 1 Loss':>14} {'Phase 3 Loss':>14} {'Δ':>10}")
    print(f"    {'─'*48}")
    checkpoints = [1, 5, 10, 20, 30, 40, 50]
    for cp in checkpoints:
        if cp <= len(p1.train_losses) and cp <= len(p3.train_losses):
            l1 = p1.train_losses[cp - 1]
            l3 = p3.train_losses[cp - 1]
            print(f"    {cp:<10} {l1:>14.4f} {l3:>14.4f} {l3 - l1:>+10.4f}")

    # ==================================================================
    # ANSWER THE RESEARCH QUESTIONS
    # ==================================================================
    print(f"\n{'='*70}")
    print("  ANSWERS TO RESEARCH QUESTIONS")
    print(f"{'='*70}")

    # Q1: Will weights be different?
    print(f"\n  Q1: Will the final weights be different?")
    if cos_sim > 0.99:
        print(f"      → Nearly identical (cosine similarity = {cos_sim:.6f}).")
        print(f"        The pretrained activations did NOT change the learned weight direction.")
    elif cos_sim > 0.90:
        print(f"      → Somewhat similar (cosine similarity = {cos_sim:.6f}).")
        print(f"        Weights share a similar direction but diverge noticeably.")
    else:
        print(f"      → YES, weights are substantially different (cosine similarity = {cos_sim:.6f}).")
        print(f"        The pretrained activation landscape leads to a different")
        print(f"        optimum in weight space, as expected from optimizing under")
        print(f"        a different (learned) activation function.")

    # Q2: Did weight training finish earlier?
    print(f"\n  Q2: Did weight training finish earlier with pretrained activations?")
    epoch_diff = p1.epochs_trained - p3.epochs_trained
    time_diff = p1.training_time - p3.training_time
    if epoch_diff > 0:
        print(f"      → YES. Phase 3 converged {epoch_diff} epochs earlier "
              f"({p3.epochs_trained} vs {p1.epochs_trained}).")
        print(f"        Time saved: {time_diff:.2f}s "
              f"({time_diff/p1.training_time*100:.1f}% faster)." if p1.training_time > 0 else "")
    elif epoch_diff == 0:
        print(f"      → NO difference. Both converged in {p1.epochs_trained} epochs.")
        if time_diff > 0.1:
            print(f"        However, Phase 3 was {time_diff:.2f}s faster in wall-clock time.")
    else:
        print(f"      → NO. Phase 3 took {-epoch_diff} MORE epochs "
              f"({p3.epochs_trained} vs {p1.epochs_trained}).")
        print(f"        The pretrained activations did not reduce convergence time here.")

    # Additional insight: did pretrained activations help accuracy?
    acc_diff = p3.best_val_acc - p1.best_val_acc
    print(f"\n  Additional insight:")
    if acc_diff > 0.005:
        print(f"    Pretrained activations IMPROVED accuracy by {acc_diff:+.4f}.")
    elif acc_diff < -0.005:
        print(f"    Pretrained activations REDUCED accuracy by {acc_diff:+.4f}.")
    else:
        print(f"    Accuracy is essentially unchanged (Δ = {acc_diff:+.4f}).")

    # Summary
    faster = "FASTER" if epoch_diff > 0 else "SLOWER" if epoch_diff < 0 else "SAME SPEED"
    better = "BETTER" if acc_diff > 0.005 else "WORSE" if acc_diff < -0.005 else "EQUIVALENT"
    print(f"\n  Summary: Pretrained activations → {faster} convergence, {better} accuracy.")
    print()


# ============================================================================
# Multi-seed runner for statistical significance
# ============================================================================

@dataclass
class AggregatedPhaseResult:
    """Aggregated results across multiple seeds."""
    name: str
    epochs: List[int] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)
    best_val_accs: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)

    @property
    def epochs_mean(self): return float(np.mean(self.epochs))
    @property
    def epochs_std(self): return float(np.std(self.epochs))
    @property
    def time_mean(self): return float(np.mean(self.times))
    @property
    def time_std(self): return float(np.std(self.times))
    @property
    def val_acc_mean(self): return float(np.mean(self.val_accs))
    @property
    def val_acc_std(self): return float(np.std(self.val_accs))
    @property
    def best_val_mean(self): return float(np.mean(self.best_val_accs))
    @property
    def best_val_std(self): return float(np.std(self.best_val_accs))
    @property
    def train_acc_mean(self): return float(np.mean(self.train_accs))
    @property
    def train_acc_std(self): return float(np.std(self.train_accs))


def run_experiment_suite(
    dataset_type: str,
    hidden_dims: List[int],
    epochs: int,
    activation_epochs: int,
    batch_size: int,
    learning_rate: float,
    seeds: List[int],
    patience: int = 10,
    per_neuron: bool = True,
) -> Dict[str, AggregatedPhaseResult]:
    """
    Run the 3-phase experiment with multiple seeds on one dataset.

    Loads data once (using first seed for train/test split), then
    iterates over all seeds for weight initialisation only.
    """
    import time as _time

    # Load data once
    print(f"Loading {dataset_type} dataset...")
    config = DatasetConfig(dataset_type=dataset_type, random_state=seeds[0])
    dm = DataManager(config)
    X_train, X_val, y_train, y_val = dm.generate_dataset()
    print(f"  Training: {len(X_train):,} | Test: {len(X_val):,} | "
          f"Features: {X_train.shape[1]} | Classes: {len(np.unique(y_train))}")

    # Accumulators
    agg = {
        "Phase 1 (Fixed ReLU)": AggregatedPhaseResult("Phase 1 (Fixed ReLU)"),
        "Phase 2 (Activation Training)": AggregatedPhaseResult("Phase 2 (Activation Training)"),
        "Phase 3 (Pretrained Act.)": AggregatedPhaseResult("Phase 3 (Pretrained Act.)"),
    }

    for i, seed in enumerate(seeds):
        print(f"\n  Seed {i+1}/{len(seeds)} (seed={seed})...")

        results = run_experiment(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hidden_dims=hidden_dims,
            weight_lr=learning_rate,
            activation_lr=learning_rate,
            weight_epochs=epochs,
            activation_epochs=activation_epochs,
            batch_size=batch_size,
            patience=patience,
            per_neuron=per_neuron,
            seed=seed,
            verbose=False,
        )

        for phase_key, agg_key in [
            ('phase1', "Phase 1 (Fixed ReLU)"),
            ('phase2', "Phase 2 (Activation Training)"),
            ('phase3', "Phase 3 (Pretrained Act.)"),
        ]:
            p = results[phase_key]
            agg[agg_key].epochs.append(p.epochs_trained)
            agg[agg_key].times.append(p.training_time)
            agg[agg_key].val_accs.append(p.final_val_acc)
            agg[agg_key].best_val_accs.append(p.best_val_acc)
            agg[agg_key].train_accs.append(p.final_train_acc)

        p1, p3 = results['phase1'], results['phase3']
        print(f"    P1: {p1.epochs_trained} ep, {p1.training_time:.1f}s, "
              f"val={p1.final_val_acc:.4f}  |  "
              f"P3: {p3.epochs_trained} ep, {p3.training_time:.1f}s, "
              f"val={p3.final_val_acc:.4f}  "
              f"(Δ={p3.final_val_acc - p1.final_val_acc:+.4f})")

    return agg


def print_suite_results(
    agg: Dict[str, AggregatedPhaseResult],
    dataset_name: str,
    n_seeds: int,
) -> None:
    """Print comparison table matching the notebook format."""
    print("\n" + "=" * 100)
    print(f"EXPERIMENT RESULTS: {dataset_name.upper()} ({n_seeds} seeds)")
    print("=" * 100)

    headers = ["Phase", "Train Acc", "Test Acc", "Time (s)", "Epochs"]
    row_fmt = "{:<45} {:>18} {:>18} {:>10} {:>8}"
    print(row_fmt.format(*headers))
    print("-" * 100)

    for name, r in agg.items():
        print(row_fmt.format(
            name[:45],
            f"{r.train_acc_mean:.4f} ± {r.train_acc_std:.4f}",
            f"{r.val_acc_mean:.4f} ± {r.val_acc_std:.4f}",
            f"{r.time_mean:.2f}",
            f"{r.epochs_mean:.1f}",
        ))
    print("-" * 100)

    p1 = agg["Phase 1 (Fixed ReLU)"]
    p3 = agg["Phase 3 (Pretrained Act.)"]
    acc_diff = p3.val_acc_mean - p1.val_acc_mean
    epoch_diff = p1.epochs_mean - p3.epochs_mean

    print(f"\n  Phase 1 vs Phase 3:")
    print(f"    Accuracy Δ:  {acc_diff:+.4f}")
    print(f"    Epoch Δ:     {epoch_diff:+.1f} epochs saved")

    # Paired t-test on epochs and accuracy
    try:
        from scipy import stats as sp_stats
        t_ep, p_ep = sp_stats.ttest_rel(p1.epochs, p3.epochs)
        t_acc, p_acc = sp_stats.ttest_rel(p1.val_accs, p3.val_accs)
        print(f"    Epochs  t-test: t={t_ep:.3f}, p={p_ep:.4f} {'*' if p_ep < 0.05 else ''}")
        print(f"    Acc     t-test: t={t_acc:.3f}, p={p_acc:.4f} {'*' if p_acc < 0.05 else ''}")
    except ImportError:
        pass


def save_suite_results(
    all_results: Dict[str, Dict[str, AggregatedPhaseResult]],
    filepath: str,
    n_seeds: int,
) -> None:
    """Save aggregated results to CSV."""
    import pandas as pd
    from datetime import datetime

    rows = []
    for dataset_name, agg in all_results.items():
        for phase_name, r in agg.items():
            rows.append({
                "dataset": dataset_name,
                "phase": phase_name,
                "n_seeds": n_seeds,
                "train_acc_mean": r.train_acc_mean,
                "train_acc_std": r.train_acc_std,
                "test_acc_mean": r.val_acc_mean,
                "test_acc_std": r.val_acc_std,
                "epochs_mean": r.epochs_mean,
                "epochs_std": r.epochs_std,
                "time_mean": r.time_mean,
                "time_std": r.time_std,
                "timestamp": datetime.now().isoformat(),
            })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"\nResults saved to: {filepath}")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    # ── Configuration (matching notebook conventions) ──────────────
    N_SEEDS = 20
    SEEDS = [42, 123, 456, 789, 1001, 1111, 2222, 3333, 4444, 5555,
             6666, 7777, 8888, 9999, 1010, 2020, 3030, 4040, 5050, 6060]
    HIDDEN_DIMS = [256, 128]
    EPOCHS = 30
    ACTIVATION_EPOCHS = 30
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01
    PATIENCE = 10

    # DATASETS = ['mnist', 'fashion_mnist', 'cifar10']
    DATASETS = ['cifar10']

    print("=" * 100)
    print("  PRETRAINED ACTIVATION EXPERIMENT")
    print("  Question: Does training activations on good weights, then training")
    print("            random weights with those activations, reduce training time?")
    print("=" * 100)
    print(f"  Seeds: {N_SEEDS} | Architecture: {HIDDEN_DIMS} | Epochs: {EPOCHS}")
    print(f"  Activation epochs: {ACTIVATION_EPOCHS} | LR: {LEARNING_RATE}")
    print(f"  Datasets: {DATASETS}")

    all_results = {}

    for dataset in DATASETS:
        print(f"\n{'#'*100}")
        print(f"  DATASET: {dataset.upper()}")
        print(f"{'#'*100}")

        agg = run_experiment_suite(
            dataset_type=dataset,
            hidden_dims=HIDDEN_DIMS,
            epochs=EPOCHS,
            activation_epochs=ACTIVATION_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            seeds=SEEDS,
            patience=PATIENCE,
            per_neuron=True,
        )
        all_results[dataset] = agg
        print_suite_results(agg, dataset, N_SEEDS)

    # ── Final summary ─────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"FINAL SUMMARY (Mean ± Std over {N_SEEDS} seeds)")
    print("=" * 100)

    for dataset_name, agg in all_results.items():
        p1 = agg["Phase 1 (Fixed ReLU)"]
        p3 = agg["Phase 3 (Pretrained Act.)"]
        acc_diff = p3.val_acc_mean - p1.val_acc_mean
        epoch_diff = p1.epochs_mean - p3.epochs_mean

        print(f"\n📊 {dataset_name.upper()}:")
        print(f"   Phase 1 (Fixed ReLU):      {p1.val_acc_mean:.4f} ± {p1.val_acc_std:.4f}  ({p1.epochs_mean:.1f} epochs)")
        print(f"   Phase 3 (Pretrained Act.):  {p3.val_acc_mean:.4f} ± {p3.val_acc_std:.4f}  ({p3.epochs_mean:.1f} epochs)")
        print(f"   Accuracy Δ: {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")
        print(f"   Epoch Δ:    {epoch_diff:+.1f} epochs")

    # ── Answers ────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("ANSWERS TO RESEARCH QUESTIONS")
    print("=" * 100)

    # Aggregate across all datasets
    all_acc_diffs = []
    all_epoch_diffs = []
    for agg in all_results.values():
        p1, p3 = agg["Phase 1 (Fixed ReLU)"], agg["Phase 3 (Pretrained Act.)"]
        all_acc_diffs.append(p3.val_acc_mean - p1.val_acc_mean)
        all_epoch_diffs.append(p1.epochs_mean - p3.epochs_mean)

    mean_acc_diff = float(np.mean(all_acc_diffs))
    mean_epoch_diff = float(np.mean(all_epoch_diffs))

    print(f"\n  Q1: Will weights be different?")
    print(f"      → Weights are expected to be nearly identical (same loss landscape")
    print(f"        with negligible activation perturbation from DynamicReLU(a≈0, b≈1)).")

    print(f"\n  Q2: Did weight training finish earlier with pretrained activations?")
    if mean_epoch_diff > 1:
        print(f"      → YES, on average {mean_epoch_diff:.1f} fewer epochs across datasets.")
    elif mean_epoch_diff < -1:
        print(f"      → NO, pretrained activations took {-mean_epoch_diff:.1f} MORE epochs.")
    else:
        print(f"      → NO significant difference ({mean_epoch_diff:+.1f} epochs on average).")
        print(f"        Pretrained activations do NOT meaningfully reduce training time.")

    print(f"\n  Overall accuracy change: {mean_acc_diff:+.4f} (averaged across datasets).")
    if abs(mean_acc_diff) < 0.005:
        print(f"  Conclusion: Pretrained activations provide NEGLIGIBLE benefit.")
    print()

    # ── Save CSV ──────────────────────────────────────────────────
    save_suite_results(all_results, "pretrained_activation_results.csv", N_SEEDS)
