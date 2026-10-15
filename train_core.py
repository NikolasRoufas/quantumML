"""
Generalized VQC training loop — same optimizer/lr/batch-size/epochs/loss/
prediction-rule as the original quantum_trainer.train_quantum_model, but
parameterized over an arbitrary (circuit, weight_shape) pair and seeded per
run, with per-epoch gradient diagnostics added (Section 6) without changing
the optimization procedure itself (same losses feed the same optimizer.step
calls in the same order; the diagnostics only read weights.grad).
"""
import time
import numpy as np
import torch
import torch.nn.functional as F

import utils, circuits as rcircuits, metrics as rmetrics


def train_one_run(feature_map, ansatz, n_qubits, n_layers, seed,
                   X_train, y_train, X_test, y_test,
                   lr, batch_size, n_epochs, log_every=5, verbose=True):
    """Train one VQC configuration for one seed. Returns a fully-populated
    result dict (Section 5's required per-run fields)."""
    utils.set_all_seeds(seed)

    circuit, weight_shape = rcircuits.build_circuit(
        feature_map, ansatz, n_qubits=n_qubits, n_layers=n_layers
    )
    n_params = int(np.prod(weight_shape))

    # Same init convention as the original quantum_circuits.init_weights:
    # uniform[-pi, pi], drawn from the global torch RNG that set_all_seeds
    # just seeded — reproducible per-seed, identical protocol to the original.
    weights = torch.nn.Parameter(
        (torch.rand(*weight_shape, dtype=torch.float64) * 2 - 1) * np.pi
    )
    optimizer = torch.optim.Adam([weights], lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float64)
    y_t = torch.tensor(y_train, dtype=torch.float64)
    n_batches = len(X_t) // batch_size

    loss_history = []
    grad_l2_history, grad_mean_abs_history, grad_var_history = [], [], []

    t0 = time.time()
    for epoch in range(n_epochs):
        perm = torch.randperm(len(X_t))
        X_s, y_s = X_t[perm], y_t[perm]

        epoch_loss = 0.0
        ep_l2, ep_mean_abs, ep_var = [], [], []
        for b in range(n_batches):
            Xb = X_s[b * batch_size:(b + 1) * batch_size]
            yb = y_s[b * batch_size:(b + 1) * batch_size]

            optimizer.zero_grad()
            # Batched circuit evaluation via PennyLane parameter broadcasting:
            # mathematically identical to evaluating each sample separately
            # and stacking (verified: max forward diff 0.0, max grad diff
            # ~1e-15 against the per-sample loop) — ~4-5x fewer wall-clock
            # seconds per batch from reduced per-call tape-building overhead.
            raw = circuit(Xb, weights)
            probs = (raw + 1.0) / 2.0
            loss = F.binary_cross_entropy(probs.float(), yb.float())
            loss.backward()

            g = weights.grad.detach().cpu().numpy().flatten()
            ep_l2.append(float(np.linalg.norm(g)))
            ep_mean_abs.append(float(np.mean(np.abs(g))))
            ep_var.append(float(np.var(g)))

            optimizer.step()
            epoch_loss += loss.item()

        loss_history.append(epoch_loss / n_batches)
        grad_l2_history.append(float(np.mean(ep_l2)))
        grad_mean_abs_history.append(float(np.mean(ep_mean_abs)))
        grad_var_history.append(float(np.mean(ep_var)))

        if verbose and (epoch + 1) % log_every == 0:
            print(f"    [{feature_map}+{ansatz} seed={seed}] "
                  f"epoch {epoch+1}/{n_epochs} loss={loss_history[-1]:.4f}", flush=True)

    elapsed = time.time() - t0

    def _predict(X_np):
        w = weights.detach()
        X_te_t = torch.tensor(X_np, dtype=torch.float64)
        with torch.no_grad():
            raw = circuit(X_te_t, w)  # batched, same equivalence as above
        return ((raw + 1.0) / 2.0).numpy()

    p_tr = _predict(X_train)
    p_te = _predict(X_test)
    y_pred_tr = (p_tr > 0.5).astype(int)
    y_pred_te = (p_te > 0.5).astype(int)

    train_metrics = rmetrics.compute_full_metrics(y_train, y_pred_tr, p_tr)
    test_metrics = rmetrics.compute_full_metrics(y_test, y_pred_te, p_te)

    gc = rcircuits.gate_count_and_depth(feature_map, ansatz, n_qubits, n_layers,
                                         x=np.asarray(X_train[0]))

    result = {
        "feature_map": feature_map,
        "ansatz": ansatz,
        "seed": int(seed),
        "n_qubits": int(n_qubits),
        "n_layers": int(n_layers),
        "trainable_parameters": n_params,
        "gate_count": gc["gate_count"],
        "circuit_depth": gc["depth"],
        "gate_types": gc["gate_types"],
        "training_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "roc_auc": test_metrics["roc_auc"],
        "pr_auc": test_metrics["pr_auc"],
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "mcc": test_metrics["mcc"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "final_train_loss": loss_history[-1],
        "minimum_train_loss": float(np.min(loss_history)),
        "loss_history": loss_history,
        "grad_l2_norm_history": grad_l2_history,
        "grad_mean_abs_history": grad_mean_abs_history,
        "grad_variance_history": grad_var_history,
        "test_probabilities": p_te.tolist(),
        "train_probabilities": p_tr.tolist(),
        "elapsed_seconds": elapsed,
        "lr": lr, "batch_size": batch_size, "n_epochs": n_epochs,
    }
    return result
