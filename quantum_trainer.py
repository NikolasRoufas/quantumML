import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import quantum_circuits as qc
import config


def _batch_predict(circuit, X_np, weights):
    """Forward-only pass for evaluation — no gradient tape."""
    w = weights.detach()
    X_t = torch.tensor(X_np, dtype=torch.float64)
    preds = []
    with torch.no_grad():
        for i in range(len(X_t)):
            out = circuit(X_t[i], w)
            preds.append(float(out))
    return np.array(preds)


def train_quantum_model(circuit, weight_shape, X_train, y_train, X_test, y_test, model_name=""):
    print(f"\n  [{model_name}]", flush=True)
    print(f"  Weight shape: {weight_shape} | Params: {int(np.prod(weight_shape))}", flush=True)

    weights = qc.init_weights(weight_shape)
    optimizer = torch.optim.Adam([weights], lr=config.LR)

    X_t = torch.tensor(X_train, dtype=torch.float64)
    y_t = torch.tensor(y_train, dtype=torch.float64)

    n_batches = len(X_train) // config.BATCH_SIZE
    history = {"loss": []}

    for epoch in range(config.N_EPOCHS):
        perm = torch.randperm(len(X_t))
        X_s, y_s = X_t[perm], y_t[perm]

        epoch_loss = 0.0
        for b in range(n_batches):
            Xb = X_s[b * config.BATCH_SIZE : (b + 1) * config.BATCH_SIZE]
            yb = y_s[b * config.BATCH_SIZE : (b + 1) * config.BATCH_SIZE]

            optimizer.zero_grad()
            raw = torch.stack([circuit(Xb[i], weights) for i in range(len(Xb))])
            probs = (raw + 1.0) / 2.0
            loss = F.binary_cross_entropy(probs.float(), yb.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        history["loss"].append(avg_loss)

        if (epoch + 1) % 5 == 0:
            probs_te = _batch_predict(circuit, X_test, weights)
            acc_te = accuracy_score(y_test, (probs_te > 0.5).astype(int))
            print(f"    Epoch {epoch+1:3d}/{config.N_EPOCHS} | "
                  f"Loss: {avg_loss:.4f} | Val Acc: {acc_te:.4f}", flush=True)

    probs_tr = _batch_predict(circuit, X_train, weights)
    probs_te = _batch_predict(circuit, X_test,  weights)
    # Map raw expectation values [-1,1] to probabilities [0,1]
    p_tr = (probs_tr + 1.0) / 2.0
    p_te = (probs_te + 1.0) / 2.0
    train_acc = accuracy_score(y_train, (p_tr > 0.5).astype(int))
    test_acc  = accuracy_score(y_test,  (p_te > 0.5).astype(int))
    print(f"    Final → Train: {train_acc:.4f} | Test: {test_acc:.4f}", flush=True)

    # Return probabilities so callers can compute AUC-ROC etc.
    return weights, history, train_acc, test_acc, p_te


def predict_proba(circuit, X_np, weights):
    """Public wrapper — returns probabilities in [0, 1]."""
    raw = _batch_predict(circuit, X_np, weights)
    return (raw + 1.0) / 2.0
