"""
Qubit scalability experiment.

Trains a ZZ + Strongly-Entangling circuit for n_qubits in [2, 3, 4, 5],
each time reducing features via PCA to match the qubit count.
Uses a fixed 700-sample subsample and 20 epochs so the sweep stays fast.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import config


SCALE_EPOCHS   = 20
SCALE_SAMPLES  = 700
SCALE_BATCH    = 16
SCALE_N_LAYERS = 2


def _make_zz_se_circuit(n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x, weights):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(2.0 * x[i], wires=i)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])
                qml.RZ(2.0 * math.pi * x[i] * x[j], wires=j)
                qml.CNOT(wires=[i, j])
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit


def _prep_data(X_raw, y_raw, n_features):
    rng = np.random.RandomState(config.SEED)
    idx = rng.choice(len(X_raw), SCALE_SAMPLES, replace=False)
    X, y = X_raw[idx], y_raw[idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=config.SEED, stratify=y
    )
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)

    pca = PCA(n_components=n_features, random_state=config.SEED)
    X_tr_pca = pca.fit_transform(X_tr_sc)
    X_te_pca = pca.transform(X_te_sc)

    pca_sc = StandardScaler()
    X_tr_pca = np.clip(pca_sc.fit_transform(X_tr_pca) * (np.pi / 3.0), -np.pi, np.pi)
    X_te_pca = np.clip(pca_sc.transform(X_te_pca)     * (np.pi / 3.0), -np.pi, np.pi)

    return X_tr_pca, X_te_pca, y_tr, y_te


def run_scalability(X_raw, y_raw, qubit_range=None):
    if qubit_range is None:
        qubit_range = [2, 3, 4, 5]

    results = {}
    for n_q in qubit_range:
        print(f"    n_qubits={n_q} ({SCALE_EPOCHS} epochs)...", end=" ", flush=True)

        X_tr, X_te, y_tr, y_te = _prep_data(X_raw, y_raw, n_features=n_q)
        circuit = _make_zz_se_circuit(n_q)

        weight_shape = (SCALE_N_LAYERS, n_q, 3)
        weights = torch.nn.Parameter(
            torch.randn(*weight_shape, dtype=torch.float64) * 0.1
        )
        optimizer = torch.optim.Adam([weights], lr=config.LR)

        X_t = torch.tensor(X_tr, dtype=torch.float64)
        y_t = torch.tensor(y_tr, dtype=torch.float64)
        n_batches = len(X_t) // SCALE_BATCH

        for _ in range(SCALE_EPOCHS):
            perm = torch.randperm(len(X_t))
            X_s, y_s = X_t[perm], y_t[perm]
            for b in range(n_batches):
                Xb = X_s[b * SCALE_BATCH : (b + 1) * SCALE_BATCH]
                yb = y_s[b * SCALE_BATCH : (b + 1) * SCALE_BATCH]
                optimizer.zero_grad()
                raw = torch.stack([circuit(Xb[i], weights) for i in range(len(Xb))])
                loss = F.binary_cross_entropy((raw + 1.0) / 2.0, yb.double())
                loss.backward()
                optimizer.step()

        # Evaluate
        w = weights.detach()
        X_te_t = torch.tensor(X_te, dtype=torch.float64)
        with torch.no_grad():
            preds = [(circuit(X_te_t[i], w).item() + 1.0) / 2.0 for i in range(len(X_te_t))]
        y_pred = (np.array(preds) > 0.5).astype(int)
        acc = float(accuracy_score(y_te, y_pred))
        n_params = SCALE_N_LAYERS * n_q * 3

        results[n_q] = {"accuracy": acc, "n_params": n_params}
        print(f"acc={acc:.4f}  params={n_params}", flush=True)

    return results
