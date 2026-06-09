"""
Quantum Kernel SVM experiment.

Each feature map (ZZ, Amplitude, Angle) defines an implicit kernel:
    K(x_i, x_j) = |<φ(x_i)|φ(x_j)>|²
where |φ(x)> is the quantum state produced by the feature map alone (no ansatz).
An SVC with this precomputed kernel is then trained and compared against
classical kernels and variational quantum classifiers.
"""
import math
import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import config
from extended_metrics import compute_metrics

N = config.N_QUBITS

# ── state-producing QNodes (no ansatz, feature map only) ──────────────────────

_dev = qml.device("default.qubit", wires=N)


@qml.qnode(_dev)
def _zz_state(x):
    for i in range(N):
        qml.Hadamard(wires=i)
    for i in range(N):
        qml.RZ(2.0 * x[i], wires=i)
    for i in range(N):
        for j in range(i + 1, N):
            qml.CNOT(wires=[i, j])
            qml.RZ(2.0 * math.pi * x[i] * x[j], wires=j)
            qml.CNOT(wires=[i, j])
    return qml.state()


@qml.qnode(_dev)
def _amp_state(x):
    qml.AmplitudeEmbedding(features=x, wires=range(N), normalize=True, pad_with=0.0)
    return qml.state()


@qml.qnode(_dev)
def _angle_state(x):
    for i in range(N):
        qml.RX(x[i], wires=i)
        qml.RY(x[i], wires=i)
    return qml.state()


# ── kernel matrix ─────────────────────────────────────────────────────────────

def _fidelity_kernel_matrix(states_a, states_b):
    """K[i,j] = |<ψ_a[i]|ψ_b[j]>|²"""
    A = np.array(states_a)  # (n_a, 2^N)
    B = np.array(states_b)  # (n_b, 2^N)
    overlap = np.abs(A.conj() @ B.T) ** 2
    return overlap.real


# ── main experiment ───────────────────────────────────────────────────────────

def run_quantum_kernel_svms(X_tr, X_te, y_tr, y_te, n_train=300, n_test=150):
    """
    Subsample the quantum dataset, compute state vectors once,
    then build kernel matrices and fit SVMs.
    Returns dict: model_name → metrics dict.
    """
    rng = np.random.RandomState(config.SEED)
    tr_idx = rng.choice(len(X_tr), min(n_train, len(X_tr)), replace=False)
    te_idx = rng.choice(len(X_te), min(n_test,  len(X_te)),  replace=False)
    Xk_tr, yk_tr = X_tr[tr_idx], y_tr[tr_idx]
    Xk_te, yk_te = X_te[te_idx], y_te[te_idx]

    state_fns = {
        "QKernel-ZZ":        _zz_state,
        "QKernel-Amplitude": _amp_state,
        "QKernel-Angle":     _angle_state,
    }

    results = {}
    for name, sfn in state_fns.items():
        print(f"    {name}...", end=" ", flush=True)

        tr_states = [np.array(sfn(x)) for x in Xk_tr]
        te_states = [np.array(sfn(x)) for x in Xk_te]

        K_tr = _fidelity_kernel_matrix(tr_states, tr_states)
        K_te = _fidelity_kernel_matrix(te_states, tr_states)

        svm = SVC(kernel="precomputed", C=1.0, probability=True)
        svm.fit(K_tr, yk_tr)

        y_pred  = svm.predict(K_te)
        y_proba = svm.predict_proba(K_te)[:, 1]

        m = compute_metrics(yk_te, y_pred, y_proba)
        results[name] = m
        print(f"acc={m['accuracy']:.4f}  AUC={m.get('auc_roc', float('nan')):.4f}", flush=True)

    return results, yk_te
