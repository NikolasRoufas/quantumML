"""
Analyses from the methodology + expressibility (Sim et al. 2019):
  1. Circuit complexity (gate count, depth, parameter count)
  2. Barren plateau — gradient norm vs circuit depth
  3. Feature map fidelity — same-class vs different-class quantum state overlap
  4. Expressibility — KL divergence to Haar-random fidelity distribution
"""
import math
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import config

N = config.N_QUBITS


# ── 1. Circuit complexity ──────────────────────────────────────────────────────

def compute_circuit_complexity():
    """
    Analytic gate counts matching Table 2 in the methodology.
    Returns dict: name → (gate_count, depth, param_count).
    """
    n, L = N, config.N_LAYERS

    # ZZ feature map gates:
    #   n Hadamard + n RZ + 2*C(n,2) CNOT + C(n,2) RZ
    n_pairs = n * (n - 1) // 2          # 6 for n=4
    zz_gates = n + n + 2 * n_pairs + n_pairs   # H + RZ_data + 2*CNOT + RZ_interact

    # Strongly entangling: L * (n * Rot[3 gates] + n CNOT)
    se_gates = L * (n * 3 + n)
    # Depth: ZZ block + L * (single-qubit + entangling)
    zz_depth = 1 + 1 + 2 * n_pairs + L * (3 + 1)
    zz_params = L * n * 3

    # Amplitude embedding (state preparation): 2^n - 1 gates
    amp_gates = (2 ** n - 1) + L * (n + n)   # prep + L*(RY + CNOT)
    amp_depth = n + L * 2
    amp_params = L * n

    # Angle embedding: 2n (RX+RY per qubit)
    angle_gates = 2 * n + L * (n * 3 + n + n)  # embed + L*(RX+RY+RZ + CNOT + CRZ)
    angle_depth = 2 + L * (3 + 1 + 1)
    angle_params = L * n * 4

    return {
        "ZZ + StronglyEnt (4q, 2L)":       (zz_gates    + se_gates, zz_depth,  zz_params),
        "Amplitude + BasicEnt (4q, 2L)":   (amp_gates,               amp_depth, amp_params),
        "Angle + CustomAnsatz (4q, 2L)":   (angle_gates,             angle_depth, angle_params),
    }


# ── 2. Barren plateau ─────────────────────────────────────────────────────────

_dev_bp = qml.device("default.qubit", wires=N)


def _make_zz_se_circuit(n_layers):
    @qml.qnode(_dev_bp, interface="autograd", diff_method="backprop")
    def circ(x, weights):
        for i in range(N):
            qml.Hadamard(wires=i)
        for i in range(N):
            qml.RZ(2.0 * x[i], wires=i)
        for i in range(N):
            for j in range(i + 1, N):
                qml.CNOT(wires=[i, j])
                qml.RZ(2.0 * math.pi * x[i] * x[j], wires=j)
                qml.CNOT(wires=[i, j])
        qml.StronglyEntanglingLayers(weights, wires=range(N))
        return qml.expval(qml.PauliZ(0))
    return circ


def barren_plateau_analysis(depths=None, n_init=30):
    """
    For each depth, sample n_init random initialisations and record gradient norms.
    Returns dict: depth → {'mean': float, 'std': float}.
    """
    if depths is None:
        depths = [1, 2, 3, 4, 5, 6]

    stats = {}
    for depth in depths:
        print(f"    depth {depth}...", end=" ", flush=True)
        circ = _make_zz_se_circuit(depth)
        grad_norms = []

        for _ in range(n_init):
            x = pnp.array(np.random.uniform(-1.0, 1.0, N), requires_grad=False)
            w = pnp.array(
                np.random.uniform(-math.pi, math.pi, (depth, N, 3)),
                requires_grad=True,
            )
            grad_fn = qml.grad(circ, argnums=1)
            grads = grad_fn(x, w)
            grad_norms.append(float(np.linalg.norm(grads)))

        stats[depth] = {
            "mean": float(np.mean(grad_norms)),
            "std":  float(np.std(grad_norms)),
        }
        print(f"mean={stats[depth]['mean']:.4f} ± {stats[depth]['std']:.4f}", flush=True)

    return stats


# ── 3. Feature map fidelity ───────────────────────────────────────────────────

_dev_fid = qml.device("default.qubit", wires=N)


@qml.qnode(_dev_fid)
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


@qml.qnode(_dev_fid)
def _amp_state(x):
    qml.AmplitudeEmbedding(features=x, wires=range(N), normalize=True, pad_with=0.0)
    return qml.state()


@qml.qnode(_dev_fid)
def _angle_state(x):
    for i in range(N):
        qml.RX(x[i], wires=i)
        qml.RY(x[i], wires=i)
    return qml.state()


def compute_fidelities(X, y, n_samples=200, n_pairs=400):
    """
    Compute |<ψ(x_i)|ψ(x_j)>|² for same-class and different-class pairs
    under each feature map.
    """
    X_sub = X[:n_samples]
    y_sub = y[:n_samples]

    state_fns = {
        "ZZ":        _zz_state,
        "Amplitude": _amp_state,
        "Angle":     _angle_state,
    }

    fidelities = {}
    for fname, sfn in state_fns.items():
        print(f"    {fname} feature map...", end=" ", flush=True)
        states = np.array([np.array(sfn(x)) for x in X_sub])

        rng = np.random.RandomState(config.SEED)
        same_fid, diff_fid = [], []

        for _ in range(n_pairs):
            i, j = rng.choice(len(states), 2, replace=False)
            fid = float(np.abs(np.dot(np.conj(states[i]), states[j])) ** 2)
            if y_sub[i] == y_sub[j]:
                same_fid.append(fid)
            else:
                diff_fid.append(fid)

        fidelities[fname] = {"same": same_fid, "diff": diff_fid}
        same_mean = np.mean(same_fid)
        diff_mean = np.mean(diff_fid)
        print(f"same={same_mean:.3f}, diff={diff_mean:.3f}", flush=True)

    return fidelities


# ── 4. Expressibility (Sim et al. 2019) ──────────────────────────────────────

_dev_expr = qml.device("default.qubit", wires=N)


def _haar_fidelity_pdf(f_vals, n_qubits):
    """Analytical Haar-random fidelity distribution: P(F) = (2^n-1)(1-F)^(2^n-2)."""
    d = 2 ** n_qubits
    return (d - 1) * (1 - f_vals) ** (d - 2)


def _make_state_circuit(feature_map_fn, ansatz_weight_shape):
    """Return a QNode that gives the full statevector after feature map + ansatz."""
    @qml.qnode(_dev_expr)
    def circ(x, weights):
        feature_map_fn(x)
        qml.StronglyEntanglingLayers(weights, wires=range(N))
        return qml.state()
    return circ


def compute_expressibility(n_samples=200):
    """
    For each ansatz, sample random (θ_1, θ_2) pairs, compute fidelity
    |<ψ(θ_1)|ψ(θ_2)>|², then compute KL divergence from Haar distribution.
    Lower KL divergence = higher expressibility.
    Returns dict: ansatz_name → {'kl': float, 'fidelities': list}.
    """
    results = {}
    configs = {
        "ZZ + StronglyEnt":    (config.N_LAYERS, N, 3),
        "Amplitude + BasicEnt": (config.N_LAYERS, N),
        "Angle + CustomAnsatz": (config.N_LAYERS, N, 4),
    }

    bins = np.linspace(0, 1, 75)
    bin_centres = (bins[:-1] + bins[1:]) / 2
    eps = 1e-10

    haar_pdf = _haar_fidelity_pdf(bin_centres, N)
    haar_pdf = haar_pdf / haar_pdf.sum() + eps

    for name, w_shape in configs.items():
        print(f"    {name}...", end=" ", flush=True)
        dev_local = qml.device("default.qubit", wires=N)

        @qml.qnode(dev_local)
        def state_circ(x, weights):
            # generic circuit: ZZ feature map + StronglyEntanglingLayers as proxy
            for i in range(N):
                qml.Hadamard(wires=i)
            for i in range(N):
                qml.RZ(2.0 * x[i], wires=i)
            qml.StronglyEntanglingLayers(
                weights.reshape(config.N_LAYERS, N, 3), wires=range(N)
            )
            return qml.state()

        # Proxy circuit always uses StronglyEntanglingLayers shape (L, N, 3)
        proxy_params = config.N_LAYERS * N * 3
        fidelities = []

        for _ in range(n_samples):
            x  = np.random.uniform(-1.0, 1.0, N)
            w1 = np.random.uniform(-math.pi, math.pi, proxy_params)
            w2 = np.random.uniform(-math.pi, math.pi, proxy_params)
            sv1 = np.array(state_circ(x, w1.reshape(config.N_LAYERS, N, 3)))
            sv2 = np.array(state_circ(x, w2.reshape(config.N_LAYERS, N, 3)))
            fid = float(np.abs(np.dot(np.conj(sv1), sv2)) ** 2)
            fidelities.append(fid)

        sampled_pdf, _ = np.histogram(fidelities, bins=bins, density=False)
        sampled_pdf = sampled_pdf / sampled_pdf.sum() + eps

        kl = float(np.sum(sampled_pdf * np.log(sampled_pdf / haar_pdf)))
        results[name] = {"kl": kl, "fidelities": fidelities}
        print(f"KL={kl:.4f}", flush=True)

    return results
