"""
Section 3 — Full 3x3 factorial circuit construction.

Feature maps and ansätze are refactored into independent, composable
PennyLane sub-routines so any of the 3 feature maps can be paired with any
of the 3 ansätze. Every sub-routine preserves the *exact* gate sequence of
the original repository (quantum_circuits.py / analysis.py) — nothing about
the underlying math was changed, only how the pieces are assembled.
"""
import math
import numpy as np
import pennylane as qml

from revision import rconfig

_devices = {}


def get_device(n_qubits):
    if n_qubits not in _devices:
        _devices[n_qubits] = qml.device("default.qubit", wires=n_qubits)
    return _devices[n_qubits]


# ── feature maps (unchanged meaning from quantum_circuits.py / analysis.py) ──

def apply_zz(x, n_qubits):
    """Original ZZ feature map — Havlicek-style, unchanged. Uses x[..., i]
    rather than x[i] so the SAME code path works for a single sample
    (x.shape == (n_qubits,)) and a batched call (x.shape == (batch, n_qubits))
    under PennyLane parameter broadcasting — for a 1D x these are identical
    (x[..., i] == x[i]); for batched x, x[..., i] selects feature i across
    the whole batch instead of accidentally selecting sample i."""
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    for i in range(n_qubits):
        qml.RZ(2.0 * x[..., i], wires=i)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            qml.CNOT(wires=[i, j])
            qml.RZ(2.0 * math.pi * x[..., i] * x[..., j], wires=j)
            qml.CNOT(wires=[i, j])


def apply_amplitude(x, n_qubits):
    """Original amplitude embedding, unchanged parameters."""
    qml.AmplitudeEmbedding(features=x, wires=range(n_qubits), normalize=True, pad_with=0.0)


def apply_angle(x, n_qubits):
    """Original angle embedding: RX then RY per qubit, unchanged (see apply_zz
    docstring re: x[..., i] vs x[i])."""
    for i in range(n_qubits):
        qml.RX(x[..., i], wires=i)
        qml.RY(x[..., i], wires=i)


FEATURE_MAP_FNS = {
    "ZZ": apply_zz,
    "Amplitude": apply_amplitude,
    "Angle": apply_angle,
}


# ── ansätze (unchanged meaning from quantum_circuits.py) ─────────────────────

def apply_strongly_ent(weights, n_qubits):
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))


def apply_basic_ent(weights, n_qubits):
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=qml.RY)


def apply_custom_ansatz(weights, n_qubits):
    """Original custom ansatz (Eq. 7): RX RY RZ per qubit + CNOT ring + CRZ ring."""
    n_layers = weights.shape[0]
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
        for i in range(n_qubits):
            qml.CRZ(weights[l, i, 3], wires=[i, (i + 1) % n_qubits])


ANSATZ_FNS = {
    "StronglyEnt": apply_strongly_ent,
    "BasicEnt": apply_basic_ent,
    "CustomAnsatz": apply_custom_ansatz,
}


def weight_shape(ansatz, n_layers, n_qubits):
    return rconfig.WEIGHT_SHAPE_RULE[ansatz](n_layers, n_qubits)


def init_weights_np(ansatz, n_layers, n_qubits, rng):
    """Uniform[-pi, pi] init — same convention as quantum_circuits.init_weights."""
    shape = weight_shape(ansatz, n_layers, n_qubits)
    return rng.uniform(-math.pi, math.pi, size=shape)


# ── circuit builder ───────────────────────────────────────────────────────

def build_circuit(feature_map, ansatz, n_qubits=None, n_layers=None,
                   interface="torch", diff_method="backprop", return_state=False):
    """Return (qnode, weight_shape) for a given (feature_map, ansatz) pair.

    qnode(x, weights) -> expval(PauliZ(0))  [or full statevector if return_state]
    """
    n_qubits = n_qubits or rconfig.N_QUBITS
    n_layers = n_layers or rconfig.N_LAYERS
    fmap_fn = FEATURE_MAP_FNS[feature_map]
    ansatz_fn = ANSATZ_FNS[ansatz]
    dev = get_device(n_qubits)
    shape = weight_shape(ansatz, n_layers, n_qubits)

    kwargs = {} if return_state else dict(interface=interface, diff_method=diff_method)

    @qml.qnode(dev, **kwargs)
    def circuit(x, weights):
        fmap_fn(x, n_qubits)
        ansatz_fn(weights, n_qubits)
        if return_state:
            return qml.state()
        return qml.expval(qml.PauliZ(0))

    return circuit, shape


def feature_map_only_state_circuit(feature_map, n_qubits=None):
    """Statevector produced by the feature map alone (no ansatz) — used for
    the feature-space separability analysis and QK-SVM kernels."""
    n_qubits = n_qubits or rconfig.N_QUBITS
    fmap_fn = FEATURE_MAP_FNS[feature_map]
    dev = get_device(n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        fmap_fn(x, n_qubits)
        return qml.state()

    return circuit


def gate_count_and_depth(feature_map, ansatz, n_qubits, n_layers, x=None):
    """Build the circuit with a representative input/weights, extract its
    tape to get an exact (not analytic-formula) gate count / depth / op-type
    list. `x` should have length n_qubits for every feature map, including
    Amplitude — AmplitudeEmbedding pads it to 2**n_qubits itself, exactly as
    the rest of the pipeline calls it (see circuits.apply_amplitude)."""
    shape = weight_shape(ansatz, n_layers, n_qubits)
    if x is None:
        # non-trivial representative input (avoids a degenerate all-zero
        # amplitude-embedding decomposition, which under-counts gates)
        rng_local = np.random.RandomState(0)
        x = rng_local.uniform(-1.0, 1.0, n_qubits)
    w = np.zeros(shape)

    dev = get_device(n_qubits)

    @qml.qnode(dev)
    def circuit(x, weights):
        FEATURE_MAP_FNS[feature_map](x, n_qubits)
        ANSATZ_FNS[ansatz](weights, n_qubits)
        return qml.expval(qml.PauliZ(0))

    raw_tape = qml.workflow.construct_tape(circuit)(x, w)
    # Expand templates (StronglyEntanglingLayers, AmplitudeEmbedding, ...) to
    # elementary gates so counts are comparable across configurations —
    # otherwise a template counts as a single "operation" regardless of size.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tape = raw_tape.expand(depth=10)
    from collections import Counter
    return {
        "gate_count": len(tape.operations),
        "gate_types": dict(Counter(op.name for op in tape.operations)),
        "depth": tape.graph.get_depth() if hasattr(tape, "graph") else None,
        "n_trainable_params": int(np.prod(shape)),
        "raw_template_gate_count": len(raw_tape.operations),
    }
