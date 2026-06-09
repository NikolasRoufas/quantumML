"""
Three quantum model configurations from the methodology:
  1. ZZ Feature Map  + Strongly Entangling Layers ansatz
  2. Amplitude Embedding + Basic Entangler Layers ansatz
  3. Angle Embedding    + Custom Ansatz (RX/RY/RZ + CNOT + CRZ)
All circuits: 4 qubits, configurable layers, torch interface with backprop.
"""
import math
import torch
import pennylane as qml
import config  # noqa: E402

N = config.N_QUBITS


dev = qml.device("default.qubit", wires=N)


# ── helpers ───────────────────────────────────────────────────────────────────

def _zz_feature_map(x):
    """Eq. (1) from methodology: ZZ entangling feature map."""
    for i in range(N):
        qml.Hadamard(wires=i)
    for i in range(N):
        qml.RZ(2.0 * x[i], wires=i)
    for i in range(N):
        for j in range(i + 1, N):
            qml.CNOT(wires=[i, j])
            qml.RZ(2.0 * math.pi * x[i] * x[j], wires=j)
            qml.CNOT(wires=[i, j])


def _custom_ansatz(weights):
    """Eq. (7): RX RY RZ per qubit + CNOT ring + CRZ ring."""
    n_layers = weights.shape[0]
    for l in range(n_layers):
        for i in range(N):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)
        for i in range(N):
            qml.CNOT(wires=[i, (i + 1) % N])
        for i in range(N):
            qml.CRZ(weights[l, i, 3], wires=[i, (i + 1) % N])


# ── circuits ──────────────────────────────────────────────────────────────────

@qml.qnode(dev, interface="torch", diff_method="backprop")
def zz_strongly_circuit(x, weights):
    """ZZ Feature Map + Strongly Entangling Layers. weights: (L, n, 3)."""
    _zz_feature_map(x)
    qml.StronglyEntanglingLayers(weights, wires=range(N))
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev, interface="torch", diff_method="backprop")
def amplitude_basic_circuit(x, weights):
    """Amplitude Embedding + Basic Entangler (RY) Layers. weights: (L, n)."""
    qml.AmplitudeEmbedding(features=x, wires=range(N), normalize=True, pad_with=0.0)
    qml.BasicEntanglerLayers(weights, wires=range(N), rotation=qml.RY)
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev, interface="torch", diff_method="backprop")
def angle_custom_circuit(x, weights):
    """Angle Embedding (RX+RY) + Custom Ansatz (RX/RY/RZ+CNOT+CRZ). weights: (L, n, 4)."""
    for i in range(N):
        qml.RX(x[i], wires=i)
        qml.RY(x[i], wires=i)
    _custom_ansatz(weights)
    return qml.expval(qml.PauliZ(0))


# ── weight shapes and initialisation ─────────────────────────────────────────

ZZ_SHAPE    = (config.N_LAYERS, N, 3)  # StronglyEntanglingLayers: Rot(phi,theta,omega)
AMP_SHAPE   = (config.N_LAYERS, N)     # BasicEntanglerLayers: single RY per qubit
ANGLE_SHAPE = (config.N_LAYERS, N, 4)  # Custom: RX, RY, RZ, CRZ


def init_weights(shape):
    # Use uniform init in [-pi, pi] — wider spread breaks symmetry and
    # avoids the flat-output region that causes all-one-class predictions.
    return torch.nn.Parameter(
        (torch.rand(*shape, dtype=torch.float64) * 2 - 1) * math.pi
    )
