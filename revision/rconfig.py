"""
Revision-specific configuration for the journal-revision experimental package.

Reuses base hyperparameters from the original `config.py` wherever the
protocol is meant to stay identical (Section 5: "same training protocol
unless a confirmed bug requires correction"). Adds only what the reviewer
response requires: seeds, factorial design axes, matched-budget targets,
barren-plateau grid, and output paths.
"""
import os
import sys

# Make repo root importable (original flat modules: config, data_prep, ...)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402  (original repo config — untouched)

# ── reused-as-is from the original protocol ──────────────────────────────────
N_QUBITS = config.N_QUBITS          # 4  (primary factorial + matched baselines)
N_LAYERS = config.N_LAYERS          # 2  (primary factorial)
LR = config.LR                      # 0.01
BATCH_SIZE = config.BATCH_SIZE      # 16
N_EPOCHS = config.N_EPOCHS          # 30
N_QUANTUM_SAMPLES = config.N_QUANTUM_SAMPLES  # 2000
TEST_SIZE = config.TEST_SIZE        # 0.3
BASE_SEED = config.SEED             # 42 — original single-seed protocol

# ── revision seeds (Section 5, 9, 19) ────────────────────────────────────────
PRIMARY_SEEDS = [42, 123, 456, 789, 2026]
MATCHED_BUDGET_SEEDS = [42, 123, 456, 789, 2026]  # all 5; runtime confirmed acceptable
SPLIT_SEEDS = [42, 123, 456, 789, 2026]

# ── primary controlled partition (Section 2) ─────────────────────────────────
PRIMARY_N_TRAIN = 1400
PRIMARY_N_TEST = 600

# ── 3x3 factorial design (Section 3) ─────────────────────────────────────────
FEATURE_MAPS = ["ZZ", "Amplitude", "Angle"]
ANSATZE = ["StronglyEnt", "BasicEnt", "CustomAnsatz"]

WEIGHT_SHAPE_RULE = {
    # ansatz -> lambda (n_layers, n_qubits) -> shape
    "StronglyEnt": lambda L, n: (L, n, 3),
    "BasicEnt":    lambda L, n: (L, n),
    "CustomAnsatz": lambda L, n: (L, n, 4),
}

# mapping of the 3 pre-existing (non-refactored) configurations, used only
# for the Section-4 equivalence check against the refactored circuit builder
ORIGINAL_CONFIGS = [
    ("ZZ", "StronglyEnt"),
    ("Amplitude", "BasicEnt"),
    ("Angle", "CustomAnsatz"),
]

# ── matched parameter-budget sensitivity (Section 9) ─────────────────────────
MATCHED_BUDGET_TARGET_PARAMS = 48
MATCHED_BUDGET_LAYERS = {
    # verified programmatically in matched_budget.py before use
    "StronglyEnt": 4,    # 4 * 4 * 3 = 48
    "BasicEnt": 12,      # 12 * 4     = 48
    "CustomAnsatz": 3,   # 3 * 4 * 4  = 48
}

# ── QK-SVM matched partition (Section 11) ─────────────────────────────────────
QKERNEL_N_TRAIN = PRIMARY_N_TRAIN   # attempt full 1400 via statevector-Gram approach
QKERNEL_N_TEST = PRIMARY_N_TEST

# ── separability (Section 13) ─────────────────────────────────────────────────
SEPARABILITY_N_PAIRS_TARGET = 5000   # per feature map
SEPARABILITY_BOOTSTRAP_N = 2000

# ── expressibility (Section 14) ───────────────────────────────────────────────
EXPRESSIBILITY_N_PAIRS = 1000        # per architecture per repetition
EXPRESSIBILITY_N_REPETITIONS = 5
EXPRESSIBILITY_BOOTSTRAP_N = 2000
EXPRESSIBILITY_N_BINS = 75

# ── barren-plateau / gradient-scaling grid (Section 16-18) ───────────────────
BP_QUBITS = [2, 3, 4, 5, 6, 8]
BP_DEPTHS = [1, 2, 3, 4, 6]
BP_N_INIT = 100
BP_ANSATZE = ["StronglyEnt", "BasicEnt", "CustomAnsatz"]

# ── noise robustness (Section 20, optional) ───────────────────────────────────
NOISE_PROBS = [0.0, 0.001, 0.005, 0.01, 0.02]

# ── output paths (Section 22) ─────────────────────────────────────────────────
RESULTS_DIR = os.path.join(_ROOT, "results", "revision")
FIGURES_DIR = os.path.join(_ROOT, "figures", "revision")

DIRS = {
    "root": RESULTS_DIR,
    "dataset": os.path.join(RESULTS_DIR, "dataset"),
    "factorial": os.path.join(RESULTS_DIR, "factorial"),
    "matched_budget": os.path.join(RESULTS_DIR, "matched_budget"),
    "matched_baselines": os.path.join(RESULTS_DIR, "matched_baselines"),
    "separability": os.path.join(RESULTS_DIR, "separability"),
    "expressibility": os.path.join(RESULTS_DIR, "expressibility"),
    "barren_plateau": os.path.join(RESULTS_DIR, "barren_plateau"),
    "split_robustness": os.path.join(RESULTS_DIR, "split_robustness"),
    "noise": os.path.join(RESULTS_DIR, "noise"),
}


def ensure_dirs():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(os.path.join(FIGURES_DIR, "confusion_matrices"), exist_ok=True)


ensure_dirs()
