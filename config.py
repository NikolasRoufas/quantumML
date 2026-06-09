import os

# Quantum circuit parameters
N_QUBITS = 4
N_LAYERS = 2
N_FEATURES = 4          # features after PCA reduction

# Training hyperparameters
LR = 0.01
BATCH_SIZE = 16
N_EPOCHS = 30

# Dataset: MAGIC Gamma Telescope (19,020 samples, 10 features, binary)
# Subsample for quantum simulation feasibility
N_QUANTUM_SAMPLES = 2000
TEST_SIZE = 0.3
SEED = 42

# Paths
FIGURES_DIR = "figures"
RESULTS_DIR  = "results"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,  exist_ok=True)
