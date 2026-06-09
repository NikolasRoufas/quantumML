import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import config


def load_magic():
    """MAGIC Gamma Telescope: 19,020 samples, 10 features, binary (gamma=1, hadron=0)."""
    print("Loading MAGIC Gamma Telescope dataset...", flush=True)
    data = fetch_openml("magic", version=1, as_frame=False, parser="auto")
    X = data.data.astype(np.float64)
    # Labels are stored as string '0' / '1' (gamma=1, hadron=0)
    y = data.target.astype(int)
    pos = int(y.sum())
    print(f"  Samples: {len(X)} | Features: {X.shape[1]} | "
          f"Gamma (1): {pos} ({pos/len(X)*100:.1f}%) | Hadron (0): {len(X)-pos}", flush=True)
    return X, y


def prepare_classical(X, y):
    """Full dataset, scaled, for classical baselines."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )
    scaler = StandardScaler()
    return scaler.fit_transform(X_tr), scaler.transform(X_te), y_tr, y_te


def prepare_quantum(X, y):
    """
    Subsample to N_QUANTUM_SAMPLES, then:
      1. StandardScaler
      2. PCA → N_FEATURES components
      3. Rescale to [-pi, pi] range for quantum encoding
    Returns (X_tr, X_te, y_tr, y_te, scaler, pca).
    """
    rng = np.random.RandomState(config.SEED)
    idx = rng.choice(len(X), config.N_QUANTUM_SAMPLES, replace=False)
    Xs, ys = X[idx], y[idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, ys, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=ys
    )

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    pca = PCA(n_components=config.N_FEATURES, random_state=config.SEED)
    X_tr_pca = pca.fit_transform(X_tr_sc)
    X_te_pca = pca.transform(X_te_sc)

    # Map to [-pi, pi] so rotation gates are well-conditioned
    pca_scaler = StandardScaler()
    X_tr_pca = pca_scaler.fit_transform(X_tr_pca)
    X_te_pca = pca_scaler.transform(X_te_pca)
    X_tr_pca = np.clip(X_tr_pca * (np.pi / 3.0), -np.pi, np.pi)
    X_te_pca = np.clip(X_te_pca * (np.pi / 3.0), -np.pi, np.pi)

    print(f"  Quantum split — train: {len(X_tr_pca)}, test: {len(X_te_pca)}", flush=True)
    return X_tr_pca, X_te_pca, y_tr, y_te, scaler, pca
