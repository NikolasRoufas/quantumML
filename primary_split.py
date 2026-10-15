"""
Section 2 — Primary controlled train/test split.

Replicates data_prep.prepare_quantum's exact subsample-then-split logic
(same RandomState(SEED) subsample of 2000, same stratified 70/30 split) but
tracks absolute indices into the full 19,020-sample array so they can be
saved and reused verbatim by every model in the primary comparison.

Preprocessing (StandardScaler -> PCA -> StandardScaler -> clip) is fit on
the training partition only, mirroring data_prep.prepare_quantum.
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import config
import rconfig, utils


def build_primary_split(X, y, save=True):
    y = np.asarray(y)
    rng = np.random.RandomState(config.SEED)
    subsample_idx = rng.choice(len(X), config.N_QUANTUM_SAMPLES, replace=False)

    all_idx = np.arange(len(subsample_idx))
    train_pos, test_pos = train_test_split(
        all_idx, test_size=config.TEST_SIZE, random_state=config.SEED,
        stratify=y[subsample_idx],
    )
    train_idx = subsample_idx[train_pos]   # absolute indices into the full 19,020 array
    test_idx = subsample_idx[test_pos]

    assert len(train_idx) == rconfig.PRIMARY_N_TRAIN, len(train_idx)
    assert len(test_idx) == rconfig.PRIMARY_N_TEST, len(test_idx)
    assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0, "train/test index leakage"

    X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scaler = StandardScaler().fit(X_tr_raw)
    X_tr_sc = scaler.transform(X_tr_raw)
    X_te_sc = scaler.transform(X_te_raw)

    pca = PCA(n_components=config.N_FEATURES, random_state=config.SEED).fit(X_tr_sc)
    X_tr_pca = pca.transform(X_tr_sc)
    X_te_pca = pca.transform(X_te_sc)

    pca_scaler = StandardScaler().fit(X_tr_pca)
    X_tr_q = np.clip(pca_scaler.transform(X_tr_pca) * (np.pi / 3.0), -np.pi, np.pi)
    X_te_q = np.clip(pca_scaler.transform(X_te_pca) * (np.pi / 3.0), -np.pi, np.pi)

    if save:
        np.save(os.path.join(rconfig.RESULTS_DIR, "primary_train_indices.npy"), train_idx)
        np.save(os.path.join(rconfig.RESULTS_DIR, "primary_test_indices.npy"), test_idx)
        meta = {
            "subsample_size": int(config.N_QUANTUM_SAMPLES),
            "subsample_random_state": int(config.SEED),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "stratify": True,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "pca_components": pca.components_.tolist(),
            "pca_mean": pca.mean_.tolist(),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "pca_scaler_mean": pca_scaler.mean_.tolist(),
            "pca_scaler_scale": pca_scaler.scale_.tolist(),
            "clip_range": [-np.pi, np.pi],
            "clip_pre_scale": np.pi / 3.0,
            "train_class_counts": {int(k): int(v) for k, v in zip(*np.unique(y_tr, return_counts=True))},
            "test_class_counts": {int(k): int(v) for k, v in zip(*np.unique(y_te, return_counts=True))},
        }
        utils.save_json(os.path.join(rconfig.RESULTS_DIR, "primary_split_preprocessing.json"), meta)
        print(f"  Primary split saved: train={len(train_idx)} test={len(test_idx)}")
        print(f"  Train class counts: {meta['train_class_counts']}  Test class counts: {meta['test_class_counts']}")

    return {
        "train_idx": train_idx, "test_idx": test_idx,
        "X_tr_raw": X_tr_raw, "X_te_raw": X_te_raw,
        "X_tr_q": X_tr_q, "X_te_q": X_te_q,
        "y_tr": y_tr, "y_te": y_te,
        "scaler": scaler, "pca": pca, "pca_scaler": pca_scaler,
    }


def load_primary_split(X, y):
    """Load previously-saved indices and rebuild the exact same fitted
    preprocessing (train-only fit) rather than re-drawing a subsample."""
    tr_path = os.path.join(rconfig.RESULTS_DIR, "primary_train_indices.npy")
    te_path = os.path.join(rconfig.RESULTS_DIR, "primary_test_indices.npy")
    if not (os.path.exists(tr_path) and os.path.exists(te_path)):
        return build_primary_split(X, y, save=True)

    y = np.asarray(y)
    train_idx = np.load(tr_path)
    test_idx = np.load(te_path)

    X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scaler = StandardScaler().fit(X_tr_raw)
    X_tr_sc = scaler.transform(X_tr_raw)
    X_te_sc = scaler.transform(X_te_raw)
    pca = PCA(n_components=config.N_FEATURES, random_state=config.SEED).fit(X_tr_sc)
    X_tr_pca = pca.transform(X_tr_sc)
    X_te_pca = pca.transform(X_te_sc)
    pca_scaler = StandardScaler().fit(X_tr_pca)
    X_tr_q = np.clip(pca_scaler.transform(X_tr_pca) * (np.pi / 3.0), -np.pi, np.pi)
    X_te_q = np.clip(pca_scaler.transform(X_te_pca) * (np.pi / 3.0), -np.pi, np.pi)

    return {
        "train_idx": train_idx, "test_idx": test_idx,
        "X_tr_raw": X_tr_raw, "X_te_raw": X_te_raw,
        "X_tr_q": X_tr_q, "X_te_q": X_te_q,
        "y_tr": y_tr, "y_te": y_te,
        "scaler": scaler, "pca": pca, "pca_scaler": pca_scaler,
    }
