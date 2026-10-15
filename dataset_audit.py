"""
Section 1 — Dataset audit.

Everything here is computed programmatically from the loaded dataset —
nothing is hard-coded from the paper or the README.
"""
import os
import inspect
import numpy as np

import data_prep
import rconfig, utils


def run_audit(save=True):
    print("=" * 65)
    print("  SECTION 1 — DATASET AUDIT")
    print("=" * 65)

    X, y = data_prep.load_magic()
    y = np.asarray(y)

    n_samples, n_features = X.shape
    original_labels = sorted(int(v) for v in np.unique(y))
    encoded_labels = original_labels  # data_prep already returns 0/1 ints

    counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    proportions = {k: v / n_samples for k, v in counts.items()}

    # positive class = the label passed as the "positive" class to sklearn's
    # binary metrics (label 1) — determined programmatically from the encoding,
    # not assumed from the docstring.
    positive_class = 1 if 1 in counts else max(counts, key=counts.get)
    negative_class = [k for k in counts if k != positive_class][0]

    source_fn = inspect.getsource(data_prep.load_magic)
    dataset_source = "sklearn.datasets.fetch_openml('magic', version=1)" \
        if "fetch_openml" in source_fn else "unknown (inspect data_prep.load_magic)"

    audit = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "original_labels": original_labels,
        "encoded_labels": encoded_labels,
        "positive_class": int(positive_class),
        "negative_class": int(negative_class),
        "class_counts": counts,
        "class_proportions": proportions,
        "dataset_source": dataset_source,
        "expected_uci_n_samples": 19020,
        "n_samples_matches_uci_expectation": bool(n_samples == 19020),
        "preprocessing_settings": {
            "quantum_subsample_size": rconfig.N_QUANTUM_SAMPLES,
            "quantum_subsample_random_state": rconfig.BASE_SEED,
            "test_size": rconfig.TEST_SIZE,
            "scaler": "sklearn.preprocessing.StandardScaler",
            "pca_n_components": config_n_features(),
            "quantum_range_clip": "[-pi, pi] after a second StandardScaler on PCA components, scaled by pi/3",
            "classical_full_dataset_scaler": "sklearn.preprocessing.StandardScaler (no PCA)",
        },
    }

    print(f"  n_samples={n_samples}  n_features={n_features}")
    print(f"  original_labels={original_labels}  positive_class={positive_class}  negative_class={negative_class}")
    print(f"  class_counts={counts}")
    print(f"  class_proportions={ {k: round(v, 4) for k, v in proportions.items()} }")
    print(f"  dataset_source={dataset_source}")
    print(f"  matches expected UCI n=19020: {audit['n_samples_matches_uci_expectation']}")

    if save:
        path = os.path.join(rconfig.DIRS["dataset"], "dataset_audit.json")
        utils.save_json(path, audit)
        # also keep a top-level copy per the spec's exact requested path
        utils.save_json(os.path.join(rconfig.RESULTS_DIR, "dataset_audit.json"), audit)
        print(f"  Saved -> {path}")

    return audit, X, y


def config_n_features():
    import config
    return config.N_FEATURES


if __name__ == "__main__":
    run_audit()
