"""Section 22 — revision_manifest.json: full reproducibility record."""
import os
import time
import datetime

from revision import rconfig, utils


def write_manifest(save=True):
    print("=" * 65)
    print("  SECTION 22 — REVISION MANIFEST")
    print("=" * 65)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    versions = utils.get_versions()
    commit = utils.get_git_commit(root)

    output_files = []
    for base in [rconfig.RESULTS_DIR, rconfig.FIGURES_DIR]:
        for dirpath, _, filenames in os.walk(base):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                output_files.append(os.path.relpath(full, root))

    manifest = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "timestamp_unix": time.time(),
        "versions": versions,
        "platform": versions.get("platform"),
        "git_commit": commit,
        "seeds": {
            "base_seed": rconfig.BASE_SEED,
            "primary_seeds": rconfig.PRIMARY_SEEDS,
            "matched_budget_seeds": rconfig.MATCHED_BUDGET_SEEDS,
            "split_seeds": rconfig.SPLIT_SEEDS,
        },
        "hyperparameters": {
            "n_qubits_primary": rconfig.N_QUBITS, "n_layers_primary": rconfig.N_LAYERS,
            "lr": rconfig.LR, "batch_size": rconfig.BATCH_SIZE, "n_epochs": rconfig.N_EPOCHS,
            "primary_n_train": rconfig.PRIMARY_N_TRAIN, "primary_n_test": rconfig.PRIMARY_N_TEST,
            "feature_maps": rconfig.FEATURE_MAPS, "ansatze": rconfig.ANSATZE,
            "matched_budget_target_params": rconfig.MATCHED_BUDGET_TARGET_PARAMS,
            "matched_budget_layers": rconfig.MATCHED_BUDGET_LAYERS,
            "barren_plateau_qubit_grid": rconfig.BP_QUBITS,
            "barren_plateau_depth_grid": rconfig.BP_DEPTHS,
            "barren_plateau_n_init": rconfig.BP_N_INIT,
            "expressibility_n_pairs": rconfig.EXPRESSIBILITY_N_PAIRS,
            "expressibility_n_repetitions": rconfig.EXPRESSIBILITY_N_REPETITIONS,
            "separability_n_pairs_target": rconfig.SEPARABILITY_N_PAIRS_TARGET,
            "noise_probs": rconfig.NOISE_PROBS,
        },
        "dataset": utils.load_json(os.path.join(rconfig.RESULTS_DIR, "dataset_audit.json")),
        "output_files": sorted(output_files),
        "n_output_files": len(output_files),
    }

    if save:
        utils.save_json(os.path.join(rconfig.RESULTS_DIR, "revision_manifest.json"), manifest)
        print(f"  Saved -> {os.path.join(rconfig.RESULTS_DIR, 'revision_manifest.json')}")
        print(f"  {len(output_files)} output files recorded.")
    return manifest
