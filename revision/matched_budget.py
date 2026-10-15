"""
Section 9 — Matched parameter-budget sensitivity experiment.

The primary 2-layer 3x3 factorial has different trainable-parameter counts
per ansatz. This secondary experiment fixes the parameter budget at (as
close to) 48 for every ansatz at 4 qubits and re-runs the full 3x3 factorial
under that constraint. It does NOT replace the primary factorial.
"""
import os
import numpy as np

from revision import rconfig, circuits, factorial, utils


def verify_matched_budget_layer_counts():
    """Programmatically confirm the layer counts in rconfig actually hit the
    target parameter budget for the current weight-shape rules."""
    n = rconfig.N_QUBITS
    verified = {}
    for ansatz, L in rconfig.MATCHED_BUDGET_LAYERS.items():
        shape = circuits.weight_shape(ansatz, L, n)
        n_params = int(np.prod(shape))
        verified[ansatz] = {
            "layers": L, "weight_shape": list(shape), "n_params": n_params,
            "matches_target": n_params == rconfig.MATCHED_BUDGET_TARGET_PARAMS,
        }
        print(f"    {ansatz}: L={L}, shape={shape}, params={n_params} "
              f"(target={rconfig.MATCHED_BUDGET_TARGET_PARAMS}) "
              f"{'OK' if verified[ansatz]['matches_target'] else 'MISMATCH'}")
    return verified


def run_matched_budget(split, resume=False):
    print("=" * 65)
    print("  SECTION 9 — MATCHED PARAMETER-BUDGET SENSITIVITY")
    print("=" * 65)
    verified = verify_matched_budget_layer_counts()
    out_dir = os.path.join(rconfig.RESULTS_DIR, "matched_budget")
    utils.save_json(os.path.join(out_dir, "matched_budget_layer_verification.json"), verified)

    mismatches = [a for a, v in verified.items() if not v["matches_target"]]
    if mismatches:
        print(f"  WARNING: layer counts for {mismatches} do not hit the exact target "
              f"budget with the current weight-shape rule — proceeding with the "
              f"nearest achievable counts, recorded above (Section 9 explicitly "
              f"allows 'approximately/exactly matched').")

    for ansatz, v in verified.items():
        configs = [(fm, ansatz) for fm in rconfig.FEATURE_MAPS]
        print(f"  Running matched-budget cell: ansatz={ansatz}, layers={v['layers']}, "
              f"params={v['n_params']} across {len(configs)} feature maps x "
              f"{len(rconfig.MATCHED_BUDGET_SEEDS)} seeds")
        factorial.run_factorial(
            split, resume=resume, seeds=rconfig.MATCHED_BUDGET_SEEDS,
            out_subdir="matched_budget", n_layers=v["layers"], configs=configs,
            tag="matched_budget",
        )

    factorial.run_statistical_analysis(out_dir=out_dir)
    print(f"  Matched-budget results in {out_dir}")
