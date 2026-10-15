"""
Section 13 — Feature-space separability analysis.

This is explicitly NOT circuit expressibility (see Section 14 for that). It
is a data-dependent diagnostic: does a feature map's fidelity kernel place
same-class points closer together than different-class points, on the
primary training partition (never the held-out test partition).
"""
import os
import numpy as np
import pandas as pd
import scipy.stats as sps

import rconfig, utils, circuits as rcircuits


def run_separability(split, save=True):
    print("=" * 65)
    print("  SECTION 13 — FEATURE-SPACE SEPARABILITY (data-dependent, not expressibility)")
    print("=" * 65)
    out_dir = rconfig.DIRS["separability"]
    X_tr, y_tr = split["X_tr_q"], split["y_tr"]
    n = len(X_tr)
    rng = np.random.RandomState(rconfig.BASE_SEED)

    all_pairs_rows = []
    summary_rows = []
    stat_tests = {}

    for fmap in rconfig.FEATURE_MAPS:
        print(f"    {fmap}: computing {n} statevectors...")
        state_circ = rcircuits.feature_map_only_state_circuit(fmap, n_qubits=rconfig.N_QUBITS)
        # Batched — mathematically identical to per-sample calls (verified
        # elsewhere: max abs diff 0.0), ~200x less per-call overhead.
        states = np.asarray(state_circ(X_tr))

        target_total = rconfig.SEPARABILITY_N_PAIRS_TARGET
        target_each = target_total // 2
        same_pairs, diff_pairs = [], []
        attempts = 0
        max_attempts = target_total * 50
        while (len(same_pairs) < target_each or len(diff_pairs) < target_each) and attempts < max_attempts:
            i, j = rng.choice(n, 2, replace=False)
            attempts += 1
            is_same = bool(y_tr[i] == y_tr[j])
            if is_same and len(same_pairs) < target_each:
                same_pairs.append((i, j))
            elif not is_same and len(diff_pairs) < target_each:
                diff_pairs.append((i, j))

        def fidelity(i, j):
            return float(np.abs(np.dot(np.conj(states[i]), states[j])) ** 2)

        same_fid = [fidelity(i, j) for i, j in same_pairs]
        diff_fid = [fidelity(i, j) for i, j in diff_pairs]

        for (i, j), f in zip(same_pairs, same_fid):
            all_pairs_rows.append({"feature_map": fmap, "i": int(i), "j": int(j),
                                    "same_class": True, "fidelity": f})
        for (i, j), f in zip(diff_pairs, diff_fid):
            all_pairs_rows.append({"feature_map": fmap, "i": int(i), "j": int(j),
                                    "same_class": False, "fidelity": f})

        same_arr, diff_arr = np.array(same_fid), np.array(diff_fid)
        same_ci = utils.bootstrap_ci(same_arr, np.mean, n_boot=rconfig.SEPARABILITY_BOOTSTRAP_N, seed=0)
        diff_ci = utils.bootstrap_ci(diff_arr, np.mean, n_boot=rconfig.SEPARABILITY_BOOTSTRAP_N, seed=1)
        diff_of_means = same_arr.mean() - diff_arr.mean()

        def diff_stat(_):
            return None  # placeholder, not used directly

        rng_b = np.random.RandomState(2)
        boot_diffs = np.empty(rconfig.SEPARABILITY_BOOTSTRAP_N)
        for b in range(rconfig.SEPARABILITY_BOOTSTRAP_N):
            s = same_arr[rng_b.randint(0, len(same_arr), len(same_arr))]
            d = diff_arr[rng_b.randint(0, len(diff_arr), len(diff_arr))]
            boot_diffs[b] = s.mean() - d.mean()
        gap_ci_lo, gap_ci_hi = np.quantile(boot_diffs, [0.025, 0.975])

        mw_stat, mw_p = sps.mannwhitneyu(same_arr, diff_arr, alternative="two-sided")

        summary_rows.append({
            "feature_map": fmap, "n_same_pairs": len(same_arr), "n_diff_pairs": len(diff_arr),
            "same_mean": float(same_arr.mean()), "same_median": float(np.median(same_arr)),
            "same_std": float(same_arr.std()),
            "diff_mean": float(diff_arr.mean()), "diff_median": float(np.median(diff_arr)),
            "diff_std": float(diff_arr.std()),
            "gap_same_minus_diff": float(diff_of_means),
            "same_mean_ci_lo": same_ci["lo"], "same_mean_ci_hi": same_ci["hi"],
            "diff_mean_ci_lo": diff_ci["lo"], "diff_mean_ci_hi": diff_ci["hi"],
            "gap_ci_lo": float(gap_ci_lo), "gap_ci_hi": float(gap_ci_hi),
            "mannwhitney_U": float(mw_stat), "mannwhitney_p": float(mw_p),
        })
        stat_tests[fmap] = {
            "test": "Mann-Whitney U (two-sided)",
            "U": float(mw_stat), "p_value": float(mw_p),
            "significant_at_0.05": bool(mw_p < 0.05),
            "gap_bootstrap_ci95": [float(gap_ci_lo), float(gap_ci_hi)],
        }
        print(f"      same={same_arr.mean():.4f} diff={diff_arr.mean():.4f} "
              f"gap={diff_of_means:+.4f} [{gap_ci_lo:+.4f},{gap_ci_hi:+.4f}] "
              f"MWU p={mw_p:.2e}")

    if save:
        pd.DataFrame(all_pairs_rows).to_csv(os.path.join(out_dir, "separability_pairs.csv"), index=False)
        pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "separability_summary.csv"), index=False)
        utils.save_json(os.path.join(out_dir, "statistical_tests.json"), stat_tests)
        print(f"  Saved -> {out_dir}")

    return summary_rows, stat_tests
