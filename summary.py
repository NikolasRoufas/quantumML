"""Sections 26-27 — final console summary and FINAL_RESULTS.json."""
import os
import pandas as pd

import rconfig, utils


def _read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def print_final_summary():
    print("\n" + "#" * 70)
    print("# FINAL SUMMARY")
    print("#" * 70)

    print("\nA. DATASET AUDIT")
    audit = utils.load_json(os.path.join(rconfig.RESULTS_DIR, "dataset_audit.json"))
    if audit:
        print(f"  n_samples={audit['n_samples']}  n_features={audit['n_features']}  "
              f"positive_class={audit['positive_class']}")
        print(f"  class_counts={audit['class_counts']}  "
              f"proportions={ {k: round(v,4) for k,v in audit['class_proportions'].items()} }")

    print("\nB. PRIMARY FACTORIAL RESULTS")
    fac = _read_csv(os.path.join(rconfig.RESULTS_DIR, "factorial", "summary.csv"))
    if fac is not None:
        cols = ["feature_map", "ansatz", "test_accuracy_mean", "test_accuracy_std",
                "f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std",
                "balanced_accuracy_mean", "balanced_accuracy_std", "mcc_mean", "mcc_std"]
        cols = [c for c in cols if c in fac.columns]
        print(fac[cols].to_string(index=False))
    else:
        print("  (not available)")

    print("\nC. FEATURE-MAP MARGINAL RESULTS")
    fm = _read_csv(os.path.join(rconfig.RESULTS_DIR, "factorial", "feature_map_summary.csv"))
    print(fm.to_string(index=False) if fm is not None else "  (not available)")

    print("\nD. ANSATZ MARGINAL RESULTS")
    an = _read_csv(os.path.join(rconfig.RESULTS_DIR, "factorial", "ansatz_summary.csv"))
    print(an.to_string(index=False) if an is not None else "  (not available)")

    print("\nE. FACTORIAL STATISTICAL TESTS")
    stats = utils.load_json(os.path.join(rconfig.RESULTS_DIR, "factorial", "statistical_tests.json"))
    if stats:
        print(f"  {stats.get('limitation_note')}")
        for metric, m in stats.get("metrics", {}).items():
            anova = m.get("two_way_anova")
            print(f"  [{metric}] two-way ANOVA:")
            if anova:
                for term, v in anova.items():
                    if term == "Residual":
                        continue
                    print(f"    {term}: F={v['F']:.3f} p={v['p']:.4g} partial_eta_sq={v['partial_eta_squared']:.3f}")
    else:
        print("  (not available)")

    print("\nF. MATCHED-PARAMETER RESULTS")
    mb = _read_csv(os.path.join(rconfig.RESULTS_DIR, "matched_budget", "summary.csv"))
    print(mb.to_string(index=False) if mb is not None else "  (not available)")

    print("\nG. MATCHED MODEL COMPARISON")
    unified = _read_csv(os.path.join(rconfig.DIRS["matched_baselines"], "unified_comparison.csv"))
    if unified is not None:
        print(unified.sort_values("test_accuracy_mean", ascending=False)
              [["method_type", "method_name", "test_accuracy_mean", "f1_mean"]].to_string(index=False))
    else:
        print("  (not available)")

    print("\nH. EXPRESSIBILITY RESULTS")
    expr = utils.load_json(os.path.join(rconfig.DIRS["expressibility"], "expressibility_results.json"))
    if expr:
        for ansatz, v in expr.items():
            ci = v["kl_ci95_pooled_bootstrap"]
            print(f"  {ansatz}: KL={v['kl_mean']:.4f} +/- {v['kl_std']:.4f}  CI95=[{ci[0]:.4f},{ci[1]:.4f}]")
    else:
        print("  (not available)")

    print("\nI. BARREN-PLATEAU SCALING SUMMARY (round-2: per-parameter variance, bootstrap CI)")
    fits = utils.load_json(os.path.join(rconfig.DIRS["barren_plateau"], "scaling_fits.json"))
    if fits:
        for ansatz, v in fits.items():
            if ansatz.startswith("_") or not isinstance(v, dict):
                continue  # skip the "_methodology_note" metadata entry
            hn = v.get("headline_vs_qubits")
            hd = v.get("headline_vs_depth")
            print(f"  {ansatz}:")
            if hn:
                label = hn.get("evidence_classification", {}).get("label", hn.get("evidence_classification"))
                print(f"    vs qubits: slope={hn['slope_point_estimate']:.4f} "
                      f"CI95=[{hn['slope_ci95_lo']:.4f},{hn['slope_ci95_hi']:.4f}] r2={hn['r_squared']:.3f} "
                      f"-> {label}")
            if hd:
                label = hd.get("evidence_classification", {}).get("label", hd.get("evidence_classification"))
                print(f"    vs depth:  slope={hd['slope_point_estimate']:.4f} "
                      f"CI95=[{hd['slope_ci95_lo']:.4f},{hd['slope_ci95_hi']:.4f}] r2={hd['r_squared']:.3f} "
                      f"-> {label}")
    else:
        print("  (not available)")

    print("\nJ. SPLIT ROBUSTNESS SUMMARY")
    sr = _read_csv(os.path.join(rconfig.DIRS["split_robustness"], "summary.csv"))
    print(sr.to_string(index=False) if sr is not None else "  (not available)")

    print("\nK. FAILURES / DEVIATIONS")
    failed = utils.load_json(os.path.join(rconfig.RESULTS_DIR, "failed_runs.json"), default=[])
    if failed:
        print(f"  {len(failed)} recorded failures:")
        for f in failed[:50]:
            print(f"    - stage={f.get('stage')} reason={f.get('reason')}")
        if len(failed) > 50:
            print(f"    ... and {len(failed)-50} more (see failed_runs.json)")
    else:
        print("  None recorded.")
    print()


def write_final_results(save=True):
    print("=" * 65)
    print("  SECTION 27 — FINAL_RESULTS.json")
    print("=" * 65)

    def _load_csv_records(path):
        df = _read_csv(path)
        return df.to_dict(orient="records") if df is not None else None

    final = {
        "dataset_audit": utils.load_json(os.path.join(rconfig.RESULTS_DIR, "dataset_audit.json")),
        "primary_split_preprocessing": utils.load_json(
            os.path.join(rconfig.RESULTS_DIR, "primary_split_preprocessing.json")),
        "circuit_validation": utils.load_json(os.path.join(rconfig.RESULTS_DIR, "circuit_validation.json")),
        "factorial": {
            "results": utils.load_json(os.path.join(rconfig.RESULTS_DIR, "factorial", "factorial_results.json")),
            "statistical_tests": utils.load_json(
                os.path.join(rconfig.RESULTS_DIR, "factorial", "statistical_tests.json")),
        },
        "matched_budget": {
            "results": utils.load_json(os.path.join(rconfig.RESULTS_DIR, "matched_budget", "factorial_results.json")),
            "layer_verification": utils.load_json(
                os.path.join(rconfig.RESULTS_DIR, "matched_budget", "matched_budget_layer_verification.json")),
            "statistical_tests": utils.load_json(
                os.path.join(rconfig.RESULTS_DIR, "matched_budget", "statistical_tests.json")),
        },
        "classical_baselines": _load_csv_records(
            os.path.join(rconfig.DIRS["matched_baselines"], "classical_summary.csv")),
        "qkernel_baselines": _load_csv_records(
            os.path.join(rconfig.DIRS["matched_baselines"], "qkernel_results.csv")),
        "unified_comparison": _load_csv_records(
            os.path.join(rconfig.DIRS["matched_baselines"], "unified_comparison.csv")),
        "separability": {
            "summary": _load_csv_records(os.path.join(rconfig.DIRS["separability"], "separability_summary.csv")),
            "statistical_tests": utils.load_json(
                os.path.join(rconfig.DIRS["separability"], "statistical_tests.json")),
        },
        "expressibility": {
            "results": utils.load_json(os.path.join(rconfig.DIRS["expressibility"], "expressibility_results.json")),
            "gradient_association": _load_csv_records(
                os.path.join(rconfig.DIRS["expressibility"], "expressibility_gradient_association.csv")),
        },
        "barren_plateau": {
            "summary": _load_csv_records(os.path.join(rconfig.DIRS["barren_plateau"], "summary.csv")),
            "scaling_fits": utils.load_json(os.path.join(rconfig.DIRS["barren_plateau"], "scaling_fits.json")),
            "metadata": utils.load_json(os.path.join(rconfig.DIRS["barren_plateau"], "metadata.json")),
        },
        "split_robustness": _load_csv_records(os.path.join(rconfig.DIRS["split_robustness"], "summary.csv")),
        "noise": _load_csv_records(os.path.join(rconfig.DIRS["noise"], "noise_results.csv")),
        "sanity_checks": utils.load_json(os.path.join(rconfig.RESULTS_DIR, "sanity_checks.json")),
        "failed_runs_and_caveats": utils.load_json(os.path.join(rconfig.RESULTS_DIR, "failed_runs.json"), default=[]),
        "manifest": utils.load_json(os.path.join(rconfig.RESULTS_DIR, "revision_manifest.json")),
    }

    if save:
        utils.save_json(os.path.join(rconfig.RESULTS_DIR, "FINAL_RESULTS.json"), final)
        print(f"  Saved -> {os.path.join(rconfig.RESULTS_DIR, 'FINAL_RESULTS.json')}")
    return final
