"""
Round-2 Phase 7 — assembles results/revision/final_review_results/, the
curated package handed to the two researchers (and committed to git, unlike
the rest of results/revision/). Reads only already-generated artifacts;
does not run or re-run any experiment.
"""
import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rconfig, utils

OUT = rconfig.DIRS["final_review"]
SR_DIR = rconfig.DIRS["split_robustness"]
BP_DIR = rconfig.DIRS["barren_plateau"]


def _mkdirs():
    for d in ["classification/confusion_matrices", "classification/split_indices",
              "classification/predictions", "gradients/bootstrap_results",
              "dataset", "figures"]:
        os.makedirs(os.path.join(OUT, d), exist_ok=True)


def _build_classification():
    fixed = os.path.join(SR_DIR, "fixed_subset")

    # repeated_split_metrics.csv — full per-run granularity
    full = pd.read_csv(os.path.join(fixed, "all_runs.csv"))
    full.to_csv(os.path.join(OUT, "classification", "repeated_split_metrics.csv"), index=False)

    # repeated_split_summary.csv — aggregated mean/std/count per method
    summ = pd.read_csv(os.path.join(fixed, "summary.csv"))
    summ.to_csv(os.path.join(OUT, "classification", "repeated_split_summary.csv"), index=False)

    # paired_comparisons.csv
    pc_path = os.path.join(SR_DIR, "paired_comparisons.csv")
    if os.path.exists(pc_path):
        shutil.copy(pc_path, os.path.join(OUT, "classification", "paired_comparisons.csv"))

    # split_indices — copy the fixed-subset npy files (the primary robustness evidence)
    idx_src = os.path.join(fixed, "split_indices")
    idx_dst = os.path.join(OUT, "classification", "split_indices")
    if os.path.isdir(idx_src):
        for fn in os.listdir(idx_src):
            shutil.copy(os.path.join(idx_src, fn), os.path.join(idx_dst, fn))

    # confusion_matrices + predictions — pulled from the raw per-run JSONL records
    cm_records, pred_rows = [], []
    for fname, mtype in [("all_runs_full.jsonl", "VQC"), ("classical_all_runs.jsonl", "Classical"),
                          ("qkernel_all_runs.jsonl", "QK-SVM")]:
        for r in utils.read_jsonl(os.path.join(fixed, fname)):
            if r.get("status") != "ok":
                continue
            name = f"{r.get('feature_map','')}+{r.get('ansatz','')}" if mtype == "VQC" \
                else (r.get("model") or f"{r.get('feature_map')} kernel")
            cm_records.append({
                "method_type": mtype, "method_name": name, "split_seed": r.get("split_seed"),
                "init_seed": r.get("init_seed"), "confusion_matrix": json.dumps(r.get("confusion_matrix")),
                "n_test": len(r.get("y_true", [])),
            })
            if "y_true" in r:
                for i, (yt, yp) in enumerate(zip(r["y_true"], r["y_pred"])):
                    pr = r.get("y_proba", [None] * len(r["y_true"]))[i] if r.get("y_proba") else None
                    pred_rows.append({
                        "method_type": mtype, "method_name": name, "split_seed": r.get("split_seed"),
                        "init_seed": r.get("init_seed"), "sample_position": i,
                        "y_true": yt, "y_pred": yp, "y_proba": pr,
                    })
    pd.DataFrame(cm_records).to_csv(
        os.path.join(OUT, "classification", "confusion_matrices", "all_confusion_matrices.csv"), index=False)
    pred_df = pd.DataFrame(pred_rows)
    for split_seed, g in pred_df.groupby("split_seed"):
        g.to_csv(os.path.join(OUT, "classification", "predictions", f"predictions_split{split_seed}.csv"),
                  index=False)
    print(f"  classification/: {len(full)} metric rows, {len(cm_records)} confusion matrices, "
          f"{len(pred_df)} prediction rows across {pred_df['split_seed'].nunique() if len(pred_df) else 0} splits")


def _build_gradients():
    summ = pd.read_csv(os.path.join(BP_DIR, "summary.csv"))
    summ.to_csv(os.path.join(OUT, "gradients", "gradient_variance_summary.csv"), index=False)

    fits = utils.load_json(os.path.join(BP_DIR, "scaling_fits.json"))
    slope_rows, ci_rows, sens_rows = [], [], []
    for ansatz, block in fits.items():
        if ansatz.startswith("_") or not isinstance(block, dict):
            continue
        for axis_key, axis_name in [("headline_vs_qubits", "qubits"), ("headline_vs_depth", "depth")]:
            fit = block.get(axis_key)
            if not fit:
                continue
            slope_rows.append({"ansatz": ansatz, "axis": axis_name,
                                "slope_point_estimate": fit["slope_point_estimate"],
                                "r_squared": fit["r_squared"], "n_points": fit["n_points"],
                                "evidence_label": fit["evidence_classification"]["label"]})
            ci_rows.append({"ansatz": ansatz, "axis": axis_name,
                             "slope_ci95_lo": fit["slope_ci95_lo"], "slope_ci95_hi": fit["slope_ci95_hi"],
                             "n_bootstrap_replicates": fit["n_bootstrap_replicates"],
                             "n_replicates_with_floored_variance": fit["n_replicates_with_floored_variance"]})
        sens = block.get("sensitivity_parameter_selection_vs_qubits", {})
        for stat_name, v in sens.items():
            if v:
                sens_rows.append({"ansatz": ansatz, "analysis": "parameter_selection", "statistic": stat_name,
                                   "slope": v["slope_point_estimate"], "r_squared": v["r_squared"],
                                   "slope_ci95_lo": v["slope_ci95_lo"], "slope_ci95_hi": v["slope_ci95_hi"]})
        loo = block.get("sensitivity_leave_one_qubit_count_out", {})
        for held_out, v in loo.items():
            if "slope_without_this_qubit_count" in v:
                sens_rows.append({"ansatz": ansatz, "analysis": "leave_one_qubit_count_out",
                                   "statistic": f"held_out_n={held_out}",
                                   "slope": v["slope_without_this_qubit_count"], "r_squared": v["r_squared"],
                                   "slope_ci95_lo": None, "slope_ci95_hi": None})

    pd.DataFrame(slope_rows).to_csv(os.path.join(OUT, "gradients", "gradient_scaling_slopes.csv"), index=False)
    pd.DataFrame(ci_rows).to_csv(os.path.join(OUT, "gradients", "slope_confidence_intervals.csv"), index=False)
    pd.DataFrame(sens_rows).to_csv(os.path.join(OUT, "gradients", "sensitivity_analysis.csv"), index=False)
    utils.save_json(os.path.join(OUT, "gradients", "bootstrap_results", "scaling_fits_full.json"), fits)
    det = utils.load_json(os.path.join(BP_DIR, "determinism_check.json"))
    if det:
        utils.save_json(os.path.join(OUT, "gradients", "bootstrap_results", "determinism_check.json"), det)
    print(f"  gradients/: {len(slope_rows)} slope rows, {len(sens_rows)} sensitivity rows")


def _build_dataset():
    prov = utils.load_json(os.path.join(rconfig.RESULTS_DIR, "dataset_provenance.json"))
    utils.save_json(os.path.join(OUT, "dataset", "provenance.json"), prov)


def _build_figures():
    fixed = os.path.join(SR_DIR, "fixed_subset")
    summ = pd.read_csv(os.path.join(fixed, "summary.csv"))
    summ = summ.sort_values("test_accuracy_mean", ascending=True)

    for metric, fname in [("test_accuracy", "repeated_split_accuracy.pdf"), ("f1", "repeated_split_f1.pdf")]:
        fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(summ))))
        colors = {"VQC": "#4C72B0", "QK-SVM": "#DD8452", "Classical": "#55A868"}
        bar_colors = [colors.get(t, "gray") for t in summ["method_type"]]
        ax.barh(summ["method_name"], summ[f"{metric}_mean"], xerr=summ[f"{metric}_std"].fillna(0),
                capsize=3, color=bar_colors)
        ax.set_xlabel(metric.replace("_", " ").title() + " (mean ± SD across 5 splits)")
        ax.set_title(f"Repeated-split {metric.replace('_',' ')} (fixed-subset, n=5 partitions)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
        ax.legend(handles, colors.keys(), loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figures", fname))
        plt.close(fig)

    fits = utils.load_json(os.path.join(BP_DIR, "scaling_fits.json"))
    bp_summ = pd.read_csv(os.path.join(BP_DIR, "summary.csv"))
    for axis, xcol, fname in [("qubits", "n_qubits", "gradient_scaling_qubits.pdf"),
                               ("depth", "depth", "gradient_scaling_depth.pdf")]:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for ansatz in rconfig.BP_ANSATZE:
            sub = bp_summ[bp_summ["ansatz"] == ansatz]
            agg = sub.groupby(xcol)["mean_of_per_param_variances"].mean().reset_index().sort_values(xcol)
            agg = agg[agg["mean_of_per_param_variances"] > 0]
            if agg.empty:
                continue
            ax.plot(agg[xcol], np.log10(agg["mean_of_per_param_variances"]), marker="o", label=ansatz)
        ax.set_xlabel(f"Number of {axis}" if axis == "qubits" else "Circuit depth (layers)")
        ax.set_ylabel("log10(mean per-parameter gradient variance)")
        ax.set_title(f"Gradient-variance scaling vs {axis} (primary statistic, round-2 corrected)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figures", fname))
        plt.close(fig)

    # a handful of representative confusion matrices (best VQC, best classical, best QK-SVM)
    cm_csv = pd.read_csv(os.path.join(OUT, "classification", "confusion_matrices", "all_confusion_matrices.csv"))
    best_names = summ.sort_values("test_accuracy_mean", ascending=False).groupby("method_type").head(1)
    picks = []
    for _, row in best_names.iterrows():
        match = cm_csv[(cm_csv["method_type"] == row["method_type"]) &
                        (cm_csv["method_name"] == row["method_name"])]
        if not match.empty:
            picks.append(match.iloc[0])
    if picks:
        fig, axes = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3.2))
        axes = np.atleast_1d(axes)
        for ax, row in zip(axes, picks):
            cm = np.array(json.loads(row["confusion_matrix"]))
            im = ax.imshow(cm, cmap="Blues")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_title(f"{row['method_name']}\n({row['method_type']}, split={row['split_seed']})", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figures", "selected_confusion_matrices.pdf"))
        plt.close(fig)
    print(f"  figures/: {len(os.listdir(os.path.join(OUT, 'figures')))} files")


def build_package():
    print("=" * 65)
    print("  ROUND-2 PHASE 7 — FINAL REVIEW RESULTS PACKAGE")
    print("=" * 65)
    _mkdirs()
    _build_classification()
    _build_gradients()
    _build_dataset()
    _build_figures()

    manifest = utils.load_json(os.path.join(rconfig.RESULTS_DIR, "revision_manifest.json"))
    if manifest:
        utils.save_json(os.path.join(OUT, "experiment_manifest.json"), manifest)
    validation = utils.load_json(os.path.join(rconfig.RESULTS_DIR, "validation_report.json"))
    if validation:
        utils.save_json(os.path.join(OUT, "validation_report.json"), validation)
    print(f"  Package assembled -> {OUT}")


if __name__ == "__main__":
    build_package()
