"""
Section 12 — One unified primary comparison table: 9 VQCs + 3 QK-SVMs +
SVM-RBF + Random Forest + MLP, all evaluated on the SAME primary 1,400/600
partition with consistent metrics. Does not mix full-dataset classical
results with the reduced-data quantum results.
"""
import os
import pandas as pd

from revision import rconfig


METRIC_COLS = ["test_accuracy", "precision", "recall", "f1", "roc_auc",
               "pr_auc", "balanced_accuracy", "mcc"]


def build_unified_comparison():
    print("=" * 65)
    print("  SECTION 12 — UNIFIED PRIMARY COMPARISON TABLE")
    print("=" * 65)
    factorial_summary_path = os.path.join(rconfig.RESULTS_DIR, "factorial", "summary.csv")
    classical_summary_path = os.path.join(rconfig.DIRS["matched_baselines"], "classical_summary.csv")
    qkernel_path = os.path.join(rconfig.DIRS["matched_baselines"], "qkernel_results.csv")

    rows = []

    if os.path.exists(factorial_summary_path):
        vqc = pd.read_csv(factorial_summary_path)
        for _, r in vqc.iterrows():
            rows.append({
                "method_type": "VQC",
                "method_name": f"{r['feature_map']}+{r['ansatz']}",
                "n_train": rconfig.PRIMARY_N_TRAIN, "n_test": rconfig.PRIMARY_N_TEST,
                "n_seeds": r.get("test_accuracy_count"),
                **{f"{m}_mean": r.get(f"{m}_mean") for m in METRIC_COLS},
                **{f"{m}_std": r.get(f"{m}_std") for m in METRIC_COLS},
            })
    else:
        print("  WARNING: factorial summary.csv not found yet — VQC rows omitted.")

    if os.path.exists(classical_summary_path):
        cls = pd.read_csv(classical_summary_path)
        for _, r in cls.iterrows():
            rows.append({
                "method_type": "Classical",
                "method_name": r["model"],
                "n_train": rconfig.PRIMARY_N_TRAIN, "n_test": rconfig.PRIMARY_N_TEST,
                "n_seeds": r.get("test_accuracy_count"),
                **{f"{m}_mean": r.get(f"{m}_mean") for m in METRIC_COLS},
                **{f"{m}_std": r.get(f"{m}_std") for m in METRIC_COLS},
            })
    else:
        print("  WARNING: classical_summary.csv not found yet — classical rows omitted.")

    if os.path.exists(qkernel_path):
        qk = pd.read_csv(qkernel_path)
        for _, r in qk.iterrows():
            rows.append({
                "method_type": "QK-SVM",
                "method_name": f"QKernel-{r['feature_map']}",
                "n_train": rconfig.PRIMARY_N_TRAIN, "n_test": rconfig.PRIMARY_N_TEST,
                "n_seeds": 1,
                **{f"{m}_mean": r.get(m) for m in METRIC_COLS},
                **{f"{m}_std": None for m in METRIC_COLS},
            })
    else:
        print("  WARNING: qkernel_results.csv not found yet — QK-SVM rows omitted.")

    df = pd.DataFrame(rows)
    out_path = os.path.join(rconfig.DIRS["matched_baselines"], "unified_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"  {len(df)} methods written -> {out_path}")
    if not df.empty:
        print(df[["method_type", "method_name", "test_accuracy_mean", "f1_mean"]]
              .sort_values("test_accuracy_mean", ascending=False).to_string(index=False))
    return df
