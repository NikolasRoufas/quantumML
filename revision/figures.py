"""
Section 21 — Publication-quality figures for the revision package.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from revision import rconfig, utils

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10})


def _save(fig, name):
    path = os.path.join(rconfig.FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {path}")


def _heatmap(df, value_col, title, fname):
    pivot = df.pivot(index="feature_map", columns="ansatz", values=value_col)
    pivot = pivot.reindex(index=rconfig.FEATURE_MAPS, columns=rconfig.ANSATZE)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(pivot.values, cmap="viridis", vmin=np.nanmin(pivot.values), vmax=np.nanmax(pivot.values))
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < np.nanmean(pivot.values) else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, fname)


def fig_factorial_heatmaps():
    path = os.path.join(rconfig.RESULTS_DIR, "factorial", "summary.csv")
    if not os.path.exists(path):
        print("    factorial summary not found, skipping heatmaps")
        return
    df = pd.read_csv(path)
    _heatmap(df, "test_accuracy_mean", "Primary factorial: mean test accuracy",
             "factorial_accuracy_heatmap.png")
    _heatmap(df, "f1_mean", "Primary factorial: mean F1", "factorial_f1_heatmap.png")
    _heatmap(df, "mcc_mean", "Primary factorial: mean MCC", "factorial_mcc_heatmap.png")


def _grouped_bar(df, group_col, series_col, value_mean_col, value_std_col, title, fname, ylabel):
    groups = sorted(df[group_col].unique())
    series = sorted(df[series_col].unique())
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, s in enumerate(series):
        means, stds = [], []
        for g in groups:
            row = df[(df[group_col] == g) & (df[series_col] == s)]
            means.append(row[value_mean_col].iloc[0] if len(row) else np.nan)
            stds.append(row[value_std_col].iloc[0] if len(row) else 0)
        ax.bar(x + k * width - 0.4 + width / 2, means, width, yerr=stds, capsize=3, label=s)
    ax.set_xticks(x, groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, fname)


def fig_accuracy_by_group():
    path = os.path.join(rconfig.RESULTS_DIR, "factorial", "summary.csv")
    if not os.path.exists(path):
        print("    factorial summary not found, skipping grouped bars")
        return
    df = pd.read_csv(path)
    _grouped_bar(df, "feature_map", "ansatz", "test_accuracy_mean", "test_accuracy_std",
                 "Test accuracy by feature map (ansätze compared within each)",
                 "accuracy_by_feature_map.png", "Test accuracy")
    _grouped_bar(df, "ansatz", "feature_map", "test_accuracy_mean", "test_accuracy_std",
                 "Test accuracy by ansatz (feature maps compared within each)",
                 "accuracy_by_ansatz.png", "Test accuracy")


def fig_expressibility():
    path = os.path.join(rconfig.DIRS["expressibility"], "summary.csv")
    if not os.path.exists(path):
        print("    expressibility summary not found, skipping")
        return
    df = pd.read_csv(path)
    import ast
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(df))
    parsed_ci = df["kl_ci95_pooled_bootstrap"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else s)
    ci_lo = parsed_ci.apply(lambda p: p[0])
    ci_hi = parsed_ci.apply(lambda p: p[1])
    # kl_mean is the mean across repetitions, while the CI comes from a
    # separate bootstrap over the pooled fidelity samples — the two can
    # disagree enough that kl_mean falls outside [ci_lo, ci_hi]; clip so the
    # error bar never goes negative (cosmetic only, doesn't touch the data).
    yerr = np.vstack([
        np.clip(df["kl_mean"] - ci_lo, 0, None),
        np.clip(ci_hi - df["kl_mean"], 0, None),
    ])
    ax.bar(x, df["kl_mean"], yerr=yerr, capsize=4, color="#4C72B0")
    ax.set_xticks(x, df["ansatz"], rotation=15)
    ax.set_ylabel("KL divergence to Haar (lower = more expressive)")
    ax.set_title("Recognized circuit expressibility (ansatz-only)")
    _save(fig, "expressibility_kl.png")


def fig_separability():
    path = os.path.join(rconfig.DIRS["separability"], "separability_summary.csv")
    if not os.path.exists(path):
        print("    separability summary not found, skipping")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["same_mean"],
           yerr=[np.clip(df["same_mean"] - df["same_mean_ci_lo"], 0, None),
                 np.clip(df["same_mean_ci_hi"] - df["same_mean"], 0, None)],
           width=width, capsize=4, label="same-class")
    ax.bar(x + width / 2, df["diff_mean"],
           yerr=[np.clip(df["diff_mean"] - df["diff_mean_ci_lo"], 0, None),
                 np.clip(df["diff_mean_ci_hi"] - df["diff_mean"], 0, None)],
           width=width, capsize=4, label="different-class")
    ax.set_xticks(x, df["feature_map"])
    ax.set_ylabel("Mean fidelity")
    ax.set_title("Feature-space separability (data-dependent, not expressibility)")
    ax.legend()
    _save(fig, "separability_fidelity.png")


def fig_gradient_scaling():
    path = os.path.join(rconfig.DIRS["barren_plateau"], "summary.csv")
    if not os.path.exists(path):
        print("    barren-plateau summary not found, skipping")
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for ansatz in rconfig.BP_ANSATZE:
        sub = df[df["ansatz"] == ansatz].groupby("n_qubits")["aggregate_grad_variance"].mean().reset_index()
        sub = sub[sub["aggregate_grad_variance"] > 0].sort_values("n_qubits")
        if sub.empty:
            continue
        ax.plot(sub["n_qubits"], np.log10(sub["aggregate_grad_variance"]), marker="o", label=ansatz)
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("log10(aggregate gradient variance)")
    ax.set_title("Gradient variance scaling vs qubit count")
    ax.legend()
    _save(fig, "gradient_scaling_qubits.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for ansatz in rconfig.BP_ANSATZE:
        sub = df[df["ansatz"] == ansatz].groupby("depth")["aggregate_grad_variance"].mean().reset_index()
        sub = sub[sub["aggregate_grad_variance"] > 0].sort_values("depth")
        if sub.empty:
            continue
        ax.plot(sub["depth"], np.log10(sub["aggregate_grad_variance"]), marker="o", label=ansatz)
    ax.set_xlabel("Circuit depth (layers)")
    ax.set_ylabel("log10(aggregate gradient variance)")
    ax.set_title("Gradient variance scaling vs circuit depth")
    ax.legend()
    _save(fig, "gradient_scaling_depth.png")


def fig_matched_model_comparison():
    path = os.path.join(rconfig.DIRS["matched_baselines"], "unified_comparison.csv")
    if not os.path.exists(path):
        print("    unified comparison not found, skipping")
        return
    df = pd.read_csv(path).sort_values("test_accuracy_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(df))))
    colors = {"VQC": "#4C72B0", "QK-SVM": "#DD8452", "Classical": "#55A868"}
    bar_colors = [colors.get(t, "gray") for t in df["method_type"]]
    xerr = df["test_accuracy_std"].fillna(0)
    ax.barh(df["method_name"], df["test_accuracy_mean"], xerr=xerr, capsize=3, color=bar_colors)
    ax.set_xlabel("Test accuracy (primary 1,400/600 partition)")
    ax.set_title("Matched-condition model comparison")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), loc="lower right")
    _save(fig, "matched_model_comparison.png")


def fig_confusion_matrices():
    path = os.path.join(rconfig.RESULTS_DIR, "factorial", "all_runs.csv")
    out_dir = os.path.join(rconfig.FIGURES_DIR, "confusion_matrices")
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(path):
        print("    factorial all_runs.csv not found, skipping confusion matrices")
        return
    df = pd.read_csv(path)
    rep = df[df["seed"] == rconfig.BASE_SEED]
    for _, row in rep.iterrows():
        try:
            cm = np.array(json.loads(row["confusion_matrix_json"]))
        except Exception:
            continue
        fig, ax = plt.subplots(figsize=(3.2, 3))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["0", "1"])
        ax.set_yticks([0, 1], ["0", "1"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        name = f"{row['feature_map']}_{row['ansatz']}"
        ax.set_title(name, fontsize=9)
        _save(fig, f"confusion_matrices/{name}.png")


def generate_all_figures():
    print("=" * 65)
    print("  SECTION 21 — FIGURES")
    print("=" * 65)
    fig_factorial_heatmaps()
    fig_accuracy_by_group()
    fig_expressibility()
    fig_separability()
    fig_gradient_scaling()
    fig_matched_model_comparison()
    fig_confusion_matrices()
    print(f"  Figures saved to {rconfig.FIGURES_DIR}")
