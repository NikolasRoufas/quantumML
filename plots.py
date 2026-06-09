"""
Publication-quality figures matching the methodology's figure descriptions.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import config

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

Q_COLORS = ["#1565C0", "#0277BD", "#00838F"]
C_COLORS = ["#B71C1C", "#880E4F", "#4A148C"]


# ── Training convergence ──────────────────────────────────────────────────────

def plot_training_convergence(histories):
    fig, ax = plt.subplots(figsize=(8, 5))
    for (label, hist), color in zip(histories.items(), Q_COLORS):
        epochs = range(1, len(hist["loss"]) + 1)
        ax.plot(epochs, hist["loss"], label=label, color=color, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("Training Convergence of Quantum Models\n(MAGIC Gamma Telescope)")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    _save(fig, "training_convergence.png")
    return fig


# ── Feature map fidelity ──────────────────────────────────────────────────────

def plot_feature_map_comparison(fidelities):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (fname, fid) in zip(axes, fidelities.items()):
        bins = np.linspace(0, 1, 30)
        ax.hist(fid["same"], bins=bins, alpha=0.70, label="Same class",
                color="#1E88E5", density=True)
        ax.hist(fid["diff"], bins=bins, alpha=0.70, label="Diff. class",
                color="#E53935", density=True)
        ax.set_title(f"{fname} Feature Map")
        ax.set_xlabel("Quantum State Fidelity  $|\\langle\\psi_i|\\psi_j\\rangle|^2$")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle="--")
    fig.suptitle("Fidelity Distribution by Feature Map", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "feature_map_comparison.png")
    return fig


# ── Barren plateau ────────────────────────────────────────────────────────────

def plot_gradient_analysis(barren_stats):
    depths = sorted(barren_stats.keys())
    means  = [barren_stats[d]["mean"] for d in depths]
    stds   = [barren_stats[d]["std"]  for d in depths]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        depths, means, yerr=stds,
        marker="o", linewidth=2, capsize=5,
        color="#E91E63", markersize=8, label="Mean gradient norm"
    )
    ax.set_xlabel("Number of Variational Layers")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Barren Plateau: Gradient Norm vs Circuit Depth\n"
                 "(ZZ + Strongly Entangling Ansatz, 4 qubits)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--", which="both")
    ax.legend()
    plt.tight_layout()
    _save(fig, "gradient_analysis.png")
    return fig


# ── Accuracy comparison ───────────────────────────────────────────────────────

def plot_accuracy_comparison(quantum_results, classical_results):
    q_names = list(quantum_results.keys())
    c_names = list(classical_results.keys())
    q_accs  = [quantum_results[n]["test_acc"]   for n in q_names]
    c_accs  = [classical_results[n]["test_acc"] for n in c_names]

    names  = q_names  + c_names
    accs   = q_accs   + c_accs
    colors = Q_COLORS + C_COLORS[: len(c_names)]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(
        names, [a * 100 for a in accs],
        color=colors, alpha=0.85, edgecolor="black", linewidth=0.5
    )
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_width() + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{acc*100:.1f}%",
            va="center", fontsize=10,
        )
    ax.set_xlabel("Test Accuracy (%)")
    ax.set_title("Classification Accuracy: Quantum vs Classical\n"
                 "(MAGIC Gamma Telescope — 19,020 samples)")
    ax.set_xlim(0, 108)
    legend_elements = [
        Patch(facecolor=Q_COLORS[0], alpha=0.85, label="Quantum"),
        Patch(facecolor=C_COLORS[0], alpha=0.85, label="Classical"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--", axis="x")
    plt.tight_layout()
    _save(fig, "accuracy_comparison.png")
    return fig


# ── Full metrics grouped bar chart ───────────────────────────────────────────

def plot_full_metrics(all_metrics):
    metric_keys  = ["accuracy", "f1", "precision", "recall"]
    metric_labels = ["Accuracy", "F1", "Precision", "Recall"]
    model_names  = list(all_metrics.keys())
    x = np.arange(len(metric_keys))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(13, 5))
    palette = Q_COLORS + C_COLORS + ["#558B2F", "#F57F17", "#4E342E"]
    for i, (name, m) in enumerate(all_metrics.items()):
        vals = [m.get(k, 0) for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=palette[i % len(palette)], alpha=0.85, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Full Classification Metrics: Quantum vs Classical vs Quantum Kernel\n"
                 "(MAGIC Gamma Telescope)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    plt.tight_layout()
    _save(fig, "full_metrics.png")
    return fig


# ── Qubit scalability ─────────────────────────────────────────────────────────

def plot_scalability(scale_results):
    n_qubits = sorted(scale_results.keys())
    accs     = [scale_results[n]["accuracy"]  for n in n_qubits]
    params   = [scale_results[n]["n_params"]  for n in n_qubits]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.plot(n_qubits, accs,   "o-", color="#1565C0", linewidth=2.5, markersize=9, label="Test Accuracy")
    ax2.plot(n_qubits, params, "s--", color="#E65100", linewidth=2, markersize=8, label="# Parameters")

    ax1.set_xlabel("Number of Qubits")
    ax1.set_ylabel("Test Accuracy", color="#1565C0")
    ax2.set_ylabel("Trainable Parameters", color="#E65100")
    ax1.tick_params(axis="y", labelcolor="#1565C0")
    ax2.tick_params(axis="y", labelcolor="#E65100")
    ax1.set_title("Scalability: Accuracy vs Number of Qubits\n"
                  "(ZZ + Strongly Entangling, 2 layers)")
    ax1.set_xticks(n_qubits)
    ax1.grid(True, alpha=0.3, linestyle="--")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    plt.tight_layout()
    _save(fig, "scalability.png")
    return fig


# ── Expressibility ────────────────────────────────────────────────────────────

def plot_expressibility(expr_results):
    names = list(expr_results.keys())
    kl_vals = [expr_results[n]["kl"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: KL divergence bar chart
    ax = axes[0]
    colors = Q_COLORS
    bars = ax.bar(range(len(names)), kl_vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, kl in zip(bars, kl_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{kl:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" + ", "\n+\n") for n in names], fontsize=9)
    ax.set_ylabel("KL Divergence from Haar")
    ax.set_title("Expressibility (lower = more expressive)")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Right: fidelity distributions
    ax2 = axes[1]
    bins = np.linspace(0, 1, 50)
    for name, color in zip(names, colors):
        ax2.hist(expr_results[name]["fidelities"], bins=bins, alpha=0.5,
                 density=True, label=name, color=color)
    # Haar reference
    f_vals = np.linspace(0, 1, 200)
    d = 2 ** 4
    haar = (d - 1) * (1 - f_vals) ** (d - 2)
    ax2.plot(f_vals, haar, "k--", linewidth=2, label="Haar (reference)")
    ax2.set_xlabel("Fidelity  $|\\langle\\psi(\\theta_1)|\\psi(\\theta_2)\\rangle|^2$")
    ax2.set_ylabel("Density")
    ax2.set_title("Fidelity Distribution vs Haar Distribution")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle("Expressibility Analysis of Variational Ansätze", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "expressibility.png")
    return fig


# ── Console tables ────────────────────────────────────────────────────────────

def print_results_table(quantum_results, classical_results):
    w = 40
    print("\n" + "=" * (w + 14), flush=True)
    print(f"  {'Model':<{w}} {'Test Acc':>10}", flush=True)
    print("=" * (w + 14), flush=True)
    for name, r in quantum_results.items():
        print(f"  {name:<{w}} {r['test_acc']*100:>9.1f}%", flush=True)
    print("-" * (w + 14), flush=True)
    for name, r in classical_results.items():
        print(f"  {name:<{w}} {r['test_acc']*100:>9.1f}%", flush=True)
    print("=" * (w + 14), flush=True)


def print_complexity_table(complexity):
    print("\n" + "=" * 68, flush=True)
    print(f"  {'Configuration':<36} {'Gates':>6} {'Depth':>6} {'Params':>7}", flush=True)
    print("=" * 68, flush=True)
    for name, (gates, depth, params) in complexity.items():
        print(f"  {name:<36} {gates:>6} {depth:>6} {params:>7}", flush=True)
    print("=" * 68, flush=True)


# ── helper ────────────────────────────────────────────────────────────────────

def _save(fig, filename):
    path = os.path.join(config.FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}", flush=True)
    plt.close(fig)
