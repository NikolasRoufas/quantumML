"""
Main experiment runner — recreates and extends all results from the methodology.

Dataset : MAGIC Gamma Telescope (UCI / OpenML)
          19,020 real astrophysics samples, 10 features, binary classification

Experiments
-----------
1.  Quantum variational classifiers (ZZ+SE, Amplitude+BE, Angle+Custom)
2.  Classical baselines (SVM-RBF, Random Forest, MLP) on full dataset
3.  Quantum Kernel SVM (ZZ / Amplitude / Angle kernels)
4.  Qubit scalability  (n_qubits in {2,3,4,5}, ZZ+SE circuit)
5.  Circuit complexity  table
6.  Barren plateau      analysis
7.  Feature map fidelity analysis
8.  Expressibility      analysis (KL divergence to Haar)

Run:  python run_experiments.py
"""
import os
import json
import time
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

import config
import data_prep
import quantum_circuits as qc
import quantum_trainer as qt
import classical_baselines as cb
import analysis
import quantum_kernel as qk
import scalability as sc
import extended_metrics as em
import plots

t0_total = time.time()
sep = "=" * 65


def elapsed():
    return f"{(time.time() - t0_total)/60:.1f} min"


# ── 1. Data ───────────────────────────────────────────────────────────────────
print(sep, flush=True)
print("  QUANTUM ML EXPERIMENTS — MAGIC GAMMA TELESCOPE", flush=True)
print(sep, flush=True)

print(f"\n[1/8] Loading and preparing data...", flush=True)
X, y = data_prep.load_magic()
X_tr_cl, X_te_cl, y_tr_cl, y_te_cl = data_prep.prepare_classical(X, y)
X_tr_q,  X_te_q,  y_tr_q,  y_te_q, _, _ = data_prep.prepare_quantum(X, y)


# ── 2. Quantum variational classifiers ───────────────────────────────────────
print(f"\n[2/8] Training quantum variational classifiers...", flush=True)
quantum_configs = [
    ("ZZ + StronglyEnt",     qc.zz_strongly_circuit,    qc.ZZ_SHAPE),
    ("Amplitude + BasicEnt", qc.amplitude_basic_circuit, qc.AMP_SHAPE),
    ("Angle + CustomAnsatz", qc.angle_custom_circuit,    qc.ANGLE_SHAPE),
]

quantum_results = {}
histories       = {}
for name, circuit, shape in quantum_configs:
    w, hist, tr_acc, te_acc, te_proba = qt.train_quantum_model(
        circuit, shape, X_tr_q, y_tr_q, X_te_q, y_te_q, model_name=name
    )
    y_pred_q = (te_proba > 0.5).astype(int)
    metrics_q = em.compute_metrics(y_te_q, y_pred_q, te_proba)
    quantum_results[name] = {
        "weights":   w,
        "train_acc": float(tr_acc),
        "test_acc":  float(te_acc),
        **{k: v for k, v in metrics_q.items()},
    }
    histories[name] = hist


# ── 3. Classical baselines ────────────────────────────────────────────────────
print(f"\n[3/8] Training classical baselines (full {len(X):,}-sample dataset)...", flush=True)
classical_raw = cb.train_classical_models(X_tr_cl, y_tr_cl, X_te_cl, y_te_cl)

classical_results = {}
for name, r in classical_raw.items():
    model = r["model"]
    y_pred_c = model.predict(X_te_cl)
    try:
        y_proba_c = model.predict_proba(X_te_cl)[:, 1]
    except AttributeError:
        y_proba_c = None
    metrics_c = em.compute_metrics(y_te_cl, y_pred_c, y_proba_c)
    classical_results[name] = {
        "train_acc": r["train_acc"],
        "test_acc":  r["test_acc"],
        **metrics_c,
    }


# ── 4. Quantum kernel SVMs ────────────────────────────────────────────────────
print(f"\n[4/8] Quantum kernel SVM experiments...", flush=True)
kernel_results, _ = qk.run_quantum_kernel_svms(
    X_tr_q, X_te_q, y_tr_q, y_te_q, n_train=300, n_test=150
)


# ── 5. Qubit scalability ──────────────────────────────────────────────────────
print(f"\n[5/8] Qubit scalability experiment (n_qubits 2-5)...", flush=True)
scale_results = sc.run_scalability(X, y, qubit_range=[2, 3, 4, 5])


# ── 6. Circuit complexity ─────────────────────────────────────────────────────
print(f"\n[6/8] Circuit complexity analysis...", flush=True)
complexity = analysis.compute_circuit_complexity()
plots.print_complexity_table(complexity)


# ── 7. Analyses ───────────────────────────────────────────────────────────────
print(f"\n[7/8] Barren plateau + fidelity + expressibility analyses...", flush=True)
print("  Barren plateau:", flush=True)
barren_stats = analysis.barren_plateau_analysis(depths=[1, 2, 3, 4, 5, 6], n_init=30)

print("  Feature map fidelity:", flush=True)
fidelities = analysis.compute_fidelities(X_tr_q, y_tr_q, n_samples=200, n_pairs=400)

print("  Expressibility (KL to Haar):", flush=True)
expr_results = analysis.compute_expressibility(n_samples=200)


# ── 8. Figures ────────────────────────────────────────────────────────────────
print(f"\n[8/8] Generating figures...", flush=True)
plots.plot_training_convergence(histories)
plots.plot_feature_map_comparison(fidelities)
plots.plot_gradient_analysis(barren_stats)
plots.plot_accuracy_comparison(quantum_results, classical_results)
plots.plot_scalability(scale_results)
plots.plot_expressibility(expr_results)

all_metrics = {**quantum_results, **kernel_results, **classical_results}
plots.plot_full_metrics(all_metrics)


# ── Summary tables ────────────────────────────────────────────────────────────
print("\n-- VARIATIONAL QUANTUM + CLASSICAL --", flush=True)
plots.print_results_table(quantum_results, classical_results)

print("\n-- QUANTUM KERNEL SVM --", flush=True)
em.print_metrics_table(kernel_results)

print("\n-- FULL METRICS (all models) --", flush=True)
em.print_metrics_table(all_metrics)


# ── Save JSON ─────────────────────────────────────────────────────────────────
def _strip(d):
    return {k: v for k, v in d.items() if not hasattr(v, "shape")}

out = {
    "dataset": "MAGIC Gamma Telescope (UCI/OpenML) — 19,020 samples, 10 features",
    "quantum_samples": config.N_QUANTUM_SAMPLES,
    "n_qubits": config.N_QUBITS,
    "n_layers": config.N_LAYERS,
    "n_epochs": config.N_EPOCHS,
    "quantum_variational": {k: _strip(v) for k, v in quantum_results.items()},
    "classical": {k: _strip(v) for k, v in classical_results.items()},
    "quantum_kernel": kernel_results,
    "scalability": {str(k): v for k, v in scale_results.items()},
    "circuit_complexity": {
        k: {"gates": g, "depth": d, "params": p}
        for k, (g, d, p) in complexity.items()
    },
    "barren_plateau": {str(k): v for k, v in barren_stats.items()},
    "expressibility": {k: {"kl": v["kl"]} for k, v in expr_results.items()},
}

results_path = os.path.join(config.RESULTS_DIR, "results.json")
with open(results_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nResults saved to {results_path}", flush=True)
print(f"Figures saved to {config.FIGURES_DIR}/", flush=True)
print(f"\nTotal runtime: {elapsed()}", flush=True)
print("Done.", flush=True)
