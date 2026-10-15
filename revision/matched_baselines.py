"""
Sections 10-11 — Controlled classical baselines and quantum-kernel baselines
on the exact primary 1,400/600 partition.

Classical models run on the SAME PCA-4 representation the VQCs see
(split['X_tr_q']/['X_te_q'], pre-clip-to-[-pi,pi] PCA-transformed features)
so the comparison isolates model family, not feature availability — this
choice is recorded explicitly in the saved metadata.
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVC as _SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from revision import rconfig, utils, metrics as rmetrics, circuits as rcircuits


# ── Section 10 — classical baselines ────────────────────────────────────────

def _make_models(seed):
    return {
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", random_state=seed, probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500,
                                         learning_rate_init=0.01, random_state=seed),
    }


def run_classical(split, resume=False):
    print("=" * 65)
    print("  SECTION 10 — CONTROLLED CLASSICAL BASELINES (matched partition)")
    print("=" * 65)
    out_dir = rconfig.DIRS["matched_baselines"]
    jsonl_path = os.path.join(out_dir, "classical_all_runs.jsonl")

    done = set()
    if resume:
        done = utils.load_completed_keys(jsonl_path, ["model", "seed"])

    X_tr, y_tr = split["X_tr_q"], split["y_tr"]
    X_te, y_te = split["X_te_q"], split["y_te"]

    rows = []
    for seed in rconfig.PRIMARY_SEEDS:
        models = _make_models(seed)
        for name, model in models.items():
            key = (name, seed)
            if key in done:
                continue
            t0 = time.time()
            try:
                model.fit(X_tr, y_tr)
                y_pred_tr = model.predict(X_tr)
                y_pred_te = model.predict(X_te)
                proba_te = model.predict_proba(X_te)[:, 1]
                tr_m = rmetrics.compute_full_metrics(y_tr, y_pred_tr)
                te_m = rmetrics.compute_full_metrics(y_te, y_pred_te, proba_te)
                rec = {
                    "model": name, "seed": int(seed), "status": "ok",
                    "n_train": len(X_tr), "n_test": len(X_te),
                    "training_accuracy": tr_m["accuracy"],
                    "test_accuracy": te_m["accuracy"], "precision": te_m["precision"],
                    "recall": te_m["recall"], "f1": te_m["f1"], "roc_auc": te_m["roc_auc"],
                    "pr_auc": te_m["pr_auc"], "balanced_accuracy": te_m["balanced_accuracy"],
                    "mcc": te_m["mcc"], "confusion_matrix": te_m["confusion_matrix"],
                    "elapsed_seconds": time.time() - t0,
                    "feature_representation": "PCA-4 (same as VQC), fit on training partition only",
                }
                print(f"    {name} seed={seed}: acc={te_m['accuracy']:.4f} "
                      f"f1={te_m['f1']:.4f} ({rec['elapsed_seconds']:.1f}s)")
            except Exception as e:
                rec = {"model": name, "seed": int(seed), "status": "failed", "error": str(e)}
                utils.record_failure(str(e), stage="classical_baselines", model=name, seed=seed)
                print(f"    {name} seed={seed}: FAILED {e}")
            utils.append_jsonl(jsonl_path, rec)
            rows.append(rec)

    all_rows = utils.read_jsonl(jsonl_path)
    ok = [r for r in all_rows if r.get("status") == "ok"]
    if ok:
        df = pd.DataFrame(ok)
        df.to_csv(os.path.join(out_dir, "classical_all_runs.csv"), index=False)
        metric_cols = ["test_accuracy", "precision", "recall", "f1", "roc_auc",
                        "pr_auc", "balanced_accuracy", "mcc"]
        summ = df.groupby("model")[metric_cols].agg(["mean", "std", "count"])
        summ.columns = ["_".join(c) for c in summ.columns]
        summ = summ.reset_index()
        summ.to_csv(os.path.join(out_dir, "classical_summary.csv"), index=False)
        print(f"  Classical baselines saved -> {out_dir}")
    return ok


# ── Section 11 — quantum-kernel baselines (efficient statevector-Gram) ─────

def run_qkernel(split, resume=False):
    print("=" * 65)
    print("  SECTION 11 — CONTROLLED QUANTUM-KERNEL BASELINES (matched partition)")
    print("=" * 65)
    out_dir = rconfig.DIRS["matched_baselines"]
    jsonl_path = os.path.join(out_dir, "qkernel_all_runs.jsonl")

    X_tr, y_tr = split["X_tr_q"], split["y_tr"]
    X_te, y_te = split["X_te_q"], split["y_te"]
    n_tr, n_te = len(X_tr), len(X_te)

    done = set()
    if resume:
        done = utils.load_completed_keys(jsonl_path, ["feature_map"])

    results = []
    for fmap in rconfig.FEATURE_MAPS:
        if (fmap,) in done:
            print(f"    {fmap}: already done, skipping (--resume)")
            continue
        print(f"    {fmap} kernel: computing statevectors for {n_tr}+{n_te} samples...")
        t0 = time.time()
        state_circ = rcircuits.feature_map_only_state_circuit(fmap, n_qubits=rconfig.N_QUBITS)

        try:
            # Batched statevector computation via PennyLane broadcasting —
            # mathematically identical to one call per sample (verified:
            # max abs diff 0.0 against the per-sample loop), ~200x fewer
            # wall-clock seconds from eliminating per-call tape-build overhead.
            tr_states = np.asarray(state_circ(X_tr))
            te_states = np.asarray(state_circ(X_te))
            t_states = time.time() - t0

            K_tr = np.abs(tr_states.conj() @ tr_states.T) ** 2
            K_te = np.abs(te_states.conj() @ tr_states.T) ** 2
            t_gram = time.time() - t0 - t_states

            svm = SVC(kernel="precomputed", C=1.0, probability=True, random_state=rconfig.BASE_SEED)
            svm.fit(K_tr, y_tr)
            y_pred = svm.predict(K_te)
            y_proba = svm.predict_proba(K_te)[:, 1]
            m = rmetrics.compute_full_metrics(y_te, y_pred, y_proba)

            rec = {
                "feature_map": fmap, "status": "ok",
                "method": "statevector-Gram (single statevector per sample, "
                          "K[i,j]=|<psi_i|psi_j>|^2) — mathematically equivalent to the "
                          "naive per-pair fidelity kernel, computed once per sample instead "
                          "of once per pair.",
                "n_train": n_tr, "n_test": n_te,
                "test_accuracy": m["accuracy"], "precision": m["precision"],
                "recall": m["recall"], "f1": m["f1"], "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"], "balanced_accuracy": m["balanced_accuracy"],
                "mcc": m["mcc"], "confusion_matrix": m["confusion_matrix"],
                "statevector_time_seconds": t_states, "gram_and_svm_time_seconds": t_gram,
                "total_elapsed_seconds": time.time() - t0,
            }
            print(f"      acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f} "
                  f"(states {t_states:.1f}s, gram+svm {t_gram:.1f}s)")
        except Exception as e:
            rec = {"feature_map": fmap, "status": "failed", "error": str(e)}
            utils.record_failure(str(e), stage="qkernel", feature_map=fmap)
            print(f"      FAILED: {e}")

        utils.append_jsonl(jsonl_path, rec)
        results.append(rec)

    all_rows = utils.read_jsonl(jsonl_path)
    ok = [r for r in all_rows if r.get("status") == "ok"]
    if ok:
        df = pd.DataFrame(ok)
        df.to_csv(os.path.join(out_dir, "qkernel_results.csv"), index=False)
        metric_cols = ["test_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                        "balanced_accuracy", "mcc"]
        df[["feature_map"] + metric_cols].to_csv(
            os.path.join(out_dir, "qkernel_summary.csv"), index=False)
        print(f"  QK-SVM results saved -> {out_dir}")
    return ok
