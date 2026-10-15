"""
Section 20 — OPTIONAL noise-robustness experiment. Only runs after all
mandatory sections; failures here never block the rest of the package.

Noise model: a DepolarizingChannel(p) is inserted on every wire immediately
after every gate in the circuit (feature map + ansatz), using PennyLane's
`default.mixed` density-matrix simulator. This is a simplified uniform
single/two-qubit depolarizing-noise model, documented here rather than
inferred implicitly.

Reduced protocol (explicitly permitted by Section 20): fewer epochs than the
primary factorial, single seed, 3 representative configurations (best,
weakest, intermediate by primary test accuracy).
"""
import os
import time
import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn.functional as F

from revision import rconfig, utils, circuits as rcircuits, metrics as rmetrics

NOISE_EPOCHS = 10  # reduced from the primary 30 (Section 20 explicitly allows this)


def _select_representative_configs():
    path = os.path.join(rconfig.RESULTS_DIR, "factorial", "summary.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df = df.sort_values("test_accuracy_mean", ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    worst = df.iloc[-1]
    mid = df.iloc[len(df) // 2]
    return [
        ("best", best["feature_map"], best["ansatz"]),
        ("intermediate", mid["feature_map"], mid["ansatz"]),
        ("weakest", worst["feature_map"], worst["ansatz"]),
    ]


def _build_noisy_circuit(feature_map, ansatz, n_qubits, n_layers, p):
    dev = qml.device("default.mixed", wires=n_qubits)
    fmap_fn = rcircuits.FEATURE_MAP_FNS[feature_map]
    ansatz_fn = rcircuits.ANSATZ_FNS[ansatz]

    def _noisy_wrap(op_fn, *fn_args):
        # Record ops via a queuing context, then re-apply with noise after each.
        with qml.queuing.AnnotatedQueue() as q:
            op_fn(*fn_args)
        for op in q.queue:
            qml.apply(op)
            if p > 0:
                for w in op.wires:
                    qml.DepolarizingChannel(p, wires=w)

    @qml.qnode(dev)
    def circuit(x, weights):
        _noisy_wrap(fmap_fn, x, n_qubits)
        _noisy_wrap(ansatz_fn, weights, n_qubits)
        return qml.expval(qml.PauliZ(0))

    return circuit


def run_noise(resume=False, save=True):
    print("=" * 65)
    print("  SECTION 20 — OPTIONAL NOISE ROBUSTNESS")
    print("=" * 65)
    reps = _select_representative_configs()
    if reps is None:
        print("  Primary factorial results not available yet — cannot select "
              "representative configs. Optional noise experiment not executed.")
        return None

    from revision import primary_split
    import data_prep
    X, y = data_prep.load_magic()
    split = primary_split.load_primary_split(X, y)
    X_tr, y_tr = split["X_tr_q"], split["y_tr"]
    X_te, y_te = split["X_te_q"], split["y_te"]

    out_dir = rconfig.DIRS["noise"]
    jsonl_path = os.path.join(out_dir, "noise_runs.jsonl")
    done = set()
    if resume:
        done = utils.load_completed_keys(jsonl_path, ["role", "p"])

    rows = []
    try:
        for role, fmap, ansatz in reps:
            for p in rconfig.NOISE_PROBS:
                if (role, p) in done:
                    continue
                print(f"    {role} ({fmap}+{ansatz}), p={p} ...")
                t0 = time.time()
                circuit = _build_noisy_circuit(fmap, ansatz, rconfig.N_QUBITS, rconfig.N_LAYERS, p)
                shape = rcircuits.weight_shape(ansatz, rconfig.N_LAYERS, rconfig.N_QUBITS)
                utils.set_all_seeds(rconfig.BASE_SEED)
                weights = torch.nn.Parameter(
                    (torch.rand(*shape, dtype=torch.float64) * 2 - 1) * np.pi
                )
                optimizer = torch.optim.Adam([weights], lr=rconfig.LR)
                X_t = torch.tensor(X_tr, dtype=torch.float64)
                y_t = torch.tensor(y_tr, dtype=torch.float64)
                n_batches = len(X_t) // rconfig.BATCH_SIZE

                for epoch in range(NOISE_EPOCHS):
                    perm = torch.randperm(len(X_t))
                    X_s, y_s = X_t[perm], y_t[perm]
                    for b in range(n_batches):
                        Xb = X_s[b * rconfig.BATCH_SIZE:(b + 1) * rconfig.BATCH_SIZE]
                        yb = y_s[b * rconfig.BATCH_SIZE:(b + 1) * rconfig.BATCH_SIZE]
                        optimizer.zero_grad()
                        raw = circuit(Xb, weights)  # batched — see train_core.py note
                        probs = (raw + 1.0) / 2.0
                        loss = F.binary_cross_entropy(probs.float(), yb.float())
                        loss.backward()
                        optimizer.step()

                w = weights.detach()
                X_te_t = torch.tensor(X_te, dtype=torch.float64)
                with torch.no_grad():
                    raw_te = circuit(X_te_t, w)  # batched
                p_te = ((raw_te + 1.0) / 2.0).numpy()
                y_pred = (p_te > 0.5).astype(int)
                m = rmetrics.compute_full_metrics(y_te, y_pred, p_te)

                rec = {"role": role, "feature_map": fmap, "ansatz": ansatz, "p": p,
                       "status": "ok", "n_epochs": NOISE_EPOCHS,
                       "test_accuracy": m["accuracy"], "f1": m["f1"], "roc_auc": m["roc_auc"],
                       "balanced_accuracy": m["balanced_accuracy"], "mcc": m["mcc"],
                       "elapsed_seconds": time.time() - t0}
                print(f"      acc={m['accuracy']:.4f} f1={m['f1']:.4f} ({rec['elapsed_seconds']:.0f}s)")
                utils.append_jsonl(jsonl_path, rec)
                rows.append(rec)
    except Exception as e:
        print(f"  Optional noise experiment not executed due to computational cost / error: {e}")
        utils.record_failure(str(e), stage="noise")
        utils.save_json(os.path.join(out_dir, "noise_not_executed.json"),
                         {"reason": str(e), "note": "Optional noise experiment not executed due to computational cost."})
        return None

    all_rows = [r for r in utils.read_jsonl(jsonl_path) if r.get("status") == "ok"]
    if save and all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(os.path.join(out_dir, "noise_results.csv"), index=False)
        print(f"  Saved -> {out_dir}")
    return all_rows
