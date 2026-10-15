"""
Section 19 — Secondary multiple-split robustness experiment.

The PRIMARY comparison (Sections 2, 5-12) keeps one fixed 1,400/600 test
partition throughout. This experiment separately asks how sensitive results
are to *which* stratified 1,400/600 partition is drawn, using 5 different
split seeds. Preprocessing is refit on each split's training data only.

Training randomness is held at the base seed (42) for every split, so that
partition identity is the only thing varying between rows — isolating the
effect of data partition from the effect of training-initialization noise
(already characterized separately by the primary 5-seed factorial).

QK-SVM repeated splitting is omitted (Section 19 explicitly allows this) —
building a fresh statevector-Gram kernel for 5 different 1,400-sample
partitions is ~5x the cost of the already-expensive Section 11 QK-SVM
evaluation, which itself already accounts for most of the non-VQC compute
budget; the fixed-partition QK-SVM result (Section 11) stands as-is.
"""
import os
import time
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import config
import rconfig, utils, train_core, metrics as rmetrics, matched_baselines


def _build_split_for_seed(X, y, split_seed):
    y = np.asarray(y)
    rng = np.random.RandomState(split_seed)
    sub_idx = rng.choice(len(X), config.N_QUANTUM_SAMPLES, replace=False)
    all_idx = np.arange(len(sub_idx))
    train_pos, test_pos = train_test_split(
        all_idx, test_size=config.TEST_SIZE, random_state=split_seed, stratify=y[sub_idx]
    )
    train_idx, test_idx = sub_idx[train_pos], sub_idx[test_pos]
    X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scaler = StandardScaler().fit(X_tr_raw)
    X_tr_sc, X_te_sc = scaler.transform(X_tr_raw), scaler.transform(X_te_raw)
    pca = PCA(n_components=config.N_FEATURES, random_state=split_seed).fit(X_tr_sc)
    X_tr_pca, X_te_pca = pca.transform(X_tr_sc), pca.transform(X_te_sc)
    pca_scaler = StandardScaler().fit(X_tr_pca)
    X_tr_q = np.clip(pca_scaler.transform(X_tr_pca) * (np.pi / 3.0), -np.pi, np.pi)
    X_te_q = np.clip(pca_scaler.transform(X_te_pca) * (np.pi / 3.0), -np.pi, np.pi)
    return {"X_tr_q": X_tr_q, "X_te_q": X_te_q, "y_tr": y_tr, "y_te": y_te,
            "train_idx": train_idx, "test_idx": test_idx}


def _vqc_worker(args):
    (fmap, ansatz, split_seed, X_tr, y_tr, X_te, y_te) = args
    import torch
    torch.set_num_threads(1)
    try:
        r = train_core.train_one_run(
            fmap, ansatz, rconfig.N_QUBITS, rconfig.N_LAYERS, rconfig.BASE_SEED,
            X_tr, y_tr, X_te, y_te, rconfig.LR, rconfig.BATCH_SIZE, rconfig.N_EPOCHS, verbose=False,
        )
        r["split_seed"] = int(split_seed)
        r["status"] = "ok"
        return r
    except Exception as e:
        return {"feature_map": fmap, "ansatz": ansatz, "split_seed": int(split_seed), "status": "failed",
                "error": str(e)}


def run_split_robustness(X, y, resume=False):
    print("=" * 65)
    print("  SECTION 19 — SPLIT-ROBUSTNESS (secondary, does not replace primary partition)")
    print("=" * 65)
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out_dir = rconfig.DIRS["split_robustness"]
    jsonl_path = os.path.join(out_dir, "all_runs_full.jsonl")
    done = set()
    if resume:
        done = utils.load_completed_keys(jsonl_path, ["feature_map", "ansatz", "split_seed"])
        print(f"  --resume: {len(done)} runs already completed.")

    configs = list(itertools.product(rconfig.FEATURE_MAPS, rconfig.ANSATZE))
    all_jobs, split_cache = [], {}
    for split_seed in rconfig.SPLIT_SEEDS:
        split_cache[split_seed] = _build_split_for_seed(X, y, split_seed)
        for fm, an in configs:
            if (fm, an, split_seed) not in done:
                s = split_cache[split_seed]
                all_jobs.append((fm, an, split_seed, s["X_tr_q"], s["y_tr"], s["X_te_q"], s["y_te"]))

    print(f"  {len(rconfig.SPLIT_SEEDS) * len(configs)} total VQC runs, {len(all_jobs)} to run.")
    max_workers = utils.default_max_workers(len(all_jobs) or 1)

    if all_jobs:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_vqc_worker, a): (a[0], a[1], a[2]) for a in all_jobs}
            n_done = 0
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"feature_map": key[0], "ansatz": key[1], "split_seed": key[2],
                         "status": "failed", "error": str(e)}
                utils.append_jsonl(jsonl_path, r)
                n_done += 1
                status = "OK" if r.get("status") == "ok" else "FAILED"
                extra = f"acc={r.get('test_accuracy'):.4f}" if r.get("status") == "ok" else r.get("error")
                print(f"    [{n_done}/{len(all_jobs)}] {status} {key[0]}+{key[1]} split_seed={key[2]} {extra}",
                      flush=True)
                if r.get("status") != "ok":
                    utils.record_failure(r.get("error", "unknown"), stage="split_robustness",
                                          feature_map=key[0], ansatz=key[1], split_seed=key[2])

    # classical baselines per split (single run per split, no extra seed repetition)
    cls_jsonl = os.path.join(out_dir, "classical_all_runs.jsonl")
    cls_done = set()
    if resume:
        cls_done = utils.load_completed_keys(cls_jsonl, ["model", "split_seed"])
    for split_seed, s in split_cache.items():
        models = matched_baselines._make_models(split_seed)
        for name, model in models.items():
            if (name, split_seed) in cls_done:
                continue
            t0 = time.time()
            try:
                model.fit(s["X_tr_q"], s["y_tr"])
                y_pred = model.predict(s["X_te_q"])
                y_proba = model.predict_proba(s["X_te_q"])[:, 1]
                m = rmetrics.compute_full_metrics(s["y_te"], y_pred, y_proba)
                rec = {"model": name, "split_seed": int(split_seed), "status": "ok",
                       "test_accuracy": m["accuracy"], "precision": m["precision"],
                       "recall": m["recall"], "f1": m["f1"], "roc_auc": m["roc_auc"],
                       "pr_auc": m["pr_auc"], "balanced_accuracy": m["balanced_accuracy"],
                       "mcc": m["mcc"], "elapsed_seconds": time.time() - t0}
            except Exception as e:
                rec = {"model": name, "split_seed": int(split_seed), "status": "failed", "error": str(e)}
                utils.record_failure(str(e), stage="split_robustness_classical", model=name, split_seed=split_seed)
            utils.append_jsonl(cls_jsonl, rec)

    _build_split_robustness_summary(out_dir, jsonl_path, cls_jsonl)


def _build_split_robustness_summary(out_dir, jsonl_path, cls_jsonl):
    records = [r for r in utils.read_jsonl(jsonl_path) if r.get("status") == "ok"]
    cls_records = [r for r in utils.read_jsonl(cls_jsonl) if r.get("status") == "ok"]

    metric_cols = ["test_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                    "balanced_accuracy", "mcc"]

    all_rows = []
    for r in records:
        row = {"method_type": "VQC", "method_name": f"{r['feature_map']}+{r['ansatz']}",
               "split_seed": r["split_seed"]}
        row.update({c: r.get(c) for c in metric_cols})
        all_rows.append(row)
    for r in cls_records:
        row = {"method_type": "Classical", "method_name": r["model"], "split_seed": r["split_seed"]}
        row.update({c: r.get(c) for c in metric_cols if c in r})
        all_rows.append(row)

    if not all_rows:
        print("  No split-robustness results to summarize yet.")
        return
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(out_dir, "all_runs.csv"), index=False)

    present_metrics = [c for c in metric_cols if c in df.columns]
    summ = df.groupby(["method_type", "method_name"])[present_metrics].agg(["mean", "std", "count"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(f"  Split-robustness summary saved -> {out_dir}")
