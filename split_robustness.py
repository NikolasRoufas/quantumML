"""
Section 19 (round-2 reviewer response) — Repeated train/test split
robustness, addressing reviewer concerns A (repeated splits), B
(reproducibility artifacts), and the QK-SVM omission.

Two variants are run, explicitly separating the two sources of variability
the original single-variant design conflated:

  FIXED-SUBSET  (primary, new): the 2,000-sample quantum data subsample is
    drawn ONCE (RandomState(config.SEED), identical to primary_split.py's
    subsample), and only the stratified 1,400/600 train/test PARTITION of
    that fixed subsample varies across SPLIT_SEEDS. This isolates partition
    identity as the only source of variation.

  VARYING-SUBSET (secondary, sensitivity — the original design): both the
    2,000-sample subsample AND the partition are redrawn together per split
    seed. Retained for comparability with the original revision, now
    explicitly labeled as a sensitivity analysis rather than the primary
    robustness evidence.

For every (variant, model, split[, init_seed]) run we save: split seed,
init seed (where applicable), training indices, test indices, true labels,
predicted labels, prediction scores, the complete confusion matrix, and
every computed metric — satisfying the reproducibility-artifact requirement
(concern B) in full, not just aggregated statistics.

QK-SVM (previously omitted from repeated splits — concern A/the explicit
instruction to fix this) is run on the FIXED-SUBSET variant only, once per
split seed, reusing matched_baselines.run_qkernel's exact statevector-Gram
implementation (3 kernels x 5 splits = 15 runs) rather than duplicating it.
It is not repeated for the varying-subset sensitivity variant: that would
be a second full QK-SVM sweep for a result explicitly scoped as secondary,
and the reviewer's repeated-split QK-SVM requirement is fully satisfied by
the primary (fixed-subset) variant.
"""
import os
import time
import itertools
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import config
import rconfig, utils, train_core, metrics as rmetrics, matched_baselines


# ── split construction ───────────────────────────────────────────────────

def _fit_quantum_preprocessing(X_tr_raw, X_te_raw, fit_seed):
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr_sc, X_te_sc = scaler.transform(X_tr_raw), scaler.transform(X_te_raw)
    pca = PCA(n_components=config.N_FEATURES, random_state=fit_seed).fit(X_tr_sc)
    X_tr_pca, X_te_pca = pca.transform(X_tr_sc), pca.transform(X_te_sc)
    pca_scaler = StandardScaler().fit(X_tr_pca)
    X_tr_q = np.clip(pca_scaler.transform(X_tr_pca) * (np.pi / 3.0), -np.pi, np.pi)
    X_te_q = np.clip(pca_scaler.transform(X_te_pca) * (np.pi / 3.0), -np.pi, np.pi)
    return X_tr_q, X_te_q


def get_fixed_subsample_idx(X):
    """The same 2,000-row subsample as primary_split.py / data_prep.prepare_quantum
    (RandomState(config.SEED)) — the canonical fixed subset for the primary
    repeated-PARTITION-only robustness analysis."""
    rng = np.random.RandomState(config.SEED)
    return rng.choice(len(X), config.N_QUANTUM_SAMPLES, replace=False)


def _build_fixed_subset_split(X, y, subsample_idx, split_seed):
    y = np.asarray(y)
    all_pos = np.arange(len(subsample_idx))
    train_pos, test_pos = train_test_split(
        all_pos, test_size=config.TEST_SIZE, random_state=split_seed,
        stratify=y[subsample_idx],
    )
    train_idx, test_idx = subsample_idx[train_pos], subsample_idx[test_pos]
    X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    X_tr_q, X_te_q = _fit_quantum_preprocessing(X_tr_raw, X_te_raw, split_seed)
    return {"X_tr_q": X_tr_q, "X_te_q": X_te_q, "y_tr": y_tr, "y_te": y_te,
            "train_idx": train_idx, "test_idx": test_idx, "split_seed": int(split_seed),
            "variant": "fixed_subset"}


def _build_varying_subset_split(X, y, split_seed):
    y = np.asarray(y)
    rng = np.random.RandomState(split_seed)
    sub_idx = rng.choice(len(X), config.N_QUANTUM_SAMPLES, replace=False)
    all_pos = np.arange(len(sub_idx))
    train_pos, test_pos = train_test_split(
        all_pos, test_size=config.TEST_SIZE, random_state=split_seed, stratify=y[sub_idx]
    )
    train_idx, test_idx = sub_idx[train_pos], sub_idx[test_pos]
    X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    X_tr_q, X_te_q = _fit_quantum_preprocessing(X_tr_raw, X_te_raw, split_seed)
    return {"X_tr_q": X_tr_q, "X_te_q": X_te_q, "y_tr": y_tr, "y_te": y_te,
            "train_idx": train_idx, "test_idx": test_idx, "split_seed": int(split_seed),
            "variant": "varying_subset"}


def _validate_split(s):
    """Section 2.5 checks, enforced at construction time rather than only
    audited afterward: disjointness and expected sizes."""
    overlap = set(s["train_idx"].tolist()) & set(s["test_idx"].tolist())
    assert not overlap, f"train/test index overlap in {s['variant']} split_seed={s['split_seed']}: {overlap}"
    assert len(s["train_idx"]) == rconfig.PRIMARY_N_TRAIN, len(s["train_idx"])
    assert len(s["test_idx"]) == rconfig.PRIMARY_N_TEST, len(s["test_idx"])
    return s


# ── VQC worker ────────────────────────────────────────────────────────────

def _vqc_worker(args):
    (variant, fmap, ansatz, split_seed, init_seed, X_tr, y_tr, X_te, y_te,
     train_idx, test_idx) = args
    # See the matching comment in factorial.py's _worker: clear CUDA out of
    # reach before torch is imported, rather than relying on nothing in the
    # stack ever probing it — this is what the remaining ~15/135 CUDA/MPS
    # failures (after removing the explicit manual_seed_all call) traced to.
    import os as _os
    _os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(1)
    try:
        r = train_core.train_one_run(
            fmap, ansatz, rconfig.N_QUBITS, rconfig.N_LAYERS, init_seed,
            X_tr, y_tr, X_te, y_te, rconfig.LR, rconfig.BATCH_SIZE, rconfig.N_EPOCHS, verbose=False,
        )
        p_te = np.asarray(r.pop("test_probabilities"))
        y_pred_te = (p_te > 0.5).astype(int)
        r.update({
            "variant": variant, "split_seed": int(split_seed), "init_seed": int(init_seed),
            "status": "ok",
            "train_idx": np.asarray(train_idx).tolist(), "test_idx": np.asarray(test_idx).tolist(),
            "y_true": np.asarray(y_te).tolist(), "y_pred": y_pred_te.tolist(),
            "y_proba": p_te.tolist(),
        })
        r.pop("train_probabilities", None)  # keep record size reasonable; test-side is what's scored
        return r
    except Exception as e:
        return {"variant": variant, "feature_map": fmap, "ansatz": ansatz, "split_seed": int(split_seed),
                "init_seed": int(init_seed), "status": "failed", "error": str(e)}


def _run_vqc_grid(variant, split_cache, jsonl_path, resume):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    done = set()
    if resume:
        done = utils.load_completed_keys(jsonl_path, ["feature_map", "ansatz", "split_seed", "init_seed"])
        print(f"  [{variant}] --resume: {len(done)} VQC runs already completed.")

    configs = list(itertools.product(rconfig.FEATURE_MAPS, rconfig.ANSATZE))
    jobs = []
    for split_seed, s in split_cache.items():
        for fm, an in configs:
            for init_seed in rconfig.SPLIT_ROBUSTNESS_INIT_SEEDS:
                if (fm, an, split_seed, init_seed) in done:
                    continue
                jobs.append((variant, fm, an, split_seed, init_seed,
                             s["X_tr_q"], s["y_tr"], s["X_te_q"], s["y_te"],
                             s["train_idx"], s["test_idx"]))

    total = len(configs) * len(split_cache) * len(rconfig.SPLIT_ROBUSTNESS_INIT_SEEDS)
    print(f"  [{variant}] {total} total VQC runs ({len(configs)} configs x {len(split_cache)} splits x "
          f"{len(rconfig.SPLIT_ROBUSTNESS_INIT_SEEDS)} inits), {len(jobs)} to run.")
    if not jobs:
        return
    # Hard-capped, not just reduced: under mp_context="spawn" each worker
    # boots a full fresh interpreter and re-imports torch/pennylane/sklearn/
    # pandas, far heavier than a forked worker. Launching dozens of those
    # simultaneously in one ProcessPoolExecutor batch produced a steady
    # ~15/135-run failure rate ("process in the process pool was terminated
    # abruptly") across three separate retries — including one retry where
    # the pending count itself was only 30, so every failing job had its own
    # dedicated worker with no queueing contention, ruling out a per-run
    # circuit/data bug. A small fixed pool trades some wall-clock time
    # (each run is ~15-30s; total compute here is cheap either way) for
    # eliminating the simultaneous-spawn burst entirely.
    max_workers = min(16, len(jobs))
    # mp_context="spawn": see the matching comment in factorial.py — fork()
    # workers crash on the first .backward() call once anything earlier in
    # this process has touched torch's autograd engine (e.g. a prior stage
    # in a combined `--stage all` run); spawn avoids that regardless of
    # execution order.
    with ProcessPoolExecutor(max_workers=max_workers,
                              mp_context=multiprocessing.get_context("spawn")) as ex:
        futures = {ex.submit(_vqc_worker, a): (a[1], a[2], a[3], a[4]) for a in jobs}
        n_done = 0
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"variant": variant, "feature_map": key[0], "ansatz": key[1], "split_seed": key[2],
                     "init_seed": key[3], "status": "failed", "error": str(e)}
            utils.append_jsonl(jsonl_path, r)
            n_done += 1
            status = "OK" if r.get("status") == "ok" else "FAILED"
            extra = f"acc={r.get('test_accuracy'):.4f}" if r.get("status") == "ok" else r.get("error")
            print(f"    [{variant} {n_done}/{len(jobs)}] {status} {key[0]}+{key[1]} "
                  f"split_seed={key[2]} init_seed={key[3]} {extra}", flush=True)
            if r.get("status") != "ok":
                utils.record_failure(r.get("error", "unknown"), stage=f"split_robustness_{variant}",
                                      feature_map=key[0], ansatz=key[1], split_seed=key[2], init_seed=key[3])


# ── classical baselines per split ────────────────────────────────────────

def _run_classical_grid(variant, split_cache, jsonl_path, resume):
    done = set()
    if resume:
        done = utils.load_completed_keys(jsonl_path, ["model", "split_seed"])
    for split_seed, s in split_cache.items():
        models = matched_baselines._make_models(split_seed)
        for name, model in models.items():
            if (name, split_seed) in done:
                continue
            t0 = time.time()
            try:
                model.fit(s["X_tr_q"], s["y_tr"])
                y_pred = model.predict(s["X_te_q"])
                y_proba = model.predict_proba(s["X_te_q"])[:, 1]
                m = rmetrics.compute_full_metrics(s["y_te"], y_pred, y_proba)
                rec = {"variant": variant, "model": name, "split_seed": int(split_seed), "status": "ok",
                       "train_idx": np.asarray(s["train_idx"]).tolist(),
                       "test_idx": np.asarray(s["test_idx"]).tolist(),
                       "y_true": np.asarray(s["y_te"]).tolist(), "y_pred": np.asarray(y_pred).tolist(),
                       "y_proba": np.asarray(y_proba).tolist(),
                       "test_accuracy": m["accuracy"], "precision": m["precision"],
                       "recall": m["recall"], "f1": m["f1"], "roc_auc": m["roc_auc"],
                       "pr_auc": m["pr_auc"], "balanced_accuracy": m["balanced_accuracy"],
                       "mcc": m["mcc"], "confusion_matrix": m["confusion_matrix"],
                       "elapsed_seconds": time.time() - t0}
                print(f"    [{variant} classical] {name} split_seed={split_seed}: acc={m['accuracy']:.4f}")
            except Exception as e:
                rec = {"variant": variant, "model": name, "split_seed": int(split_seed),
                       "status": "failed", "error": str(e)}
                utils.record_failure(str(e), stage=f"split_robustness_classical_{variant}",
                                      model=name, split_seed=split_seed)
            utils.append_jsonl(jsonl_path, rec)


# ── orchestration ─────────────────────────────────────────────────────────

def run_split_robustness(X, y, resume=False):
    print("=" * 65)
    print("  SECTION 19 (ROUND-2) — REPEATED-SPLIT ROBUSTNESS")
    print("  Primary: fixed-subset (partition-only variability) + QK-SVM")
    print("  Secondary/sensitivity: varying-subset (original design)")
    print("=" * 65)
    out_dir = rconfig.DIRS["split_robustness"]
    os.makedirs(os.path.join(out_dir, "fixed_subset"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "varying_subset"), exist_ok=True)

    fixed_sub_idx = get_fixed_subsample_idx(X)
    fixed_cache = {s: _validate_split(_build_fixed_subset_split(X, y, fixed_sub_idx, s))
                   for s in rconfig.SPLIT_SEEDS}
    varying_cache = {s: _validate_split(_build_varying_subset_split(X, y, s))
                      for s in rconfig.SPLIT_SEEDS}

    # save index/validation artifacts up front (Section 2.5)
    index_report = {"fixed_subset": {}, "varying_subset": {}}
    for variant, cache in [("fixed_subset", fixed_cache), ("varying_subset", varying_cache)]:
        for split_seed, s in cache.items():
            d = os.path.join(out_dir, variant, "split_indices")
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, f"train_idx_seed{split_seed}.npy"), s["train_idx"])
            np.save(os.path.join(d, f"test_idx_seed{split_seed}.npy"), s["test_idx"])
            index_report[variant][str(split_seed)] = {
                "n_train": len(s["train_idx"]), "n_test": len(s["test_idx"]),
                "disjoint": len(set(s["train_idx"].tolist()) & set(s["test_idx"].tolist())) == 0,
                "train_class_counts": {int(k): int(v) for k, v in
                                        zip(*np.unique(s["y_tr"], return_counts=True))},
                "test_class_counts": {int(k): int(v) for k, v in
                                       zip(*np.unique(s["y_te"], return_counts=True))},
            }
    utils.save_json(os.path.join(out_dir, "split_index_validation.json"), index_report)

    # VQC grids
    _run_vqc_grid("fixed_subset", fixed_cache, os.path.join(out_dir, "fixed_subset", "all_runs_full.jsonl"), resume)
    _run_vqc_grid("varying_subset", {s: varying_cache[s] for s in rconfig.SPLIT_SEEDS},
                  os.path.join(out_dir, "varying_subset", "all_runs_full.jsonl"), resume)

    # classical baselines, both variants
    _run_classical_grid("fixed_subset", fixed_cache,
                         os.path.join(out_dir, "fixed_subset", "classical_all_runs.jsonl"), resume)
    _run_classical_grid("varying_subset", varying_cache,
                         os.path.join(out_dir, "varying_subset", "classical_all_runs.jsonl"), resume)

    # QK-SVM — fixed-subset variant only (concern A's explicit omission fix)
    qk_dir = os.path.join(out_dir, "fixed_subset")
    for split_seed, s in fixed_cache.items():
        matched_baselines.run_qkernel(
            s, resume=resume, out_dir=qk_dir, jsonl_name="qkernel_all_runs.jsonl",
            key_fields=("feature_map", "split_seed"), extra_fields={"split_seed": int(split_seed)},
            label=f"SECTION 11-R2 — REPEATED-SPLIT QK-SVM (fixed-subset, split_seed={split_seed})",
        )

    _build_summaries(out_dir)
    _build_paired_comparisons(out_dir)
    print(f"  Split-robustness (round-2) complete -> {out_dir}")


# ── summaries + paired comparisons ───────────────────────────────────────

METRIC_COLS = ["test_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
               "balanced_accuracy", "mcc"]


def _build_summaries(out_dir):
    for variant in ["fixed_subset", "varying_subset"]:
        vdir = os.path.join(out_dir, variant)
        vqc_jsonl = os.path.join(vdir, "all_runs_full.jsonl")
        cls_jsonl = os.path.join(vdir, "classical_all_runs.jsonl")
        qk_jsonl = os.path.join(vdir, "qkernel_all_runs.jsonl")

        vqc = [r for r in utils.read_jsonl(vqc_jsonl) if r.get("status") == "ok"]
        cls = [r for r in utils.read_jsonl(cls_jsonl) if r.get("status") == "ok"]
        qk = [r for r in utils.read_jsonl(qk_jsonl) if r.get("status") == "ok"] if os.path.exists(qk_jsonl) else []

        rows = []
        for r in vqc:
            row = {"method_type": "VQC", "method_name": f"{r['feature_map']}+{r['ansatz']}",
                   "split_seed": r["split_seed"], "init_seed": r.get("init_seed")}
            row.update({c: r.get(c) for c in METRIC_COLS})
            rows.append(row)
        for r in cls:
            row = {"method_type": "Classical", "method_name": r["model"], "split_seed": r["split_seed"],
                   "init_seed": None}
            row.update({c: r.get(c) for c in METRIC_COLS if c in r})
            rows.append(row)
        for r in qk:
            row = {"method_type": "QK-SVM", "method_name": f"{r['feature_map']} kernel",
                   "split_seed": r["split_seed"], "init_seed": None}
            row.update({c: r.get(c) for c in METRIC_COLS if c in r})
            rows.append(row)

        if not rows:
            print(f"  [{variant}] no results to summarize yet.")
            continue
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(vdir, "all_runs.csv"), index=False)

        # aggregate VQC init-seed repetitions WITHIN each split before any
        # split-level comparison (Section 2.4's explicit requirement)
        vqc_df = df[df["method_type"] == "VQC"]
        if not vqc_df.empty:
            per_split = (vqc_df.groupby(["method_name", "split_seed"])[METRIC_COLS]
                         .mean().reset_index())
            per_split.to_csv(os.path.join(vdir, "vqc_per_split_init_averaged.csv"), index=False)

        present = [c for c in METRIC_COLS if c in df.columns]
        summ = df.groupby(["method_type", "method_name"])[present].agg(["mean", "std", "count"])
        summ.columns = ["_".join(c) for c in summ.columns]
        summ = summ.reset_index()
        summ.to_csv(os.path.join(vdir, "summary.csv"), index=False)
        print(f"  [{variant}] summary saved -> {vdir}")

    # merge fixed-subset + varying-subset summaries into the single
    # top-level filenames the round-2 package spec names
    fixed_runs = os.path.join(out_dir, "fixed_subset", "all_runs.csv")
    if os.path.exists(fixed_runs):
        df = pd.read_csv(fixed_runs)
        df.to_csv(os.path.join(out_dir, "all_runs.csv"), index=False)
        present = [c for c in METRIC_COLS if c in df.columns]
        summ = df.groupby(["method_type", "method_name"])[present].agg(["mean", "std", "count"])
        summ.columns = ["_".join(c) for c in summ.columns]
        summ.reset_index().to_csv(os.path.join(out_dir, "summary.csv"), index=False)


def _build_paired_comparisons(out_dir):
    """Section 2.4 — paired split-level differences for the principal
    scientific comparisons, computed on the fixed-subset (primary) variant
    only, using init-seed-averaged VQC performance per split. n=5 splits:
    reported as paired mean difference + bootstrap 95% CI, explicitly NOT
    framed as a significance claim given the small number of partitions."""
    vdir = os.path.join(out_dir, "fixed_subset")
    per_split_path = os.path.join(vdir, "vqc_per_split_init_averaged.csv")
    all_runs_path = os.path.join(vdir, "all_runs.csv")
    if not (os.path.exists(per_split_path) and os.path.exists(all_runs_path)):
        print("  No fixed-subset results yet — skipping paired comparisons.")
        return

    vqc = pd.read_csv(per_split_path)
    all_df = pd.read_csv(all_runs_path)
    cls = all_df[all_df["method_type"] == "Classical"]

    comparisons = []

    def _paired_row(name_a, vals_a, name_b, vals_b, metric):
        common_splits = sorted(set(vals_a.keys()) & set(vals_b.keys()))
        if len(common_splits) < 2:
            return None
        diffs = np.array([vals_a[s] - vals_b[s] for s in common_splits])
        ci = utils.bootstrap_ci(diffs, np.mean, n_boot=2000, seed=0)
        try:
            w_stat, w_p = sps.wilcoxon(diffs) if len(diffs) >= 2 and np.any(diffs != 0) else (float("nan"), float("nan"))
        except Exception:
            w_stat, w_p = float("nan"), float("nan")
        return {
            "metric": metric, "method_a": name_a, "method_b": name_b, "n_splits": len(common_splits),
            "mean_diff_a_minus_b": float(np.mean(diffs)), "diff_ci_lo": ci["lo"], "diff_ci_hi": ci["hi"],
            "wilcoxon_statistic": float(w_stat) if np.isfinite(w_stat) else None,
            "wilcoxon_p_value": float(w_p) if np.isfinite(w_p) else None,
            "caveat": "n=5 paired partitions — descriptive, not a claim of statistical significance",
        }

    for metric in ["test_accuracy", "f1"]:
        by_method = {m: dict(zip(g["split_seed"], g[metric])) for m, g in vqc.groupby("method_name")}
        method_names = sorted(by_method)
        # all pairwise VQC-vs-VQC comparisons among the 9 configs
        for a, b in itertools.combinations(method_names, 2):
            row = _paired_row(a, by_method[a], b, by_method[b], metric)
            if row:
                comparisons.append(row)
        # each VQC config vs. the best classical baseline on that same split
        if not cls.empty:
            best_per_split = cls.loc[cls.groupby("split_seed")[metric].idxmax()]
            best_vals = dict(zip(best_per_split["split_seed"], best_per_split[metric]))
            best_name_per_split = dict(zip(best_per_split["split_seed"], best_per_split["method_name"]))
            for a in method_names:
                row = _paired_row(a, by_method[a], "best_classical_per_split", best_vals, metric)
                if row:
                    import json as _json
                    row["best_classical_model_by_split"] = _json.dumps(
                        {int(k): v for k, v in best_name_per_split.items()})
                    comparisons.append(row)

    if comparisons:
        pd.DataFrame(comparisons).to_csv(os.path.join(out_dir, "paired_comparisons.csv"), index=False)
        print(f"  Paired comparisons saved -> {os.path.join(out_dir, 'paired_comparisons.csv')} "
              f"({len(comparisons)} rows)")
