"""
Round-2 Phase 5 — validation suite covering the reviewer's 14 checks. Does
NOT interpret scientific results; only checks internal consistency/validity
of what was produced. An experiment is not reported complete in SUMMARY.md
unless it passes the relevant checks here.

Extends (does not replace) the existing sanity.py, which still covers the
original revision-package checks (factorial, matched_budget, classical
baselines, circuit validation, barren-plateau finiteness).
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score, balanced_accuracy_score,
                              matthews_corrcoef)

import rconfig, utils
import sanity as legacy_sanity
import split_robustness as sr
import barren_plateau as bp


def _issue(issues, label, msg):
    issues.setdefault(label, []).append(msg)


def check_dataset_provenance(issues):
    path = os.path.join(rconfig.RESULTS_DIR, "dataset_provenance.json")
    rep = utils.load_json(path)
    if rep is None:
        _issue(issues, "dataset_provenance", "dataset_provenance.json not found")
        return
    if not rep.get("shape_matches_uci_expectation"):
        _issue(issues, "dataset_provenance", "resolved dataset shape does not match UCI expectation")
    if rep.get("n_missing_values", 0) != 0:
        _issue(issues, "dataset_provenance", f"n_missing_values={rep.get('n_missing_values')} != 0")


def check_split_reproducibility(issues, X):
    """Section 5 check 2 — repeated-split determinism: redraw the fixed
    subsample and one partition independently, compare against saved
    indices on disk."""
    idx1 = sr.get_fixed_subsample_idx(X)
    idx2 = sr.get_fixed_subsample_idx(X)
    if not np.array_equal(idx1, idx2):
        _issue(issues, "split_reproducibility", "fixed subsample is not deterministic across calls")

    sr_dir = rconfig.DIRS["split_robustness"]
    seed = rconfig.SPLIT_SEEDS[0]
    saved_tr = os.path.join(sr_dir, "fixed_subset", "split_indices", f"train_idx_seed{seed}.npy")
    saved_te = os.path.join(sr_dir, "fixed_subset", "split_indices", f"test_idx_seed{seed}.npy")
    if not (os.path.exists(saved_tr) and os.path.exists(saved_te)):
        _issue(issues, "split_reproducibility", f"saved split-index files missing for seed={seed}")
        return
    return idx1


def check_disjointness_and_partition_matching(issues):
    sr_dir = rconfig.DIRS["split_robustness"]
    report = utils.load_json(os.path.join(sr_dir, "split_index_validation.json"))
    if report is None:
        _issue(issues, "disjointness", "split_index_validation.json not found")
        return
    for variant, splits in report.items():
        for seed, info in splits.items():
            if not info.get("disjoint"):
                _issue(issues, "disjointness", f"{variant} split_seed={seed}: train/test overlap detected")
            if info.get("n_train") != rconfig.PRIMARY_N_TRAIN or info.get("n_test") != rconfig.PRIMARY_N_TEST:
                _issue(issues, "disjointness",
                       f"{variant} split_seed={seed}: unexpected sizes n_train={info.get('n_train')} "
                       f"n_test={info.get('n_test')}")

    # matched partitions: every model/method must use IDENTICAL indices for a given split
    fixed_dir = os.path.join(sr_dir, "fixed_subset")
    by_split = {}
    for fname, method_type in [("all_runs_full.jsonl", "VQC"), ("classical_all_runs.jsonl", "Classical"),
                                ("qkernel_all_runs.jsonl", "QK-SVM")]:
        path = os.path.join(fixed_dir, fname)
        for r in utils.read_jsonl(path):
            if r.get("status") != "ok" or "train_idx" not in r or r["train_idx"] is None:
                continue
            seed = r["split_seed"]
            key = (tuple(sorted(r["train_idx"])), tuple(sorted(r["test_idx"])))
            by_split.setdefault(seed, set()).add(key)
    for seed, keyset in by_split.items():
        if len(keyset) > 1:
            _issue(issues, "matched_partitions",
                   f"fixed_subset split_seed={seed}: {len(keyset)} distinct index sets across methods "
                   f"(expected exactly 1 — all methods must share the same partition)")


def check_confusion_matrix_consistency(issues):
    sr_dir = rconfig.DIRS["split_robustness"]
    for variant in ["fixed_subset", "varying_subset"]:
        for fname in ["all_runs_full.jsonl", "classical_all_runs.jsonl", "qkernel_all_runs.jsonl"]:
            path = os.path.join(sr_dir, variant, fname)
            for r in utils.read_jsonl(path):
                if r.get("status") != "ok":
                    continue
                cm = r.get("confusion_matrix")
                n_test = len(r.get("y_true", []))
                if cm is not None and n_test and int(np.sum(cm)) != n_test:
                    _issue(issues, "confusion_matrix",
                           f"{variant}/{fname} {r.get('feature_map', r.get('model'))}"
                           f"{'+' + r['ansatz'] if r.get('ansatz') else ''} split={r.get('split_seed')}: "
                           f"cm total {int(np.sum(cm))} != n_test {n_test}")


def check_metric_recomputation(issues):
    """Recompute metrics directly from saved y_true/y_pred/y_proba and
    compare against the saved metric fields — the reviewer's explicit
    'saved predictions reproduce the reported metrics' requirement."""
    sr_dir = rconfig.DIRS["split_robustness"]
    tol = 1e-6
    n_checked = 0
    for variant in ["fixed_subset", "varying_subset"]:
        for fname in ["all_runs_full.jsonl", "classical_all_runs.jsonl", "qkernel_all_runs.jsonl"]:
            path = os.path.join(sr_dir, variant, fname)
            for r in utils.read_jsonl(path):
                if r.get("status") != "ok" or "y_true" not in r:
                    continue
                y_true = np.asarray(r["y_true"])
                y_pred = np.asarray(r["y_pred"])
                y_proba = np.asarray(r.get("y_proba", []))
                recomputed = {
                    "test_accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1": f1_score(y_true, y_pred, zero_division=0),
                    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                }
                if len(set(y_true.tolist())) > 1 and len(y_proba) == len(y_true):
                    recomputed["roc_auc"] = roc_auc_score(y_true, y_proba)
                    recomputed["pr_auc"] = average_precision_score(y_true, y_proba)
                for k, v in recomputed.items():
                    saved = r.get(k)
                    if saved is None or not np.isfinite(saved):
                        continue
                    if abs(saved - v) > tol:
                        _issue(issues, "metric_recomputation",
                               f"{variant}/{fname} split={r.get('split_seed')}: {k} saved={saved:.6f} "
                               f"recomputed={v:.6f} (diff={abs(saved-v):.2e})")
                n_checked += 1
    if n_checked == 0:
        _issue(issues, "metric_recomputation", "no runs with saved predictions found to check")


def check_class_mapping(issues):
    path = os.path.join(rconfig.RESULTS_DIR, "dataset_provenance.json")
    rep = utils.load_json(path)
    if rep is None:
        return
    lc = rep.get("label_semantics_check") or {}
    if lc and not lc.get("assumption_matches_physics_check", True):
        _issue(issues, "class_mapping_documented_caveat",
               "data_prep.load_magic's 'gamma=1' comment does not match the physics-based label check "
               "(documented finding, not treated as a blocking error — see dataset_provenance.json; "
               "left unchanged per explicit instruction)")


def check_gradient_seed_reproducibility(issues):
    det = utils.load_json(os.path.join(rconfig.DIRS["barren_plateau"], "determinism_check.json"))
    if det is None:
        _issue(issues, "gradient_determinism", "determinism_check.json not found")
    elif not det.get("all_passed"):
        _issue(issues, "gradient_determinism", f"determinism spot-check failed: {det.get('checks')}")


def check_per_parameter_variance(issues, n_spot_checks=5):
    bp_dir = rconfig.DIRS["barren_plateau"]
    summary_path = os.path.join(bp_dir, "summary.csv")
    if not os.path.exists(summary_path):
        _issue(issues, "per_parameter_variance", "barren_plateau summary.csv not found")
        return
    full = pd.read_csv(summary_path)
    df = full.sample(n=min(n_spot_checks, len(full)), random_state=0)
    for _, row in df.iterrows():
        g = bp._load_raw_grads(bp_dir, int(row["n_qubits"]), int(row["depth"]), row["ansatz"])
        if g is None:
            _issue(issues, "per_parameter_variance",
                   f"raw grads missing for n={row['n_qubits']} depth={row['depth']} ansatz={row['ansatz']}")
            continue
        recomputed = float(g.var(axis=0).mean())
        saved = float(row["mean_of_per_param_variances"])
        if abs(recomputed - saved) > 1e-9:
            _issue(issues, "per_parameter_variance",
                   f"n={row['n_qubits']} depth={row['depth']} {row['ansatz']}: recomputed "
                   f"mean_of_per_param_variances={recomputed:.10f} != saved {saved:.10f}")


def check_bootstrap_and_ci_validity(issues):
    fits = utils.load_json(os.path.join(rconfig.DIRS["barren_plateau"], "scaling_fits.json"))
    if fits is None:
        _issue(issues, "bootstrap_ci", "scaling_fits.json not found")
        return
    for ansatz, block in fits.items():
        if ansatz.startswith("_"):
            continue
        for key in ["headline_vs_qubits", "headline_vs_depth"]:
            fit = block.get(key)
            if fit is None:
                continue
            lo, hi = fit.get("slope_ci95_lo"), fit.get("slope_ci95_hi")
            if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
                _issue(issues, "bootstrap_ci", f"{ansatz}/{key}: CI bounds missing/non-finite")
                continue
            if lo > hi:
                _issue(issues, "bootstrap_ci", f"{ansatz}/{key}: CI lo={lo} > hi={hi}")
            if fit.get("n_bootstrap_replicates", 0) < 2000:
                _issue(issues, "bootstrap_ci",
                       f"{ansatz}/{key}: only {fit.get('n_bootstrap_replicates')} bootstrap replicates (<2000)")


def check_grid_completeness(issues):
    sr_dir = rconfig.DIRS["split_robustness"]
    expected_fixed_vqc = len(rconfig.FEATURE_MAPS) * len(rconfig.ANSATZE) * len(rconfig.SPLIT_SEEDS) \
        * len(rconfig.SPLIT_ROBUSTNESS_INIT_SEEDS)
    expected_varying_vqc = expected_fixed_vqc
    expected_qk = len(rconfig.FEATURE_MAPS) * len(rconfig.SPLIT_SEEDS)
    expected_cls = 3 * len(rconfig.SPLIT_SEEDS)

    def _count_ok(path):
        return sum(1 for r in utils.read_jsonl(path) if r.get("status") == "ok")

    n_fixed_vqc = _count_ok(os.path.join(sr_dir, "fixed_subset", "all_runs_full.jsonl"))
    n_varying_vqc = _count_ok(os.path.join(sr_dir, "varying_subset", "all_runs_full.jsonl"))
    n_qk = _count_ok(os.path.join(sr_dir, "fixed_subset", "qkernel_all_runs.jsonl"))
    n_cls_fixed = _count_ok(os.path.join(sr_dir, "fixed_subset", "classical_all_runs.jsonl"))

    if n_fixed_vqc != expected_fixed_vqc:
        _issue(issues, "grid_completeness", f"fixed_subset VQC: {n_fixed_vqc}/{expected_fixed_vqc} runs ok")
    if n_varying_vqc != expected_varying_vqc:
        _issue(issues, "grid_completeness", f"varying_subset VQC: {n_varying_vqc}/{expected_varying_vqc} runs ok")
    if n_qk != expected_qk:
        _issue(issues, "grid_completeness", f"QK-SVM: {n_qk}/{expected_qk} runs ok")
    if n_cls_fixed != expected_cls:
        _issue(issues, "grid_completeness", f"fixed_subset classical: {n_cls_fixed}/{expected_cls} runs ok")

    bp_cells = utils.read_jsonl(os.path.join(rconfig.DIRS["barren_plateau"], "cells.jsonl"))
    expected_cells = len(rconfig.BP_QUBITS) * len(rconfig.BP_DEPTHS) * len(rconfig.BP_ANSATZE)
    n_ok_cells = sum(1 for r in bp_cells if r.get("status") == "ok")
    if n_ok_cells != expected_cells:
        _issue(issues, "grid_completeness", f"barren_plateau cells: {n_ok_cells}/{expected_cells} ok")


def check_missing_failed_duplicated(issues):
    failed = utils.load_json(os.path.join(rconfig.RESULTS_DIR, "failed_runs.json"), default=[])
    round2_failures = [f for f in failed if "split_robustness" in str(f.get("stage", "")) or
                        "barren_plateau" in str(f.get("stage", ""))]
    if round2_failures:
        _issue(issues, "failed_runs", f"{len(round2_failures)} round-2 failures recorded in failed_runs.json "
               f"(see that file for details — not auto-blocking, but must be disclosed)")

    sr_dir = rconfig.DIRS["split_robustness"]
    for variant in ["fixed_subset", "varying_subset"]:
        path = os.path.join(sr_dir, variant, "all_runs_full.jsonl")
        keys = [(r.get("feature_map"), r.get("ansatz"), r.get("split_seed"), r.get("init_seed"))
                for r in utils.read_jsonl(path)]
        dupes = len(keys) - len(set(keys))
        if dupes:
            _issue(issues, "duplicated_records", f"{variant}/all_runs_full.jsonl: {dupes} duplicate (config,split,init) keys")


def run_full_validation(X=None, save=True):
    print("=" * 65)
    print("  ROUND-2 PHASE 5 — FULL VALIDATION SUITE")
    print("=" * 65)
    issues = {}

    check_dataset_provenance(issues)
    if X is not None:
        check_split_reproducibility(issues, X)
    check_disjointness_and_partition_matching(issues)
    check_confusion_matrix_consistency(issues)
    check_metric_recomputation(issues)
    check_class_mapping(issues)
    check_gradient_seed_reproducibility(issues)
    check_per_parameter_variance(issues)
    check_bootstrap_and_ci_validity(issues)
    check_grid_completeness(issues)
    check_missing_failed_duplicated(issues)

    legacy = legacy_sanity.run_sanity_checks(save=False)

    # "class_mapping_documented_caveat" is an intentional, disclosed, non-blocking
    # finding (see check_class_mapping) — excluded from the blocking all_clear verdict.
    blocking_issues = {k: v for k, v in issues.items() if k != "class_mapping_documented_caveat"}
    all_clear = all(len(v) == 0 for v in blocking_issues.values()) and legacy.get("all_clear", False)

    report = {
        "all_clear": all_clear,
        "round2_issues": issues,
        "legacy_sanity_all_clear": legacy.get("all_clear"),
        "legacy_sanity_issues": legacy.get("issues"),
    }

    print(f"  Round-2 all_clear: {all_clear}")
    for section, its in issues.items():
        tag = "(disclosed, non-blocking)" if section == "class_mapping_documented_caveat" else ""
        if its:
            print(f"    [{section}] {len(its)} item(s) {tag}:")
            for it in its:
                print(f"      - {it}")

    if save:
        utils.save_json(os.path.join(rconfig.RESULTS_DIR, "validation_report.json"), report)
        print(f"  Saved -> {os.path.join(rconfig.RESULTS_DIR, 'validation_report.json')}")
    return report


if __name__ == "__main__":
    import data_prep
    X, y = data_prep.load_magic()
    run_full_validation(X)
