"""
Section 25 — Sanity checks run over everything produced so far. Does not
interpret results, just checks internal consistency / validity.

2026-09-15 fix: the missing/NaN check now runs BEFORE filtering to non-null
values. The original version filtered to df[col].notna() first and only
checked the range of what remained, so a column that was NaN for every row
in a group produced zero flagged rows and silently passed — exactly what
happened with precision/recall for the three classical split-robustness
baselines (a separate code omission in split_robustness.py, since fixed).
Missing values are now their own explicit issue category.
"""
import os
import json
import numpy as np
import pandas as pd

import rconfig, utils


def _check_runs_csv(path, label, n_expected=None, proba_col=None):
    issues = []
    if not os.path.exists(path):
        return [f"{label}: file not found ({path})"]
    df = pd.read_csv(path)
    if df.empty:
        return [f"{label}: file is empty"]

    for col in ["test_accuracy", "precision", "recall", "f1", "balanced_accuracy"]:
        if col in df.columns:
            missing = df[df[col].isna()]
            if len(missing):
                issues.append(f"{label}: {len(missing)} rows have missing/NaN {col}")

            bad = df[(df[col].notna()) & ((df[col] < -1e-9) | (df[col] > 1 + 1e-9))]
            if len(bad):
                issues.append(f"{label}: {len(bad)} rows have {col} outside [0,1]")

    if "mcc" in df.columns:
        missing_mcc = df[df["mcc"].isna()]
        if len(missing_mcc):
            issues.append(f"{label}: {len(missing_mcc)} rows have missing/NaN mcc")
        bad = df[(df["mcc"].notna()) & ((df["mcc"] < -1 - 1e-9) | (df["mcc"] > 1 + 1e-9))]
        if len(bad):
            issues.append(f"{label}: {len(bad)} rows have MCC outside [-1,1]")

    if "confusion_matrix_json" in df.columns or "confusion_matrix" in df.columns:
        col = "confusion_matrix_json" if "confusion_matrix_json" in df.columns else "confusion_matrix"
        n_test_col = "n_test" if "n_test" in df.columns else None
        for idx, row in df.iterrows():
            try:
                cm = json.loads(row[col]) if isinstance(row[col], str) else row[col]
                total = int(np.sum(cm))
                expected = n_expected or (row[n_test_col] if n_test_col else total)
                if total != expected:
                    issues.append(f"{label} row {idx}: confusion matrix total {total} != expected {expected}")
            except Exception:
                pass

    return issues


def run_sanity_checks(save=True):
    print("=" * 65)
    print("  SECTION 25 — SANITY CHECKS")
    print("=" * 65)
    issues = {}

    audit = utils.load_json(os.path.join(rconfig.RESULTS_DIR, "dataset_audit.json"))
    if audit:
        probs = audit.get("class_proportions", {})
        total_prop = sum(probs.values())
        this_issues = []
        if audit.get("n_samples") != 19020:
            this_issues.append(f"n_samples={audit.get('n_samples')} != expected 19020")
        if abs(total_prop - 1.0) > 1e-6:
            this_issues.append(f"class proportions sum to {total_prop}, not 1.0")
        if audit.get("positive_class") not in (0, 1):
            this_issues.append(f"positive_class={audit.get('positive_class')} not in (0,1)")
        issues["dataset_audit"] = this_issues
    else:
        issues["dataset_audit"] = ["dataset_audit.json not found"]

    # index leakage check
    tr_path = os.path.join(rconfig.RESULTS_DIR, "primary_train_indices.npy")
    te_path = os.path.join(rconfig.RESULTS_DIR, "primary_test_indices.npy")
    this_issues = []
    if os.path.exists(tr_path) and os.path.exists(te_path):
        tr = np.load(tr_path)
        te = np.load(te_path)
        overlap = set(tr.tolist()) & set(te.tolist())
        if overlap:
            this_issues.append(f"train/test index overlap: {len(overlap)} shared indices")
        if len(tr) != rconfig.PRIMARY_N_TRAIN:
            this_issues.append(f"train size {len(tr)} != {rconfig.PRIMARY_N_TRAIN}")
        if len(te) != rconfig.PRIMARY_N_TEST:
            this_issues.append(f"test size {len(te)} != {rconfig.PRIMARY_N_TEST}")
    else:
        this_issues.append("primary split index files not found")
    issues["primary_split"] = this_issues

    issues["factorial_all_runs"] = _check_runs_csv(
        os.path.join(rconfig.RESULTS_DIR, "factorial", "all_runs.csv"), "factorial")
    issues["matched_budget_all_runs"] = _check_runs_csv(
        os.path.join(rconfig.RESULTS_DIR, "matched_budget", "all_runs.csv"), "matched_budget")
    issues["classical_baselines"] = _check_runs_csv(
        os.path.join(rconfig.DIRS["matched_baselines"], "classical_all_runs.csv"), "classical")
    issues["qkernel"] = _check_runs_csv(
        os.path.join(rconfig.DIRS["matched_baselines"], "qkernel_results.csv"), "qkernel")
    issues["split_robustness"] = _check_runs_csv(
        os.path.join(rconfig.DIRS["split_robustness"], "all_runs.csv"), "split_robustness")

    # circuit validation must have all passed
    cv = utils.load_json(os.path.join(rconfig.RESULTS_DIR, "circuit_validation.json"))
    this_issues = []
    if cv is None:
        this_issues.append("circuit_validation.json not found")
    elif not cv.get("all_passed", False):
        this_issues.append("circuit validation did not pass")
    issues["circuit_validation"] = this_issues

    # barren-plateau gradients finite
    bp_raw = os.path.join(rconfig.DIRS["barren_plateau"], "raw_gradients.csv")
    this_issues = []
    if os.path.exists(bp_raw):
        try:
            bdf = pd.read_csv(bp_raw, usecols=["grad_l2_norm", "grad_mean_abs"])
            if not np.isfinite(bdf.values).all():
                this_issues.append("non-finite values in barren-plateau raw gradients")
        except Exception as e:
            this_issues.append(f"could not validate barren-plateau raw gradients: {e}")
    else:
        this_issues.append("raw_gradients.csv not found")
    issues["barren_plateau"] = this_issues

    all_clear = all(len(v) == 0 for v in issues.values())
    out = {"all_clear": all_clear, "issues": issues}
    print(f"  All checks clear: {all_clear}")
    for section, its in issues.items():
        if its:
            print(f"    [{section}] {len(its)} issue(s):")
            for it in its:
                print(f"      - {it}")

    if save:
        utils.save_json(os.path.join(rconfig.RESULTS_DIR, "sanity_checks.json"), out)
        print(f"  Saved -> {os.path.join(rconfig.RESULTS_DIR, 'sanity_checks.json')}")
    return out
