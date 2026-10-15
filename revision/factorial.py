"""
Sections 5-8 — Primary 3x3 factorial VQC experiment: 45 runs (9 configs x 5
seeds), aggregation, and two-way ANOVA statistical analysis.
"""
import os
import time
import itertools
import numpy as np
import pandas as pd

from revision import rconfig, utils, train_core


# ── module-level worker (must be picklable for multiprocessing) ────────────

def _worker(args):
    (feature_map, ansatz, seed, n_qubits, n_layers,
     X_tr, y_tr, X_te, y_te, lr, batch_size, n_epochs) = args
    import torch
    torch.set_num_threads(1)  # avoid BLAS oversubscription across parallel workers
    t0 = time.time()
    try:
        result = train_core.train_one_run(
            feature_map, ansatz, n_qubits, n_layers, seed,
            X_tr, y_tr, X_te, y_te, lr, batch_size, n_epochs, verbose=False,
        )
        result["status"] = "ok"
        return result
    except Exception as e:
        import traceback
        return {
            "feature_map": feature_map, "ansatz": ansatz, "seed": seed,
            "status": "failed", "error": str(e), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - t0,
        }


def run_factorial(split, resume=False, max_workers=None, seeds=None,
                   out_subdir="factorial", n_qubits=None, n_layers=None,
                   configs=None, tag="primary"):
    """Section 5: run every (feature_map, ansatz, seed) VQC configuration on
    the fixed primary partition. Parallelized across independent runs (same
    math, no protocol change — Section 24)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out_dir = os.path.join(rconfig.RESULTS_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "all_runs_full.jsonl")

    seeds = seeds or rconfig.PRIMARY_SEEDS
    configs = configs or list(itertools.product(rconfig.FEATURE_MAPS, rconfig.ANSATZE))
    n_qubits = n_qubits or rconfig.N_QUBITS
    n_layers = n_layers if n_layers is not None else rconfig.N_LAYERS

    all_jobs = [(fm, an, seed) for (fm, an) in configs for seed in seeds]

    done_keys = set()
    if resume:
        done_keys = utils.load_completed_keys(jsonl_path, ["feature_map", "ansatz", "seed"])
        print(f"  --resume: {len(done_keys)} runs already completed, skipping them.")

    pending = [(fm, an, s) for (fm, an, s) in all_jobs if (fm, an, s) not in done_keys]
    print(f"  {tag}: {len(all_jobs)} total configurations, {len(pending)} to run.")

    max_workers = max_workers or utils.default_max_workers(len(pending) or 1)

    X_tr, y_tr = split["X_tr_q"], split["y_tr"]
    X_te, y_te = split["X_te_q"], split["y_te"]

    job_args = [
        (fm, an, seed, n_qubits, n_layers, X_tr, y_tr, X_te, y_te,
         rconfig.LR, rconfig.BATCH_SIZE, rconfig.N_EPOCHS)
        for (fm, an, seed) in pending
    ]

    n_completed = 0
    if job_args:
        print(f"  Launching {len(job_args)} runs across {max_workers} worker processes...")
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_worker, a): (a[0], a[1], a[2]) for a in job_args}
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {"feature_map": key[0], "ansatz": key[1], "seed": key[2],
                              "status": "failed", "error": str(e)}
                utils.append_jsonl(jsonl_path, result)  # persisted immediately (Section 24)
                n_completed += 1
                if result.get("status") == "ok":
                    print(f"    [{n_completed}/{len(job_args)}] OK "
                          f"{result['feature_map']}+{result['ansatz']} seed={result['seed']} "
                          f"test_acc={result['test_accuracy']:.4f} "
                          f"({result['elapsed_seconds']:.0f}s)", flush=True)
                else:
                    print(f"    [{n_completed}/{len(job_args)}] FAILED "
                          f"{result['feature_map']}+{result['ansatz']} seed={result['seed']}: "
                          f"{result.get('error')}", flush=True)
                    utils.record_failure(result.get("error", "unknown"), stage=f"factorial:{tag}", **key_dict(key))

    build_aggregates(out_dir, jsonl_path, tag=tag)
    return jsonl_path


def key_dict(key):
    return {"feature_map": key[0], "ansatz": key[1], "seed": key[2]}


# ── Section 7 — aggregation ────────────────────────────────────────────────

SCALAR_COLS = [
    "feature_map", "ansatz", "seed", "n_qubits", "n_layers",
    "trainable_parameters", "gate_count", "circuit_depth",
    "training_accuracy", "test_accuracy", "precision", "recall", "f1",
    "roc_auc", "pr_auc", "balanced_accuracy", "mcc",
    "final_train_loss", "minimum_train_loss", "elapsed_seconds",
]

METRIC_COLS = ["test_accuracy", "precision", "recall", "f1", "roc_auc",
               "pr_auc", "balanced_accuracy", "mcc", "final_train_loss"]


def load_runs_df(jsonl_path):
    import json
    records = utils.read_jsonl(jsonl_path)
    ok = [r for r in records if r.get("status") == "ok"]
    if not ok:
        return pd.DataFrame(columns=SCALAR_COLS)
    df = pd.DataFrame(ok)
    for c in SCALAR_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df["confusion_matrix_json"] = df["confusion_matrix"].apply(json.dumps) if "confusion_matrix" in df else "[]"
    df["loss_history_json"] = df["loss_history"].apply(json.dumps) if "loss_history" in df else "[]"
    df["test_probabilities_json"] = df["test_probabilities"].apply(json.dumps) if "test_probabilities" in df else "[]"
    return df


def build_aggregates(out_dir, jsonl_path, tag="primary"):
    df = load_runs_df(jsonl_path)
    if df.empty:
        print("  No successful runs to aggregate yet.")
        return

    cols = SCALAR_COLS + ["confusion_matrix_json", "loss_history_json", "test_probabilities_json"]
    df[cols].to_csv(os.path.join(out_dir, "all_runs.csv"), index=False)

    summary = (df.groupby(["feature_map", "ansatz"])[METRIC_COLS]
               .agg(["mean", "std", "count"]))
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    fmap_summary = (df.groupby("feature_map")[METRIC_COLS].agg(["mean", "std", "count"]))
    fmap_summary.columns = ["_".join(c) for c in fmap_summary.columns]
    fmap_summary = fmap_summary.reset_index()
    fmap_summary.to_csv(os.path.join(out_dir, "feature_map_summary.csv"), index=False)

    ansatz_summary = (df.groupby("ansatz")[METRIC_COLS].agg(["mean", "std", "count"]))
    ansatz_summary.columns = ["_".join(c) for c in ansatz_summary.columns]
    ansatz_summary = ansatz_summary.reset_index()
    ansatz_summary.to_csv(os.path.join(out_dir, "ansatz_summary.csv"), index=False)

    out = {
        "tag": tag,
        "n_runs": int(len(df)),
        "n_configs": int(df.groupby(["feature_map", "ansatz"]).ngroups),
        "per_config_summary": summary.to_dict(orient="records"),
        "feature_map_marginal": fmap_summary.to_dict(orient="records"),
        "ansatz_marginal": ansatz_summary.to_dict(orient="records"),
    }
    utils.save_json(os.path.join(out_dir, "factorial_results.json"), out)
    print(f"  Aggregates written to {out_dir}")


# ── Section 8 — statistical analysis ────────────────────────────────────────

def run_statistical_analysis(out_dir=None):
    import scipy.stats as sps
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        HAVE_STATSMODELS = True
    except ImportError:
        HAVE_STATSMODELS = False

    out_dir = out_dir or os.path.join(rconfig.RESULTS_DIR, "factorial")
    df = load_runs_df(os.path.join(out_dir, "all_runs_full.jsonl"))
    if df.empty or df.groupby(["feature_map", "ansatz"]).ngroups < 2:
        print("  Not enough data for statistical analysis yet.")
        return {}

    results = {
        "n_seeds_per_cell": int(df.groupby(["feature_map", "ansatz"]).size().min()),
        "limitation_note": (
            "Only 5 repeated runs per (feature_map x ansatz) cell. Two-way ANOVA "
            "F-tests and post-hoc comparisons below are exploratory / low-powered; "
            "p-values and effect sizes should be interpreted as descriptive evidence, "
            "not confirmatory inference."
        ),
        "metrics": {},
    }

    for metric in ["test_accuracy", "f1"]:
        m = {}
        cell_groups = [g[metric].values for _, g in df.groupby(["feature_map", "ansatz"])]

        # assumption checks
        shapiro_p = []
        for arr in cell_groups:
            if len(arr) >= 3 and np.std(arr) > 1e-12:
                shapiro_p.append(float(sps.shapiro(arr).pvalue))
        try:
            levene_stat, levene_p = sps.levene(*cell_groups)
        except Exception:
            levene_stat, levene_p = float("nan"), float("nan")
        m["assumptions"] = {
            "shapiro_p_per_cell": shapiro_p,
            "shapiro_min_p": float(np.min(shapiro_p)) if shapiro_p else None,
            "levene_stat": float(levene_stat), "levene_p": float(levene_p),
            "note": "With n=5/cell, normality tests are low-powered; treat as diagnostic only.",
        }

        if HAVE_STATSMODELS:
            model = smf.ols(f"{metric} ~ C(feature_map) * C(ansatz)", data=df).fit()
            anova_tbl = sm.stats.anova_lm(model, typ=2)
            ss_resid = anova_tbl.loc["Residual", "sum_sq"]
            anova_json = {}
            for term in anova_tbl.index:
                if term == "Residual":
                    continue
                ss = anova_tbl.loc[term, "sum_sq"]
                partial_eta_sq = float(ss / (ss + ss_resid))
                anova_json[term] = {
                    "df": float(anova_tbl.loc[term, "df"]),
                    "F": float(anova_tbl.loc[term, "F"]),
                    "p": float(anova_tbl.loc[term, "PR(>F)"]),
                    "partial_eta_squared": partial_eta_sq,
                }
            anova_json["Residual"] = {
                "df": float(anova_tbl.loc["Residual", "df"]),
                "sum_sq": float(ss_resid),
            }
            m["two_way_anova"] = anova_json

            # post-hoc Tukey HSD over the 9 cells (multiplicity-corrected)
            df["_cell"] = df["feature_map"] + " + " + df["ansatz"]
            tukey = pairwise_tukeyhsd(df[metric].values, df["_cell"].values, alpha=0.05)
            tukey_rows = []
            for row in tukey.summary().data[1:]:
                tukey_rows.append({
                    "group1": row[0], "group2": row[1], "meandiff": float(row[2]),
                    "p_adj": float(row[3]), "lower": float(row[4]), "upper": float(row[5]),
                    "reject": bool(row[6]),
                })
            m["posthoc_tukey_hsd"] = tukey_rows
        else:
            m["two_way_anova"] = None
            m["posthoc_tukey_hsd"] = None
            m["note"] = "statsmodels not available — ANOVA/Tukey skipped."

        results["metrics"][metric] = m

    utils.save_json(os.path.join(out_dir, "statistical_tests.json"), results)
    print(f"  Statistical tests saved -> {os.path.join(out_dir, 'statistical_tests.json')}")
    return results
