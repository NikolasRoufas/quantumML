"""
Sections 16-18 — Dedicated initialization-based barren-plateau / gradient-
scaling experiment. Gradients are evaluated BEFORE any training, at random
parameter initializations, across a (qubit count x depth x ansatz) grid.
This is a separate, purpose-built analysis from the per-epoch training
gradient diagnostics recorded in Section 6.

Implementation assumption (Section 16.C): to keep the data/cost definition
consistent across qubit counts (avoiding confounding from feature-map
dimensionality, e.g. AmplitudeEmbedding padding to 2^n), every cell uses a
simple fixed encoding — one RX(x_i) rotation per qubit — with x sampled once
per qubit-count n (shared across all depths/ansätze at that n) and the cost
is the raw expectation value <PauliZ(0)>, matching the observable used
throughout the rest of the pipeline (before the BCE wrapper).
"""
import os
import math
import time
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
import scipy.stats as sps

from revision import rconfig, utils, circuits as rcircuits


def _build_bp_circuit(n_qubits, ansatz):
    dev = qml.device("default.qubit", wires=n_qubits)
    ansatz_fn = rcircuits.ANSATZ_FNS[ansatz]

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(x, weights):
        for i in range(n_qubits):
            qml.RX(x[i], wires=i)
        ansatz_fn(weights, n_qubits)
        return qml.expval(qml.PauliZ(0))

    return circuit


def run_barren_plateau(resume=False, save=True):
    print("=" * 65)
    print("  SECTION 16-17 — DEDICATED BARREN-PLATEAU GRADIENT-SCALING SWEEP")
    print("=" * 65)
    out_dir = rconfig.DIRS["barren_plateau"]
    raw_path = os.path.join(out_dir, "raw_gradients.csv")
    summary_path = os.path.join(out_dir, "summary.csv")
    jsonl_path = os.path.join(out_dir, "cells.jsonl")

    done_cells = set()
    if resume:
        done_cells = utils.load_completed_keys(jsonl_path, ["n_qubits", "depth", "ansatz"])
        print(f"  --resume: {len(done_cells)} cells already completed.")

    metadata = {
        "encoding": "RX(x_i) per qubit, x fixed per qubit-count n (shared across depths/ansatze at that n)",
        "cost": "expval(PauliZ(0)) — raw expectation value, evaluated at random init, before any training",
        "n_init": rconfig.BP_N_INIT,
        "qubit_grid": rconfig.BP_QUBITS,
        "depth_grid": rconfig.BP_DEPTHS,
        "ansatze": rconfig.BP_ANSATZE,
        "definitions": {
            "mean_grad": "mean of all signed gradient entries pooled across every init and parameter",
            "mean_abs_grad": "mean(|gradient|) pooled across every init and parameter",
            "median_abs_grad": "median(|gradient|) pooled across every init and parameter",
            "grad_l2_norm_mean": "mean, over inits, of the per-init L2 norm of the full gradient vector",
            "grad_l2_norm_std": "std, over inits, of the per-init L2 norm",
            "aggregate_grad_variance": "variance of ALL gradient entries pooled across every init and parameter",
            "representative_param_variance": "variance, ACROSS the n_init random initializations, of ONE fixed "
                "trainable parameter's gradient (parameter index 0) — the classic McClean et al. (2018) "
                "single-parameter barren-plateau diagnostic",
            "representative_param_variance_ci95": "percentile bootstrap 95% CI (resampling the n_init axis) "
                "for representative_param_variance",
            "iqr_abs_grad": "interquartile range of |gradient|, pooled across every init and parameter",
        },
    }
    utils.save_json(os.path.join(out_dir, "metadata.json"), metadata)

    raw_rows = []
    summary_rows = []
    cells = [(n, d, a) for n in rconfig.BP_QUBITS for d in rconfig.BP_DEPTHS for a in rconfig.BP_ANSATZE]
    print(f"  {len(cells)} (n_qubits, depth, ansatz) cells x {rconfig.BP_N_INIT} inits each.")

    x_by_n = {n: np.random.RandomState(n).uniform(-1.0, 1.0, n) for n in rconfig.BP_QUBITS}

    n_done = 0
    t_start = time.time()
    for (n, depth, ansatz) in cells:
        key = (n, depth, ansatz)
        if key in done_cells:
            n_done += 1
            continue
        try:
            shape = rcircuits.weight_shape(ansatz, depth, n)
            n_params = int(np.prod(shape))
            circuit = _build_bp_circuit(n, ansatz)
            x = x_by_n[n]

            grad_fn = qml.grad(circuit, argnums=1)
            grads = np.zeros((rconfig.BP_N_INIT, n_params))
            costs = np.zeros(rconfig.BP_N_INIT)
            rng = np.random.RandomState(hash((n, depth, ansatz)) % (2 ** 31))

            for init_i in range(rconfig.BP_N_INIT):
                w = pnp.array(rng.uniform(-math.pi, math.pi, shape), requires_grad=True)
                xp = pnp.array(x, requires_grad=False)
                cost_val = float(circuit(xp, w))
                g = np.asarray(grad_fn(xp, w)).flatten()
                grads[init_i] = g
                costs[init_i] = cost_val
                raw_rows.append({
                    "n_qubits": n, "depth": depth, "ansatz": ansatz, "init_id": init_i,
                    "cost": cost_val, "grad_l2_norm": float(np.linalg.norm(g)),
                    "grad_mean_abs": float(np.mean(np.abs(g))),
                    "grad_vector_json": np.array2string(g, precision=8, separator=",", threshold=10000),
                })

            flat = grads.flatten()
            l2_per_init = np.linalg.norm(grads, axis=1)
            rep_param = grads[:, 0]
            rep_var_ci = utils.bootstrap_ci(rep_param, np.var, n_boot=2000, seed=0)
            q75, q25 = np.percentile(np.abs(flat), [75, 25])

            summary_rows.append({
                "n_qubits": n, "depth": depth, "ansatz": ansatz,
                "n_init": rconfig.BP_N_INIT, "n_params": n_params,
                "mean_grad": float(np.mean(flat)), "mean_abs_grad": float(np.mean(np.abs(flat))),
                "median_abs_grad": float(np.median(np.abs(flat))),
                "grad_l2_norm_mean": float(np.mean(l2_per_init)), "grad_l2_norm_std": float(np.std(l2_per_init)),
                "aggregate_grad_variance": float(np.var(flat)), "aggregate_grad_std": float(np.std(flat)),
                "representative_param_variance": float(np.var(rep_param)),
                "representative_param_variance_ci_lo": rep_var_ci["lo"],
                "representative_param_variance_ci_hi": rep_var_ci["hi"],
                "iqr_abs_grad": float(q75 - q25),
            })
            utils.append_jsonl(jsonl_path, {"n_qubits": n, "depth": depth, "ansatz": ansatz, "status": "ok"})
            n_done += 1
            if n_done % 10 == 0 or n_done == len(cells):
                print(f"    [{n_done}/{len(cells)}] cells done ({(time.time()-t_start)/60:.1f} min elapsed)",
                      flush=True)
        except Exception as e:
            utils.record_failure(str(e), stage="barren_plateau", n_qubits=n, depth=depth, ansatz=ansatz)
            utils.append_jsonl(jsonl_path, {"n_qubits": n, "depth": depth, "ansatz": ansatz, "status": "failed",
                                             "error": str(e)})
            print(f"    FAILED n={n} depth={depth} ansatz={ansatz}: {e}")

    if save and raw_rows:
        # append to existing files if resuming
        raw_df = pd.DataFrame(raw_rows)
        if resume and os.path.exists(raw_path):
            raw_df.to_csv(raw_path, mode="a", header=False, index=False)
        else:
            raw_df.to_csv(raw_path, index=False)

    if save and summary_rows:
        summ_df = pd.DataFrame(summary_rows)
        if resume and os.path.exists(summary_path):
            old = pd.read_csv(summary_path)
            summ_df = pd.concat([old, summ_df], ignore_index=True)
        summ_df.to_csv(summary_path, index=False)
        print(f"  Saved -> {raw_path}, {summary_path}")

    return summary_rows


def run_scaling_analysis(save=True):
    """Section 18 — fit scaling trends and report evidence classification
    (does NOT auto-declare a barren plateau)."""
    print("=" * 65)
    print("  SECTION 18 — BARREN-PLATEAU SCALING ANALYSIS")
    print("=" * 65)
    out_dir = rconfig.DIRS["barren_plateau"]
    summary_path = os.path.join(out_dir, "summary.csv")
    if not os.path.exists(summary_path):
        print("  No barren-plateau summary found — run Section 16-17 first.")
        return {}
    df = pd.read_csv(summary_path)

    fits = {}
    for ansatz in rconfig.BP_ANSATZE:
        sub = df[df["ansatz"] == ansatz]
        fits[ansatz] = {}

        # vs qubit count, at each depth (average across depths for a headline fit,
        # plus per-depth fits for transparency)
        vs_n_by_depth = {}
        for depth in sorted(sub["depth"].unique()):
            s = sub[sub["depth"] == depth].sort_values("n_qubits")
            if len(s) < 3 or (s["aggregate_grad_variance"] <= 0).any():
                vs_n_by_depth[str(depth)] = {"note": "insufficient or non-positive variance data for log-fit"}
                continue
            x = s["n_qubits"].values.astype(float)
            logy = np.log10(s["aggregate_grad_variance"].values)
            reg = sps.linregress(x, logy)
            vs_n_by_depth[str(depth)] = {
                "slope_log10var_per_qubit": float(reg.slope), "intercept": float(reg.intercept),
                "r_value": float(reg.rvalue), "r_squared": float(reg.rvalue ** 2),
                "p_value": float(reg.pvalue), "stderr": float(reg.stderr),
            }
        fits[ansatz]["log_variance_vs_qubits_by_depth"] = vs_n_by_depth

        # headline: average variance across depths per qubit count, then fit
        agg = sub.groupby("n_qubits")["aggregate_grad_variance"].mean().reset_index().sort_values("n_qubits")
        headline_n = None
        if len(agg) >= 3 and (agg["aggregate_grad_variance"] > 0).all():
            x = agg["n_qubits"].values.astype(float)
            logy = np.log10(agg["aggregate_grad_variance"].values)
            reg = sps.linregress(x, logy)
            headline_n = {
                "slope_log10var_per_qubit": float(reg.slope), "r_squared": float(reg.rvalue ** 2),
                "p_value": float(reg.pvalue),
            }
            evidence = _classify_evidence(reg.slope, reg.pvalue, reg.rvalue ** 2)
            headline_n["evidence_classification"] = evidence
        fits[ansatz]["headline_vs_qubits"] = headline_n

        # vs depth, at each qubit count
        vs_d_by_n = {}
        for n in sorted(sub["n_qubits"].unique()):
            s = sub[sub["n_qubits"] == n].sort_values("depth")
            if len(s) < 3 or (s["aggregate_grad_variance"] <= 0).any():
                vs_d_by_n[str(n)] = {"note": "insufficient or non-positive variance data for log-fit"}
                continue
            x = s["depth"].values.astype(float)
            logy = np.log10(s["aggregate_grad_variance"].values)
            reg = sps.linregress(x, logy)
            vs_d_by_n[str(n)] = {
                "slope_log10var_per_depth": float(reg.slope), "r_squared": float(reg.rvalue ** 2),
                "p_value": float(reg.pvalue),
            }
        fits[ansatz]["log_variance_vs_depth_by_qubits"] = vs_d_by_n

        agg_d = sub.groupby("depth")["aggregate_grad_variance"].mean().reset_index().sort_values("depth")
        headline_d = None
        if len(agg_d) >= 3 and (agg_d["aggregate_grad_variance"] > 0).all():
            x = agg_d["depth"].values.astype(float)
            logy = np.log10(agg_d["aggregate_grad_variance"].values)
            reg = sps.linregress(x, logy)
            headline_d = {
                "slope_log10var_per_depth": float(reg.slope), "r_squared": float(reg.rvalue ** 2),
                "p_value": float(reg.pvalue),
            }
            headline_d["evidence_classification"] = _classify_evidence(reg.slope, reg.pvalue, reg.rvalue ** 2)
        fits[ansatz]["headline_vs_depth"] = headline_d

        print(f"    {ansatz}: vs-qubits {headline_n and headline_n.get('evidence_classification')}; "
              f"vs-depth {headline_d and headline_d.get('evidence_classification')}")

    if save:
        utils.save_json(os.path.join(out_dir, "scaling_fits.json"), fits)
        print(f"  Saved -> {os.path.join(out_dir, 'scaling_fits.json')}")
    return fits


def _classify_evidence(slope, p_value, r_squared):
    """Never auto-declares a barren plateau — classifies the strength of
    evidence for gradient suppression / exponential-type scaling."""
    if slope < 0 and p_value < 0.05 and r_squared > 0.5:
        return "evidence compatible with barren-plateau-like exponential gradient suppression"
    if slope < 0 and p_value < 0.10:
        return "some evidence of gradient suppression, but not conclusive for asymptotic barren-plateau scaling"
    return "insufficient evidence for barren-plateau-like scaling in the tested range"
