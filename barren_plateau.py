"""
Sections 16-18 (round-2 reviewer response) — Dedicated initialization-based
barren-plateau / gradient-scaling experiment, corrected per reviewer
concern C.

Preserves from the original design (Section 16.C): the (qubit x depth x
ansatz) grid, BP_N_INIT=100, the fixed RX(x_i)-per-qubit encoding (x shared
across depths/ansatze at a given qubit count n), and the raw <PauliZ(0)>
cost before any training.

Corrected:
  - Primary diagnostic is now the PER-PARAMETER initialization variance
    V_p = Var_i(g[i, p]) for every trainable parameter p, not a single
    statistic pooling every (init, parameter) pair together. The old pooled
    "aggregate_grad_variance" is retained only as an explicitly-labeled
    secondary comparison.
  - Seeding is derived via hashlib.sha256 of the cell's own identity, not
    Python's built-in hash() (salted per-process by default — not
    reproducible across runs for a seed built from a string).
  - Full per-init gradient matrices are saved as .npy (not just a CSV of
    string-encoded vectors) so the scaling-regression step can bootstrap
    over the actual initialization axis rather than refitting a point
    estimate only.
  - Scaling-regression slopes get a 2,000-replicate initialization-level
    bootstrap 95% CI (resampling complete per-init gradient vectors within
    each cell, recomputing the per-parameter variance statistic, then
    refitting the regression — never resampling individual gradient entries
    as if they were independent observations).
  - Sensitivity analyses: parameter-selection (representative / mean / median
    / pooled all compared), and leave-one-qubit-out refits.
"""
import os
import math
import time
import hashlib
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
import scipy.stats as sps

import rconfig, utils, circuits as rcircuits


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


def _deterministic_cell_seed(n_qubits, depth, ansatz):
    """Deterministic across processes and Python versions — unlike
    hash((n, depth, ansatz)), which salts string hashing per-process by
    default (PYTHONHASHSEED) and is therefore NOT reproducible run-to-run
    for a tuple containing a string."""
    key = f"{n_qubits}|{depth}|{ansatz}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big") % (2 ** 31)


def _raw_grads_path(out_dir, n, depth, ansatz):
    d = os.path.join(out_dir, "raw_grads")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"n{n}_d{depth}_{ansatz}.npy")


def _compute_cell(n, depth, ansatz, x, n_init=None):
    """Runs the n_init gradient evaluations for one grid cell. Returns
    (grads (n_init x n_params), costs (n_init,), seed_used)."""
    n_init = n_init or rconfig.BP_N_INIT
    shape = rcircuits.weight_shape(ansatz, depth, n)
    n_params = int(np.prod(shape))
    circuit = _build_bp_circuit(n, ansatz)
    grad_fn = qml.grad(circuit, argnums=1)
    seed = _deterministic_cell_seed(n, depth, ansatz)
    rng = np.random.RandomState(seed)

    grads = np.zeros((n_init, n_params))
    costs = np.zeros(n_init)
    for init_i in range(n_init):
        w = pnp.array(rng.uniform(-math.pi, math.pi, shape), requires_grad=True)
        xp = pnp.array(x, requires_grad=False)
        costs[init_i] = float(circuit(xp, w))
        grads[init_i] = np.asarray(grad_fn(xp, w)).flatten()
    return grads, costs, seed


def verify_determinism(sample_cells=3, n_init_check=10):
    """Section 3.3 — spot-check that independent executions with the same
    configuration reproduce identical gradients. Recomputes a handful of
    cells twice (fresh RNG state each time, same deterministic seed) and
    asserts exact (not just close) agreement, since both runs are pure
    numpy/autograd with no nondeterministic ops."""
    print("  Verifying determinism (spot check)...")
    cells = [(n, d, a) for n in rconfig.BP_QUBITS[:2] for d in rconfig.BP_DEPTHS[:1]
             for a in rconfig.BP_ANSATZE][:sample_cells]
    x_by_n = {n: np.random.RandomState(n).uniform(-1.0, 1.0, n) for n in rconfig.BP_QUBITS}
    results = []
    for (n, d, a) in cells:
        g1, c1, seed1 = _compute_cell(n, d, a, x_by_n[n], n_init=n_init_check)
        g2, c2, seed2 = _compute_cell(n, d, a, x_by_n[n], n_init=n_init_check)
        max_diff = float(np.max(np.abs(g1 - g2)))
        ok = bool(np.array_equal(g1, g2)) and seed1 == seed2
        results.append({"n_qubits": n, "depth": d, "ansatz": a, "seed": seed1,
                         "max_abs_diff": max_diff, "exact_match": ok})
        print(f"    n={n} depth={d} {a}: seed={seed1} max_abs_diff={max_diff:.2e} exact_match={ok}")
    all_ok = all(r["exact_match"] for r in results)
    print(f"  Determinism check: {'PASS' if all_ok else 'FAIL'}")
    return {"all_passed": all_ok, "checks": results}


def run_barren_plateau(resume=False, save=True):
    print("=" * 65)
    print("  SECTION 16-17 (ROUND-2) — BARREN-PLATEAU GRADIENT-SCALING SWEEP")
    print("=" * 65)
    out_dir = rconfig.DIRS["barren_plateau"]
    raw_path = os.path.join(out_dir, "raw_gradients.csv")
    summary_path = os.path.join(out_dir, "summary.csv")
    jsonl_path = os.path.join(out_dir, "cells.jsonl")

    determinism_report = verify_determinism()
    utils.save_json(os.path.join(out_dir, "determinism_check.json"), determinism_report)

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
        "seed_derivation": "hashlib.sha256(f'{n_qubits}|{depth}|{ansatz}') truncated to 31 bits — "
            "deterministic across processes/runs, unlike Python's salted built-in hash()",
        "definitions": {
            "mean_grad": "mean of all signed gradient entries pooled across every init and parameter "
                "(SECONDARY — does not distinguish parameter identity from init identity)",
            "mean_abs_grad": "mean(|gradient|) pooled across every init and parameter",
            "median_abs_grad": "median(|gradient|) pooled across every init and parameter",
            "grad_l2_norm_mean": "mean, over inits, of the per-init L2 norm of the full gradient vector",
            "grad_l2_norm_std": "std, over inits, of the per-init L2 norm",
            "aggregate_grad_variance": "SECONDARY/legacy statistic — variance of ALL gradient entries "
                "pooled across every init AND parameter together. Retained for comparison with the "
                "original (pre-round-2) analysis only; not used as primary evidence because it conflates "
                "parameter-to-parameter variation with initialization-to-initialization variation.",
            "per_param_variance_vector": "V_p = Var_i(g[i, p]) for every trainable parameter p — the "
                "PRIMARY per-cell statistic, one value per parameter, each computed purely across the "
                "n_init random initializations (McClean et al. 2018's diagnostic, applied to every "
                "parameter rather than just one).",
            "mean_of_per_param_variances": "mean over p of V_p — PRIMARY headline scaling statistic.",
            "median_of_per_param_variances": "median over p of V_p — sensitivity/robustness companion "
                "to the mean.",
            "representative_param_variance": "V_p for one PRE-SPECIFIED parameter, index 0 (the classic "
                "single-parameter McClean et al. 2018 diagnostic) — reported for direct comparability "
                "with the literature, and as a parameter-selection sensitivity check against the "
                "mean/median across all parameters.",
            "n_exact_zero_grads": "count of gradient entries (pooled over init x parameter) that are "
                "exactly 0.0 — investigated rather than filtered out, per the reviewer's explicit "
                "instruction not to remove zero-valued gradients to manufacture a stronger trend.",
            "frac_underflow_like": "fraction of |gradient| entries below 1e-12 — a looser 'numerically "
                "negligible' threshold than exact zero, flagging possible underflow in deep/high-qubit "
                "cells even when not exactly representable as 0.0.",
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
            grads, costs, seed_used = _compute_cell(n, depth, ansatz, x_by_n[n])
            n_params = grads.shape[1]
            np.save(_raw_grads_path(out_dir, n, depth, ansatz), grads)

            for init_i in range(rconfig.BP_N_INIT):
                g = grads[init_i]
                raw_rows.append({
                    "n_qubits": n, "depth": depth, "ansatz": ansatz, "init_id": init_i, "seed": seed_used,
                    "cost": float(costs[init_i]), "grad_l2_norm": float(np.linalg.norm(g)),
                    "grad_mean_abs": float(np.mean(np.abs(g))),
                    "grad_vector_json": np.array2string(g, precision=8, separator=",", threshold=10000),
                })

            flat = grads.flatten()
            l2_per_init = np.linalg.norm(grads, axis=1)
            per_param_var = grads.var(axis=0)  # V_p for every p — PRIMARY
            rep_param = grads[:, 0]
            rep_var_ci = utils.bootstrap_ci(rep_param, np.var, n_boot=2000, seed=0)
            mean_ppv_ci = utils.bootstrap_ci(
                np.array([grads[:, p].var() for p in range(n_params)]), np.mean, n_boot=2000, seed=1)
            q75, q25 = np.percentile(np.abs(flat), [75, 25])

            n_exact_zero = int(np.sum(flat == 0.0))
            n_underflow_like = int(np.sum((flat != 0.0) & (np.abs(flat) < 1e-12)))

            summary_rows.append({
                "n_qubits": n, "depth": depth, "ansatz": ansatz, "seed": seed_used,
                "n_init": rconfig.BP_N_INIT, "n_params": n_params,
                "mean_of_per_param_variances": float(np.mean(per_param_var)),
                "median_of_per_param_variances": float(np.median(per_param_var)),
                "mean_of_per_param_variances_ci_lo": mean_ppv_ci["lo"],
                "mean_of_per_param_variances_ci_hi": mean_ppv_ci["hi"],
                "representative_param_variance": float(np.var(rep_param)),
                "representative_param_variance_ci_lo": rep_var_ci["lo"],
                "representative_param_variance_ci_hi": rep_var_ci["hi"],
                "aggregate_grad_variance": float(np.var(flat)), "aggregate_grad_std": float(np.std(flat)),
                "mean_grad": float(np.mean(flat)), "mean_abs_grad": float(np.mean(np.abs(flat))),
                "median_abs_grad": float(np.median(np.abs(flat))),
                "grad_l2_norm_mean": float(np.mean(l2_per_init)), "grad_l2_norm_std": float(np.std(l2_per_init)),
                "iqr_abs_grad": float(q75 - q25),
                "n_exact_zero_grads": n_exact_zero, "frac_exact_zero_grads": n_exact_zero / flat.size,
                "n_underflow_like": n_underflow_like, "frac_underflow_like": n_underflow_like / flat.size,
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


# ── scaling regression with initialization-level bootstrap ─────────────────

NONPOSITIVE_VARIANCE_FLOOR_NOTE = (
    "Before any log10 transform, a variance estimate <= 0 (possible for a "
    "bootstrap replicate that happens to resample near-duplicate "
    "initializations, though not observed in the point-estimate data) is "
    "floored at 1e-300 (far below any physically meaningful gradient "
    "variance in this experiment) rather than dropped, so every bootstrap "
    "replicate still contributes a finite slope estimate; floored "
    "replicates are counted and reported separately."
)


def _load_raw_grads(out_dir, n, depth, ansatz):
    path = _raw_grads_path(out_dir, n, depth, ansatz)
    if not os.path.exists(path):
        return None
    return np.load(path)


def _stat_mean_ppv(grads):
    return float(grads.var(axis=0).mean())


def _stat_median_ppv(grads):
    return float(np.median(grads.var(axis=0)))


def _stat_rep_param(grads):
    return float(grads[:, 0].var())


def _stat_pooled(grads):
    return float(grads.flatten().var())


def _bootstrap_regression(x_to_grad_matrices, n_boot, seed, stat_fn):
    """x_to_grad_matrices: {x_value: [grads_matrix, ...]} — each x's y-value
    at a given bootstrap replicate is stat_fn(resampled cell), averaged over
    every grads matrix listed for that x (a single matrix for a fixed-depth
    or fixed-qubit regression; several matrices, one per depth/qubit, for a
    headline regression averaged across the other axis). Resampling is
    always over the n_init axis WITHIN each cell, never over pooled
    individual gradient entries.
    """
    xs = sorted(x_to_grad_matrices)
    if len(xs) < 3:
        return None
    rng = np.random.RandomState(seed)
    n_floored = 0
    slopes = np.empty(n_boot)
    for b in range(n_boot):
        ys = []
        for x in xs:
            mats = x_to_grad_matrices[x]
            vals = []
            for m in mats:
                n_init = m.shape[0]
                resample_idx = rng.randint(0, n_init, n_init)
                vals.append(stat_fn(m[resample_idx]))
            ys.append(float(np.mean(vals)))
        ys = np.asarray(ys)
        floored = ys <= 0
        if floored.any():
            n_floored += 1
            ys = np.where(floored, 1e-300, ys)
        logy = np.log10(ys)
        reg = sps.linregress(np.asarray(xs, dtype=float), logy)
        slopes[b] = reg.slope
    lo, hi = np.quantile(slopes, [0.025, 0.975])

    # point estimate from the (unresampled) data itself
    point_ys = np.asarray([float(np.mean([stat_fn(m) for m in x_to_grad_matrices[x]])) for x in xs])
    point_floored = point_ys <= 0
    point_ys_safe = np.where(point_floored, 1e-300, point_ys)
    point_reg = sps.linregress(np.asarray(xs, dtype=float), np.log10(point_ys_safe))

    return {
        "x_values": xs, "n_points": len(xs),
        "slope_point_estimate": float(point_reg.slope),
        "slope_bootstrap_mean": float(np.mean(slopes)),
        "slope_ci95_lo": float(lo), "slope_ci95_hi": float(hi),
        "r_squared": float(point_reg.rvalue ** 2), "intercept": float(point_reg.intercept),
        "n_bootstrap_replicates": n_boot, "n_replicates_with_floored_variance": n_floored,
        "any_point_floored": bool(point_floored.any()),
    }


def run_scaling_analysis(save=True, n_boot=2000):
    """Section 18 (round-2) — fit scaling trends off the PRIMARY per-parameter
    variance statistic, with bootstrap 95% CIs, plus sensitivity analyses.
    Never auto-declares a barren plateau."""
    print("=" * 65)
    print("  SECTION 18 (ROUND-2) — BARREN-PLATEAU SCALING ANALYSIS")
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

        # ── fixed-depth qubit-scaling regressions (bootstrap CI) ──────────
        vs_n_by_depth = {}
        for depth in sorted(sub["depth"].unique()):
            s = sub[sub["depth"] == depth].sort_values("n_qubits")
            x_to_mats = {}
            for _, row in s.iterrows():
                g = _load_raw_grads(out_dir, int(row["n_qubits"]), int(depth), ansatz)
                if g is not None:
                    x_to_mats[float(row["n_qubits"])] = [g]
            fit = _bootstrap_regression(x_to_mats, n_boot, seed=0, stat_fn=_stat_mean_ppv)
            vs_n_by_depth[str(depth)] = fit or {"note": "insufficient points with saved raw gradients"}
        fits[ansatz]["per_param_variance_vs_qubits_by_depth"] = vs_n_by_depth

        # headline: average mean_of_per_param_variances across depths per qubit count
        headline_x_to_mats = {}
        for _, row in sub.iterrows():
            g = _load_raw_grads(out_dir, int(row["n_qubits"]), int(row["depth"]), ansatz)
            if g is not None:
                headline_x_to_mats.setdefault(float(row["n_qubits"]), []).append(g)
        headline_n = _bootstrap_regression(headline_x_to_mats, n_boot, seed=0, stat_fn=_stat_mean_ppv)
        if headline_n:
            headline_n["evidence_classification"] = _classify_evidence(
                headline_n["slope_point_estimate"], headline_n["slope_ci95_lo"], headline_n["slope_ci95_hi"],
                headline_n["r_squared"])
        fits[ansatz]["headline_vs_qubits"] = headline_n

        # ── fixed-qubit depth-scaling regressions (bootstrap CI) ──────────
        vs_d_by_n = {}
        for n in sorted(sub["n_qubits"].unique()):
            s = sub[sub["n_qubits"] == n].sort_values("depth")
            x_to_mats = {}
            for _, row in s.iterrows():
                g = _load_raw_grads(out_dir, int(n), int(row["depth"]), ansatz)
                if g is not None:
                    x_to_mats[float(row["depth"])] = [g]
            fit = _bootstrap_regression(x_to_mats, n_boot, seed=0, stat_fn=_stat_mean_ppv)
            vs_d_by_n[str(n)] = fit or {"note": "insufficient points with saved raw gradients"}
        fits[ansatz]["per_param_variance_vs_depth_by_qubits"] = vs_d_by_n

        headline_x_to_mats_d = {}
        for _, row in sub.iterrows():
            g = _load_raw_grads(out_dir, int(row["n_qubits"]), int(row["depth"]), ansatz)
            if g is not None:
                headline_x_to_mats_d.setdefault(float(row["depth"]), []).append(g)
        headline_d = _bootstrap_regression(headline_x_to_mats_d, n_boot, seed=0, stat_fn=_stat_mean_ppv)
        if headline_d:
            headline_d["evidence_classification"] = _classify_evidence(
                headline_d["slope_point_estimate"], headline_d["slope_ci95_lo"], headline_d["slope_ci95_hi"],
                headline_d["r_squared"])
        fits[ansatz]["headline_vs_depth"] = headline_d

        # ── sensitivity: parameter-selection (same headline-vs-qubits x's, different stat) ──
        sensitivity = {}
        for stat_name, stat_fn in [("mean_per_param_variance", _stat_mean_ppv),
                                    ("median_per_param_variance", _stat_median_ppv),
                                    ("representative_param_p0_variance", _stat_rep_param),
                                    ("pooled_aggregate_variance_legacy", _stat_pooled)]:
            r = _bootstrap_regression(headline_x_to_mats, n_boot=500, seed=0, stat_fn=stat_fn)
            sensitivity[stat_name] = {
                "slope_point_estimate": r["slope_point_estimate"], "r_squared": r["r_squared"],
                "slope_ci95_lo": r["slope_ci95_lo"], "slope_ci95_hi": r["slope_ci95_hi"],
            } if r else None
        fits[ansatz]["sensitivity_parameter_selection_vs_qubits"] = sensitivity

        # ── sensitivity: leave-one-qubit-out refits of the headline-vs-qubits point estimate ──
        loo = {}
        xs_sorted = sorted(headline_x_to_mats)
        for held_out in xs_sorted:
            reduced = {x: m for x, m in headline_x_to_mats.items() if x != held_out}
            if len(reduced) < 3:
                loo[str(held_out)] = {"note": "too few remaining points"}
                continue
            ys = np.asarray([float(np.mean([_stat_mean_ppv(m) for m in reduced[x]])) for x in sorted(reduced)])
            ys_safe = np.where(ys <= 0, 1e-300, ys)
            reg = sps.linregress(np.asarray(sorted(reduced), dtype=float), np.log10(ys_safe))
            loo[str(held_out)] = {"slope_without_this_qubit_count": float(reg.slope), "r_squared": float(reg.rvalue ** 2)}
        fits[ansatz]["sensitivity_leave_one_qubit_count_out"] = loo

        hn_cls = headline_n and headline_n.get("evidence_classification", {}).get("label")
        hd_cls = headline_d and headline_d.get("evidence_classification", {}).get("label")
        print(f"    {ansatz}: vs-qubits [{hn_cls}]; vs-depth [{hd_cls}]")

    fits["_methodology_note"] = {
        "primary_statistic": "mean_of_per_param_variances (V_p = Var_i(g[i,p]) averaged over p) — "
            "see metadata.json for the full definition and rationale.",
        "secondary_statistic": "pooled aggregate_grad_variance, retained only in the "
            "sensitivity_parameter_selection_vs_qubits block for comparison with the original "
            "(pre-round-2) analysis.",
        "bootstrap": "2,000-replicate initialization-level bootstrap (resampling complete per-init "
            "gradient vectors with replacement within each cell, recomputing the per-parameter "
            "variance statistic per replicate, then refitting the full log10(variance) ~ grid-axis "
            "regression) for every headline and fixed-depth/fixed-qubit regression.",
        "nonpositive_variance_handling": NONPOSITIVE_VARIANCE_FLOOR_NOTE,
    }

    if save:
        utils.save_json(os.path.join(out_dir, "scaling_fits.json"), fits)
        print(f"  Saved -> {os.path.join(out_dir, 'scaling_fits.json')}")
    return fits


def _classify_evidence(slope_point, ci_lo, ci_hi, r_squared):
    """Never auto-declares a barren plateau. Distinguishes finite-size
    gradient suppression / parameter-dependent behavior / gradient
    concentration from language implying asymptotic barren-plateau scaling,
    which these small simulated circuits (<=8 qubits, a LOCAL observable)
    cannot establish on their own."""
    ci_excludes_zero = (ci_hi < 0) or (ci_lo > 0)
    negative_and_significant = slope_point < 0 and ci_hi < 0

    if negative_and_significant and r_squared > 0.5:
        label = "gradient_suppression_with_grid_range"
        text = ("Gradient variance decreases with the tested grid axis, and the bootstrap 95% CI "
                "for the slope excludes zero (CI entirely negative) with R^2 > 0.5. This is evidence "
                "of FINITE-SIZE gradient suppression over the tested range (<=8 qubits, local "
                "PauliZ(0) observable). It is NOT, by itself, sufficient evidence for ASYMPTOTIC "
                "barren-plateau scaling (which requires evidence the trend continues/accelerates at "
                "much larger qubit counts, beyond what small-scale classical simulation can probe).")
    elif slope_point < 0 and ci_excludes_zero:
        label = "gradient_suppression_weaker_fit"
        text = ("Slope is negative with a 95% CI excluding zero, but R^2 <= 0.5 — some evidence of "
                "gradient suppression over the tested range, but the log-linear fit is not strongly "
                "explanatory; treat as suggestive, not conclusive.")
    elif slope_point < 0:
        label = "inconclusive_negative_trend"
        text = ("Point estimate is negative but the bootstrap 95% CI includes zero — insufficient "
                "evidence to distinguish a genuine decreasing trend from sampling noise at this grid "
                "resolution and n_init.")
    else:
        label = "no_evidence_of_suppression"
        text = ("Point estimate is non-negative, or the CI straddles/exceeds zero in the positive "
                "direction — no evidence of gradient-variance suppression over the tested range for "
                "this ansatz/axis.")
    return {"label": label, "text": text}
