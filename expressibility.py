"""
Sections 14-15 — Recognized circuit expressibility (Sim et al. 2019 style)
and its descriptive association with gradient variance.

This REPLACES the original analysis.compute_expressibility, which was found
to be broken: it built the identical circuit (ZZ feature map +
StronglyEntanglingLayers) for every "ansatz" label, so it never actually
compared BasicEnt/CustomAnsatz — the KL values reported in the README are
noise from re-sampling one fixed circuit three times, not a real ansatz
comparison. See circuit_validation.json / the accompanying report.

Here, expressibility assesses the ANSATZ (Section 14 requirement): each
ansatz is evaluated on its own, with NO feature map / data dependence —
random parameter pairs are drawn and applied directly to |0>^n, exactly the
protocol in Sim et al. (2019).
"""
import os
import math
import numpy as np
import pandas as pd
import pennylane as qml

import rconfig, utils, circuits as rcircuits


def _haar_fidelity_pdf(f_vals, n_qubits):
    d = 2 ** n_qubits
    return (d - 1) * (1 - f_vals) ** (d - 2)


def _kl_from_fidelities(fidelities, n_qubits, n_bins):
    bins = np.linspace(0, 1, n_bins)
    centres = (bins[:-1] + bins[1:]) / 2
    eps = 1e-10
    haar_pdf = _haar_fidelity_pdf(centres, n_qubits)
    haar_pdf = haar_pdf / haar_pdf.sum() + eps
    hist, _ = np.histogram(fidelities, bins=bins, density=False)
    sampled_pdf = hist / hist.sum() + eps
    return float(np.sum(sampled_pdf * np.log(sampled_pdf / haar_pdf)))


def _frame_potential_proxy(fidelities, t=1):
    """Empirical t-th frame potential estimate: E[F^t] over sampled pairs.
    Secondary/optional measure (Section 14)."""
    return float(np.mean(np.asarray(fidelities) ** t))


def run_expressibility(save=True):
    print("=" * 65)
    print("  SECTION 14 — RECOGNIZED CIRCUIT EXPRESSIBILITY (ansatz-only, label-free)")
    print("=" * 65)
    n = rconfig.N_QUBITS
    L = rconfig.N_LAYERS
    out_dir = rconfig.DIRS["expressibility"]
    dev = qml.device("default.qubit", wires=n)

    raw_rows = []
    boot_rows = []
    summary_rows = {}

    for ansatz in rconfig.BP_ANSATZE:
        print(f"    {ansatz}: sampling {rconfig.EXPRESSIBILITY_N_REPETITIONS} x "
              f"{rconfig.EXPRESSIBILITY_N_PAIRS} random state pairs...")
        shape = rcircuits.weight_shape(ansatz, L, n)
        ansatz_fn = rcircuits.ANSATZ_FNS[ansatz]

        @qml.qnode(dev)
        def state_circ(weights, _ansatz_fn=ansatz_fn):
            _ansatz_fn(weights, n)
            return qml.state()

        rep_kls = []
        all_fid_this_ansatz = []
        for rep in range(rconfig.EXPRESSIBILITY_N_REPETITIONS):
            rng = np.random.RandomState(1000 * rep + hash(ansatz) % 1000)
            fids = []
            for _ in range(rconfig.EXPRESSIBILITY_N_PAIRS):
                w1 = rng.uniform(-math.pi, math.pi, shape)
                w2 = rng.uniform(-math.pi, math.pi, shape)
                sv1 = np.asarray(state_circ(w1))
                sv2 = np.asarray(state_circ(w2))
                fid = float(np.abs(np.dot(np.conj(sv1), sv2)) ** 2)
                fids.append(fid)
            kl = _kl_from_fidelities(fids, n, rconfig.EXPRESSIBILITY_N_BINS)
            rep_kls.append(kl)
            all_fid_this_ansatz.extend(fids)
            for f in fids:
                raw_rows.append({"ansatz": ansatz, "repetition": rep, "fidelity": f})

        rep_kls = np.array(rep_kls)
        ci = utils.bootstrap_ci(rep_kls, np.mean, n_boot=rconfig.EXPRESSIBILITY_BOOTSTRAP_N, seed=42)
        # also bootstrap directly over the pooled fidelity samples (recomputing KL each draw)
        rng_b = np.random.RandomState(7)
        pooled = np.array(all_fid_this_ansatz)
        boot_kls = np.empty(500)  # cheaper N here — recomputing KL per resample
        for b in range(500):
            sample = pooled[rng_b.randint(0, len(pooled), len(pooled))]
            boot_kls[b] = _kl_from_fidelities(sample, n, rconfig.EXPRESSIBILITY_N_BINS)
        pooled_ci_lo, pooled_ci_hi = np.quantile(boot_kls, [0.025, 0.975])

        frame_potential = _frame_potential_proxy(pooled, t=1)

        summary_rows[ansatz] = {
            "n_qubits": n, "n_layers": L, "n_pairs_per_repetition": rconfig.EXPRESSIBILITY_N_PAIRS,
            "n_repetitions": rconfig.EXPRESSIBILITY_N_REPETITIONS,
            "kl_mean": float(rep_kls.mean()), "kl_std": float(rep_kls.std()),
            "kl_ci95_across_repetitions": [ci["lo"], ci["hi"]],
            "kl_ci95_pooled_bootstrap": [float(pooled_ci_lo), float(pooled_ci_hi)],
            "frame_potential_t1_proxy": frame_potential,
            "note": "Lower KL divergence to the Haar-random fidelity distribution = "
                    "more expressive. This metric is data-independent — it characterizes "
                    "the ansatz alone, not any feature map or dataset.",
        }
        for b_kl in rep_kls:
            boot_rows.append({"ansatz": ansatz, "kl": float(b_kl)})
        print(f"      KL mean={rep_kls.mean():.4f} std={rep_kls.std():.4f} "
              f"CI95=[{ci['lo']:.4f},{ci['hi']:.4f}]")

    if save:
        pd.DataFrame(raw_rows).to_csv(os.path.join(out_dir, "raw_results.csv"), index=False)
        pd.DataFrame(boot_rows).to_csv(os.path.join(out_dir, "bootstrap_results.csv"), index=False)
        summ_df = pd.DataFrame([{"ansatz": a, **v} for a, v in summary_rows.items()])
        summ_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
        utils.save_json(os.path.join(out_dir, "expressibility_results.json"), summary_rows)
        print(f"  Saved -> {out_dir}")

    return summary_rows


def run_association(save=True):
    """Section 15 — descriptive association between ansatz expressibility and
    gradient variance / magnitude, from the dedicated barren-plateau results
    (Section 16-18) at the SAME (n_qubits, n_layers) as the primary factorial."""
    print("=" * 65)
    print("  SECTION 15 — EXPRESSIBILITY vs TRAINABILITY ASSOCIATION (descriptive)")
    print("=" * 65)
    expr_path = os.path.join(rconfig.DIRS["expressibility"], "expressibility_results.json")
    bp_path = os.path.join(rconfig.DIRS["barren_plateau"], "summary.csv")

    expr = utils.load_json(expr_path)
    if expr is None:
        print("  Expressibility results not found — run Section 14 first.")
        return None
    if not os.path.exists(bp_path):
        print("  Barren-plateau summary not found — run Section 16-18 first. Skipping association.")
        return None

    bp = pd.read_csv(bp_path)
    sub = bp[(bp["n_qubits"] == rconfig.N_QUBITS) & (bp["depth"] == rconfig.N_LAYERS)]
    if sub.empty:
        print(f"  No barren-plateau rows at n_qubits={rconfig.N_QUBITS}, depth={rconfig.N_LAYERS}. Skipping.")
        return None

    rows = []
    for ansatz, e in expr.items():
        bp_row = sub[sub["ansatz"] == ansatz]
        if bp_row.empty:
            continue
        rows.append({
            "ansatz": ansatz,
            "kl_mean": e["kl_mean"],
            "grad_variance_mean": float(bp_row["aggregate_grad_variance"].iloc[0]),
            "grad_abs_mean": float(bp_row["mean_abs_grad"].iloc[0]),
        })
    df = pd.DataFrame(rows)
    if len(df) >= 3:
        r_var = float(df["kl_mean"].corr(df["grad_variance_mean"], method="spearman"))
        r_abs = float(df["kl_mean"].corr(df["grad_abs_mean"], method="spearman"))
    else:
        r_var = r_abs = None

    df["note"] = ("Descriptive association only across 3 ansatz families — "
                   "not sufficient to claim causation.")
    if save:
        out_path = os.path.join(rconfig.DIRS["expressibility"], "expressibility_gradient_association.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path}")
        if r_var is not None:
            print(f"  Spearman r(KL, grad_variance) = {r_var:.3f} (n=3, descriptive only — "
                  f"do not treat as a hypothesis test)")
    return df
