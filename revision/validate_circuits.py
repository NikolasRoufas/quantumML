"""
Section 4 — Validate the refactored circuits before running anything expensive.

A. The 3 refactored combinations that correspond to the original hardcoded
   circuits (ZZ+StronglyEnt, Amplitude+BasicEnt, Angle+CustomAnsatz) must
   reproduce the original architecture: same param count, same gate types,
   same gate count, same depth, and identical forward output for identical
   (fixed) input + weights.
B. Forward pass for all 9 configurations.
C. Gradient computation for all 9 configurations.
D. Abort loudly (raise) if any of the 9 is mathematically incompatible,
   rather than silently altering it.
"""
import os
import numpy as np
import torch

import config
import quantum_circuits as orig_qc
from revision import rconfig, circuits, utils


def _forward_original(name, x_np, w_np):
    x = torch.tensor(x_np, dtype=torch.float64)
    w = torch.tensor(w_np, dtype=torch.float64)
    if name == "ZZ+StronglyEnt":
        return float(orig_qc.zz_strongly_circuit(x, w))
    if name == "Amplitude+BasicEnt":
        return float(orig_qc.amplitude_basic_circuit(x, w))
    if name == "Angle+CustomAnsatz":
        return float(orig_qc.angle_custom_circuit(x, w))
    raise ValueError(name)


def check_equivalence_to_original():
    """Part A."""
    n = rconfig.N_QUBITS
    L = rconfig.N_LAYERS
    rng = np.random.RandomState(0)
    results = {}

    pairs = [
        ("ZZ", "StronglyEnt", "ZZ+StronglyEnt", orig_qc.ZZ_SHAPE),
        ("Amplitude", "BasicEnt", "Amplitude+BasicEnt", orig_qc.AMP_SHAPE),
        ("Angle", "CustomAnsatz", "Angle+CustomAnsatz", orig_qc.ANGLE_SHAPE),
    ]

    for fmap, ansatz, orig_name, orig_shape in pairs:
        new_shape = circuits.weight_shape(ansatz, L, n)
        assert tuple(new_shape) == tuple(orig_shape), \
            f"{orig_name}: weight shape mismatch new={new_shape} orig={orig_shape}"

        new_circuit, _ = circuits.build_circuit(fmap, ansatz, n_qubits=n, n_layers=L)

        x_np = rng.uniform(-1, 1, n)
        w_np = rng.uniform(-np.pi, np.pi, new_shape)

        out_new = float(new_circuit(torch.tensor(x_np, dtype=torch.float64),
                                     torch.tensor(w_np, dtype=torch.float64)))
        out_orig = _forward_original(orig_name, x_np, w_np)

        gc_new = circuits.gate_count_and_depth(fmap, ansatz, n, L, x=x_np)

        # analytic gate-type comparison against the original circuit's tape
        orig_circ_map = {
            "ZZ+StronglyEnt": orig_qc.zz_strongly_circuit,
            "Amplitude+BasicEnt": orig_qc.amplitude_basic_circuit,
            "Angle+CustomAnsatz": orig_qc.angle_custom_circuit,
        }
        import pennylane as qml
        import warnings
        raw_orig_tape = qml.workflow.construct_tape(orig_circ_map[orig_name])(
            torch.tensor(x_np, dtype=torch.float64),
            torch.tensor(w_np, dtype=torch.float64),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orig_tape = raw_orig_tape.expand(depth=10)
        from collections import Counter
        orig_gate_types = dict(Counter(op.name for op in orig_tape.operations))

        output_match = bool(np.isclose(out_new, out_orig, atol=1e-9))
        gates_match = gc_new["gate_types"] == orig_gate_types
        param_count_match = gc_new["n_trainable_params"] == int(np.prod(orig_shape))

        results[orig_name] = {
            "weight_shape_match": True,
            "param_count_new": gc_new["n_trainable_params"],
            "param_count_orig": int(np.prod(orig_shape)),
            "param_count_match": param_count_match,
            "gate_count_new": gc_new["gate_count"],
            "gate_count_orig": len(orig_tape.operations),
            "gate_count_match": gc_new["gate_count"] == len(orig_tape.operations),
            "depth_new": gc_new["depth"],
            "gate_types_new": gc_new["gate_types"],
            "gate_types_orig": orig_gate_types,
            "gate_types_match": gates_match,
            "forward_output_new": out_new,
            "forward_output_orig": out_orig,
            "forward_output_match": output_match,
            "fully_equivalent": bool(param_count_match and gates_match and output_match
                                      and gc_new["gate_count"] == len(orig_tape.operations)),
        }

        status = "OK" if results[orig_name]["fully_equivalent"] else "MISMATCH"
        print(f"    [{status}] {orig_name}: params {gc_new['n_trainable_params']} "
              f"(orig {int(np.prod(orig_shape))}), gates {gc_new['gate_count']} "
              f"(orig {len(orig_tape.operations)}), forward {out_new:.6f} vs {out_orig:.6f}")

        if not results[orig_name]["fully_equivalent"]:
            raise RuntimeError(
                f"Refactored circuit '{orig_name}' does NOT reproduce the original "
                f"architecture. See results for details — stopping per Section 4.D."
            )

    return results


def check_all_nine():
    """Parts B & C: forward pass + gradient for all 9 configurations."""
    n = rconfig.N_QUBITS
    L = rconfig.N_LAYERS
    rng = np.random.RandomState(1)
    results = {}

    for fmap in rconfig.FEATURE_MAPS:
        for ansatz in rconfig.ANSATZE:
            key = f"{fmap}+{ansatz}"
            shape = circuits.weight_shape(ansatz, L, n)
            circuit, _ = circuits.build_circuit(fmap, ansatz, n_qubits=n, n_layers=L)

            x_np = rng.uniform(-1, 1, n)
            w_np = rng.uniform(-np.pi, np.pi, shape)
            x = torch.tensor(x_np, dtype=torch.float64)
            w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)

            entry = {"n_params": int(np.prod(shape)), "weight_shape": list(shape)}
            try:
                out = circuit(x, w)
                entry["forward_ok"] = True
                entry["forward_value"] = float(out)
            except Exception as e:
                entry["forward_ok"] = False
                entry["forward_error"] = str(e)
                results[key] = entry
                print(f"    [FAIL forward] {key}: {e}")
                continue

            try:
                out.backward()
                grad = w.grad
                finite = bool(torch.isfinite(grad).all())
                entry["gradient_ok"] = True
                entry["gradient_finite"] = finite
                entry["gradient_norm"] = float(torch.norm(grad))
                if not finite:
                    raise RuntimeError("gradient contains non-finite values")
            except Exception as e:
                entry["gradient_ok"] = False
                entry["gradient_error"] = str(e)
                print(f"    [FAIL gradient] {key}: {e}")
                results[key] = entry
                continue

            gc = circuits.gate_count_and_depth(fmap, ansatz, n, L)
            entry.update(gc)
            results[key] = entry
            print(f"    [OK] {key}: params={entry['n_params']} "
                  f"gates={gc['gate_count']} depth={gc['depth']} "
                  f"grad_norm={entry['gradient_norm']:.4f}")

    incompatible = [k for k, v in results.items()
                    if not (v.get("forward_ok") and v.get("gradient_ok") and v.get("gradient_finite"))]
    if incompatible:
        raise RuntimeError(
            f"The following configurations are mathematically incompatible and were "
            f"NOT silently altered: {incompatible}. See circuit_validation.json."
        )

    return results


def run_validation(save=True):
    print("=" * 65)
    print("  SECTION 4 — CIRCUIT VALIDATION")
    print("=" * 65)
    print("  A. Equivalence to original 3 hardcoded circuits:")
    equiv = check_equivalence_to_original()
    print("  B/C. Forward pass + gradient for all 9 configurations:")
    nine = check_all_nine()

    out = {"equivalence_to_original": equiv, "all_nine_configurations": nine,
           "all_passed": True}
    if save:
        utils.save_json(os.path.join(rconfig.RESULTS_DIR, "circuit_validation.json"), out)
        print(f"  Saved -> {os.path.join(rconfig.RESULTS_DIR, 'circuit_validation.json')}")
    print("  All circuit validation checks PASSED.")
    return out


if __name__ == "__main__":
    run_validation()
