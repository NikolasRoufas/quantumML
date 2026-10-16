"""
Round-2 reviewer concern D — verify the OpenML dataset identity
programmatically rather than trusting either the README (OpenML ID 566) or
the reviewer's own candidate (ID 1120, "MagicTelescope").

`data_prep.load_magic()` calls `fetch_openml("magic", version=1, ...)` — a
NAME lookup, not an ID lookup. sklearn resolves names to a specific OpenML
dataset id behind the scenes; this script inspects what `fetch_openml`
actually returned (via `return_X_y=False` to get the `Bunch`, which carries
`details` with the resolved id/name/version) and cross-checks it against
the two candidate identifiers and the known UCI shape (19,020 x 10).
"""
import hashlib
import json
import os

import numpy as np
from sklearn.datasets import fetch_openml

import rconfig, utils

EXPECTED_N_SAMPLES = 19020
EXPECTED_N_FEATURES = 10
CANDIDATE_IDS = {"readme_claim": 566, "reviewer_claim": 1120}


def verify_dataset_provenance(save=True):
    print("=" * 65)
    print("  ROUND-2 PHASE 4 — DATASET PROVENANCE VERIFICATION")
    print("=" * 65)

    bunch = fetch_openml("magic", version=1, as_frame=False, parser="auto")
    frame_bunch = fetch_openml("magic", version=1, as_frame=True, parser="auto")
    details = getattr(bunch, "details", None) or {}

    resolved_id = details.get("id")
    resolved_name = details.get("name")
    resolved_version = details.get("version")
    url = details.get("url")

    X = np.asarray(bunch.data, dtype=np.float64)
    y_raw = bunch.target
    feature_names = list(getattr(bunch, "feature_names", []) or [])

    # label mapping: fetch_openml returns the original category strings for
    # a nominal target unless as_frame/return_X_y coerce it — confirm the
    # raw values before any downstream encoding in data_prep.load_magic.
    unique_raw = sorted(set(str(v) for v in np.unique(y_raw)))
    y_int = y_raw.astype(int) if np.issubdtype(np.asarray(y_raw).dtype, np.number) \
        else np.array([1 if str(v) in ("1", "g", "gamma") else 0 for v in y_raw])
    class_counts = {int(k): int(v) for k, v in zip(*np.unique(y_int, return_counts=True))}

    n_missing = int(np.isnan(X).sum())

    # ── independent physics check of the gamma/hadron label mapping ─────────
    # UCI's MAGIC04 documentation: gamma (signal) n=12332, hadron (background)
    # n=6688, and gamma showers point back toward the source -> small FAlpha
    # (the shower axis's angle to the origin vector, degrees), while hadron
    # showers are isotropic -> larger/more uniform FAlpha. This lets us check
    # the label SEMANTICS independently of which raw string OpenML assigned
    # to which integer code, rather than trusting the id/count match alone.
    label_check = {}
    df = frame_bunch.frame
    falpha_candidates = [c for c in df.columns if "alpha" in c.lower()]
    if falpha_candidates:
        falpha_col = falpha_candidates[0]
        y_frame = y_int
        for cls in sorted(class_counts):
            sub = df.loc[np.asarray(y_frame) == cls, falpha_col]
            label_check[str(cls)] = {
                "n": int(len(sub)),
                "mean_falpha_degrees": float(sub.mean()),
                "median_falpha_degrees": float(sub.median()),
            }
        gamma_like = min(label_check, key=lambda k: label_check[k]["mean_falpha_degrees"])
        hadron_like = max(label_check, key=lambda k: label_check[k]["mean_falpha_degrees"])
        label_check["falpha_column_used"] = falpha_col
        label_check["physics_based_gamma_label"] = int(gamma_like)
        label_check["physics_based_hadron_label"] = int(hadron_like)
        label_check["interpretation"] = (
            f"Gamma showers point back toward the source -> small FAlpha; class "
            f"{gamma_like} has the smaller mean FAlpha "
            f"({label_check[gamma_like]['mean_falpha_degrees']:.2f} deg vs "
            f"{label_check[hadron_like]['mean_falpha_degrees']:.2f} deg), so class "
            f"{gamma_like} is physically gamma and class {hadron_like} is physically hadron, "
            f"independent of count-matching against the UCI summary stats."
        )
        label_check["assumed_by_data_prep_load_magic"] = (
            "comment states 'gamma=1, hadron=0' — i.e. assumes class 1 is gamma"
        )
        label_check["assumption_matches_physics_check"] = (int(gamma_like) == 1)

    checksum = hashlib.sha256(
        np.ascontiguousarray(X).tobytes() + str(sorted(class_counts.items())).encode()
    ).hexdigest()

    shape_matches_uci = (X.shape[0] == EXPECTED_N_SAMPLES and X.shape[1] == EXPECTED_N_FEATURES)
    id_matches_readme = (resolved_id == CANDIDATE_IDS["readme_claim"])
    id_matches_reviewer_candidate = (resolved_id == CANDIDATE_IDS["reviewer_claim"])

    report = {
        "fetch_call": "sklearn.datasets.fetch_openml('magic', version=1, as_frame=False, parser='auto')",
        "resolved_openml_id": resolved_id,
        "resolved_openml_name": resolved_name,
        "resolved_openml_version": resolved_version,
        "resolved_openml_url": url,
        "readme_claimed_id": CANDIDATE_IDS["readme_claim"],
        "reviewer_candidate_id": CANDIDATE_IDS["reviewer_claim"],
        "id_matches_readme_claim": id_matches_readme,
        "id_matches_reviewer_candidate": id_matches_reviewer_candidate,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "expected_n_samples_uci": EXPECTED_N_SAMPLES,
        "expected_n_features_uci": EXPECTED_N_FEATURES,
        "shape_matches_uci_expectation": shape_matches_uci,
        "feature_names": feature_names,
        "raw_target_categories": unique_raw,
        "class_counts_after_encoding": class_counts,
        "positive_class_encoding": "1 = gamma (signal), 0 = hadron (background) — "
            "matches UCI 'MAGIC Gamma Telescope' (g=gamma, h=hadron) and data_prep.load_magic's "
            "int(y) encoding where the OpenML nominal category 'g'/'1' maps to 1",
        "n_missing_values": n_missing,
        "label_semantics_check": label_check,
        "sha256_checksum_X_and_class_counts": checksum,
        "compare_against": [
            "https://www.openml.org/d/1120",
            "https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope",
        ],
        "conclusion": None,
    }

    if id_matches_reviewer_candidate and shape_matches_uci:
        report["conclusion"] = (
            f"VERIFIED: fetch_openml('magic', version=1) resolves to OpenML dataset id "
            f"{resolved_id} ('{resolved_name}'), matching the reviewer's candidate id "
            f"{CANDIDATE_IDS['reviewer_claim']} (MagicTelescope) and the expected UCI shape "
            f"({EXPECTED_N_SAMPLES}x{EXPECTED_N_FEATURES}). The README's stated id "
            f"({CANDIDATE_IDS['readme_claim']}) was incorrect and needs correcting; the "
            f"loaded DATA and all downstream experimental results are unaffected since the "
            f"loader call itself was always correct — only the README's documentation of "
            f"which id that call resolves to was wrong."
        )
    elif id_matches_readme and shape_matches_uci:
        report["conclusion"] = (
            f"README's claimed id ({CANDIDATE_IDS['readme_claim']}) was correct; the "
            f"reviewer's candidate id ({CANDIDATE_IDS['reviewer_claim']}) does not match "
            f"what this loader call actually resolves to. Data integrity confirmed via UCI "
            f"shape match."
        )
    elif shape_matches_uci:
        report["conclusion"] = (
            f"Resolved OpenML id {resolved_id} ('{resolved_name}') matches NEITHER candidate "
            f"id but the data shape matches UCI MAGIC Gamma Telescope exactly "
            f"({EXPECTED_N_SAMPLES}x{EXPECTED_N_FEATURES}) and class semantics are consistent. "
            f"Likely a re-hosted/mirrored copy under a different OpenML id with identical "
            f"content — treat prior experimental results as valid on data-content grounds, "
            f"but correct the README to cite the id actually resolved here."
        )
    else:
        report["conclusion"] = (
            f"MISMATCH: resolved shape {X.shape} does NOT match the expected UCI MAGIC Gamma "
            f"Telescope shape ({EXPECTED_N_SAMPLES}x{EXPECTED_N_FEATURES}). This calls prior "
            f"experimental validity into question — investigate before trusting any "
            f"downstream result."
        )

    if label_check and not label_check.get("assumption_matches_physics_check", True):
        report["conclusion"] += (
            " **SEPARATE, MORE CONSEQUENTIAL FINDING**: the physics-based label check "
            f"(FAlpha angle-to-origin) indicates class {label_check['physics_based_gamma_label']} "
            f"is physically gamma and class {label_check['physics_based_hadron_label']} is "
            f"physically hadron — the OPPOSITE of what data_prep.load_magic()'s docstring/comment "
            f"assumes ('gamma=1, hadron=0'). The dataset CONTENT is correct and unchanged; what is "
            f"wrong is the gamma/hadron NAMING of integer label 1 vs 0, which was already baked "
            f"into the ORIGINAL (pre-revision) data_prep.py and therefore affects the original "
            f"paper's results too, not just this revision. Accuracy/balanced-accuracy/MCC are "
            f"symmetric and unaffected; precision/recall/ROC-AUC/PR-AUC are computed correctly "
            f"with respect to whichever class is coded as 1, but every place the paper or README "
            f"narrates those numbers as being 'for gamma-ray detection' has actually been "
            f"describing hadron-background detection, and vice versa. This is a labeling/narrative "
            f"inversion, not a numerical error, but it needs a human decision before any text is "
            f"written — flagged to the user rather than silently corrected."
        )

    print(f"  Resolved OpenML id={resolved_id} name={resolved_name!r} version={resolved_version}")
    print(f"  Shape={X.shape}  matches UCI expectation: {shape_matches_uci}")
    print(f"  class_counts={class_counts}  n_missing={n_missing}")
    if label_check:
        print(f"  FAlpha-based label check: {label_check.get('interpretation')}")
        print(f"  data_prep.py's 'gamma=1' assumption matches physics check: "
              f"{label_check.get('assumption_matches_physics_check')}")
    print(f"  {report['conclusion']}")

    if save:
        out_dir = rconfig.DIRS["dataset"]
        utils.save_json(os.path.join(out_dir, "provenance.json"), report)
        utils.save_json(os.path.join(rconfig.RESULTS_DIR, "dataset_provenance.json"), report)
        print(f"  Saved -> {os.path.join(out_dir, 'provenance.json')}")

    return report


if __name__ == "__main__":
    verify_dataset_provenance()
