"""
Journal-revision experimental package — main entry point.

Does NOT overwrite run_experiments.py or any original module. Supports
staged, resumable execution so expensive analyses don't need to be rerun
from scratch after an interruption.

Usage:
    python run_revision_experiments.py --stage audit
    python run_revision_experiments.py --stage validate-circuits
    python run_revision_experiments.py --stage factorial [--resume]
    python run_revision_experiments.py --stage matched-budget [--resume]
    python run_revision_experiments.py --stage baselines [--resume]
    python run_revision_experiments.py --stage separability
    python run_revision_experiments.py --stage expressibility
    python run_revision_experiments.py --stage barren-plateau [--resume]
    python run_revision_experiments.py --stage split-robustness [--resume]
    python run_revision_experiments.py --stage noise [--resume]
    python run_revision_experiments.py --stage unified
    python run_revision_experiments.py --stage figures
    python run_revision_experiments.py --stage sanity
    python run_revision_experiments.py --stage manifest
    python run_revision_experiments.py --stage summary
    python run_revision_experiments.py --stage all [--resume] [--skip-noise]
"""
import argparse
import time
import traceback

from revision import rconfig, utils

STAGE_ORDER = [
    "audit", "validate-circuits", "factorial", "matched-budget", "baselines",
    "separability", "expressibility", "barren-plateau", "split-robustness",
    "unified", "figures", "sanity", "noise", "manifest", "summary",
]


def _load_data():
    import data_prep
    X, y = data_prep.load_magic()
    return X, y


def stage_audit(args, ctx):
    from revision import dataset_audit
    audit, X, y = dataset_audit.run_audit()
    ctx["X"], ctx["y"], ctx["audit"] = X, y, audit


def stage_validate_circuits(args, ctx):
    from revision import validate_circuits
    validate_circuits.run_validation()


def stage_factorial(args, ctx):
    from revision import primary_split, factorial
    X, y = ctx.get("X"), ctx.get("y")
    if X is None:
        X, y = _load_data()
    split = primary_split.load_primary_split(X, y)
    factorial.run_factorial(split, resume=args.resume)
    factorial.run_statistical_analysis()


def stage_matched_budget(args, ctx):
    from revision import primary_split, matched_budget
    X, y = ctx.get("X"), ctx.get("y")
    if X is None:
        X, y = _load_data()
    split = primary_split.load_primary_split(X, y)
    matched_budget.run_matched_budget(split, resume=args.resume)


def stage_baselines(args, ctx):
    from revision import primary_split, matched_baselines
    X, y = ctx.get("X"), ctx.get("y")
    if X is None:
        X, y = _load_data()
    split = primary_split.load_primary_split(X, y)
    matched_baselines.run_classical(split, resume=args.resume)
    matched_baselines.run_qkernel(split, resume=args.resume)


def stage_separability(args, ctx):
    from revision import primary_split, separability
    X, y = ctx.get("X"), ctx.get("y")
    if X is None:
        X, y = _load_data()
    split = primary_split.load_primary_split(X, y)
    separability.run_separability(split)


def stage_expressibility(args, ctx):
    from revision import expressibility
    expressibility.run_expressibility()
    expressibility.run_association()


def stage_barren_plateau(args, ctx):
    from revision import barren_plateau
    barren_plateau.run_barren_plateau(resume=args.resume)
    barren_plateau.run_scaling_analysis()


def stage_split_robustness(args, ctx):
    from revision import split_robustness
    X, y = ctx.get("X"), ctx.get("y")
    if X is None:
        X, y = _load_data()
    split_robustness.run_split_robustness(X, y, resume=args.resume)


def stage_noise(args, ctx):
    from revision import noise
    noise.run_noise(resume=args.resume)


def stage_unified(args, ctx):
    from revision import unified
    unified.build_unified_comparison()


def stage_figures(args, ctx):
    from revision import figures
    figures.generate_all_figures()


def stage_sanity(args, ctx):
    from revision import sanity
    sanity.run_sanity_checks()


def stage_manifest(args, ctx):
    from revision import manifest
    manifest.write_manifest()


def stage_summary(args, ctx):
    from revision import summary
    summary.print_final_summary()
    summary.write_final_results()


STAGE_FNS = {
    "audit": stage_audit,
    "validate-circuits": stage_validate_circuits,
    "factorial": stage_factorial,
    "matched-budget": stage_matched_budget,
    "baselines": stage_baselines,
    "separability": stage_separability,
    "expressibility": stage_expressibility,
    "barren-plateau": stage_barren_plateau,
    "split-robustness": stage_split_robustness,
    "noise": stage_noise,
    "unified": stage_unified,
    "figures": stage_figures,
    "sanity": stage_sanity,
    "manifest": stage_manifest,
    "summary": stage_summary,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                         choices=list(STAGE_FNS.keys()) + ["all"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-noise", action="store_true",
                         help="Skip the optional noise-robustness stage when --stage all")
    args = parser.parse_args()

    rconfig.ensure_dirs()
    ctx = {}

    stages = STAGE_ORDER if args.stage == "all" else [args.stage]
    if args.stage == "all" and args.skip_noise:
        stages = [s for s in stages if s != "noise"]

    for stage in stages:
        print("\n" + "#" * 70)
        print(f"# STAGE: {stage}")
        print("#" * 70, flush=True)
        t0 = time.time()
        try:
            STAGE_FNS[stage](args, ctx)
        except Exception as e:
            print(f"\n!!! STAGE '{stage}' FAILED: {e}")
            traceback.print_exc()
            utils.record_failure(str(e), stage=stage, traceback=traceback.format_exc())
            if stage == "noise":
                print("  Optional noise stage failed/skipped — continuing.")
                continue
            if args.stage == "all":
                print(f"  Continuing to remaining stages where scientifically valid.")
                continue
            raise
        print(f"# STAGE '{stage}' done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
