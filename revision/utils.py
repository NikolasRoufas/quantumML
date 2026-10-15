"""Shared helpers for the revision experimental package."""
import os
import sys
import json
import time
import random
import platform
import subprocess
import numpy as np


# ── seeding ────────────────────────────────────────────────────────────────

def set_all_seeds(seed):
    """Seed every RNG that influences a run: python, numpy, torch, PennyLane's
    numpy-wrapper RNG (shared with plain numpy)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ── JSON-safety ───────────────────────────────────────────────────────────

def to_jsonable(obj):
    """Recursively convert numpy/torch/pandas-ish objects to plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "detach") and hasattr(obj, "numpy"):  # torch tensor
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


# ── incremental / resumable run store ────────────────────────────────────
# A "run store" is a JSONL file: one JSON object per completed run. This lets
# --resume skip configurations that already have a persisted record, and
# every individual run is flushed to disk immediately (Section 24).

def load_completed_keys(jsonl_path, key_fields):
    """Return the set of (key_fields...) tuples already present in a JSONL run log."""
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(tuple(rec.get(k) for k in key_fields))
    return done


def append_jsonl(jsonl_path, record):
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(to_jsonable(record)) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(jsonl_path):
    if not os.path.exists(jsonl_path):
        return []
    out = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def record_failure(reason, **context):
    """Append one failure record to results/revision/failed_runs.json (Section 24)."""
    from revision import rconfig
    path = os.path.join(rconfig.RESULTS_DIR, "failed_runs.json")
    existing = load_json(path, default=[])
    existing.append(to_jsonable({"reason": str(reason), "timestamp": time.time(), **context}))
    save_json(path, existing)


# ── bootstrap CI ──────────────────────────────────────────────────────────

def bootstrap_ci(values, statistic=np.mean, n_boot=2000, ci=0.95, seed=0):
    """Percentile bootstrap CI for `statistic` applied to 1D array `values`."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.RandomState(seed)
    point = float(statistic(values))
    if len(values) == 1:
        return {"point": point, "lo": point, "hi": point, "n": 1}
    boots = np.empty(n_boot)
    n = len(values)
    for b in range(n_boot):
        sample = values[rng.randint(0, n, n)]
        boots[b] = statistic(sample)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n": int(n)}


# ── timing ────────────────────────────────────────────────────────────────

class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0


# ── environment / manifest helpers ───────────────────────────────────────

def get_git_commit(root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def get_versions():
    versions = {"python": sys.version, "platform": platform.platform()}
    for mod_name in ["pennylane", "torch", "numpy", "sklearn", "scipy", "pandas", "statsmodels"]:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = None
    return versions


def confusion_totals_ok(cm, n_expected):
    return int(np.sum(cm)) == int(n_expected)


def effective_cpu_budget():
    """Real usable CPU count for this process, respecting a cgroup quota if
    present (many rented cloud containers report `nproc`/os.cpu_count() from
    the host while actually being throttled to a much smaller cgroup quota —
    verified on this vast.ai instance: nproc=16 but cpu.max quota ~= 1.92)."""
    import os as _os
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return max(1.0, int(quota) / int(period))
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0:
            return max(1.0, quota / period)
    except Exception:
        pass
    return float(_os.cpu_count() or 1)


def default_max_workers(n_pending=None, headroom=1.5):
    """Worker count matched to the real cgroup CPU budget, with a small
    pipelining multiplier since individual runs are I/O-light but not
    perfectly CPU-bound every instant."""
    budget = effective_cpu_budget()
    workers = max(1, int(round(budget * headroom)))
    if n_pending is not None:
        workers = max(1, min(workers, n_pending))
    return workers
