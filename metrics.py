"""
Extended metrics for the revision package. Wraps/extends the original
`extended_metrics.compute_metrics` (accuracy, F1, precision, recall, MCC,
AUC-ROC) with PR-AUC, balanced accuracy, and the confusion matrix, which the
revision protocol additionally requires.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
)


def compute_full_metrics(y_true, y_pred, y_proba=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else float("nan"),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    m["confusion_matrix"] = cm.tolist()
    m["confusion_matrix_total"] = int(cm.sum())
    m["n_test"] = int(len(y_true))

    if y_proba is not None and len(set(y_true)) > 1:
        y_proba = np.asarray(y_proba, dtype=float)
        m["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        m["pr_auc"] = float(average_precision_score(y_true, y_proba))
    else:
        m["roc_auc"] = float("nan")
        m["pr_auc"] = float("nan")
    return m


def sanity_check_metrics(y_true, y_pred, y_proba, metrics_dict, n_expected):
    """Returns list of issue strings (empty if all checks pass)."""
    issues = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = np.array(metrics_dict["confusion_matrix"])
    if int(cm.sum()) != int(n_expected):
        issues.append(f"confusion matrix total {int(cm.sum())} != n_expected {n_expected}")

    for key in ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]:
        v = metrics_dict.get(key)
        if v is None or not np.isfinite(v) or not (0.0 - 1e-9 <= v <= 1.0 + 1e-9):
            issues.append(f"{key}={v} out of [0,1] or non-finite")

    mcc = metrics_dict.get("mcc")
    if mcc is not None and np.isfinite(mcc) and not (-1.0 - 1e-9 <= mcc <= 1.0 + 1e-9):
        issues.append(f"mcc={mcc} out of [-1,1]")

    if y_proba is not None:
        yp = np.asarray(y_proba, dtype=float)
        if not np.all(np.isfinite(yp)):
            issues.append("y_proba contains non-finite values")
        thresholded = (yp > 0.5).astype(int)
        if not np.array_equal(thresholded, y_pred):
            issues.append("y_pred does not match thresholding y_proba at 0.5 (prediction rule mismatch)")

    if metrics_dict.get("f1", 0) == 0.0:
        # F1=0 is only valid if there are no true positives predicted
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        if tp != 0:
            issues.append(f"F1=0 but true positives={tp} (inconsistent)")

    return issues
