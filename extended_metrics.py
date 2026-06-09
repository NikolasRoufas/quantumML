"""
Extended classification metrics beyond accuracy.
Used for every model to produce publication-quality Table 3.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
)


def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Returns dict with accuracy, F1, precision, recall, MCC,
    and AUC-ROC (if probabilities are supplied).
    """
    m = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc":       float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_proba is not None and len(set(y_true)) > 1:
        m["auc_roc"] = float(roc_auc_score(y_true, y_proba))
    return m


def print_metrics_table(all_metrics):
    header = f"  {'Model':<38} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6} {'MCC':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, m in all_metrics.items():
        auc = f"{m['auc_roc']:.3f}" if "auc_roc" in m else "  N/A"
        print(
            f"  {name:<38} "
            f"{m['accuracy']:.3f}  "
            f"{m['f1']:.3f}  "
            f"{m['precision']:.3f}  "
            f"{m['recall']:.3f}  "
            f"{auc}  "
            f"{m['mcc']:.3f}"
        )
    print("=" * len(header))
