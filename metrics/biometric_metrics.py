import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)

def compute_basic_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }

def compute_eer(y_true, y_score):
    y_true = np.array(y_true)

    if len(np.unique(y_true)) < 2:
        return {
            "eer": None,
            "far": None,
            "frr": None
        }

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2

    return {
        "eer": eer,
        "far": fpr[idx],
        "frr": fnr[idx]
    }