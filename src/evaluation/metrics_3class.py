"""3-Class Evaluation Metrics for SentiMix Hindi-English Sentiment Analysis.

Computes comprehensive multiclass performance metrics:
- Overall Accuracy
- Macro Precision, Macro Recall, Macro F1 (primary official SemEval metric)
- Per-class metrics (Precision, Recall, F1, Support) for Positive (0), Negative (1), Neutral (2)
- 3x3 Confusion Matrix
- Optional secondary binary subset analysis (strictly excluding Neutral)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

LABEL_POSITIVE = 0
LABEL_NEGATIVE = 1
LABEL_NEUTRAL  = 2

CLASS_NAMES: Dict[int, str] = {
    LABEL_POSITIVE: "positive",
    LABEL_NEGATIVE: "negative",
    LABEL_NEUTRAL:  "neutral",
}


def compute_3class_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Dict[str, Any]:
    """Compute primary 3-class empirical evaluation metrics.

    Args:
        y_true: Ground truth labels (0, 1, or 2)
        y_pred: Predicted labels (0, 1, or 2)

    Returns:
        Dictionary containing accuracy, macro metrics, per-class breakdown,
        and 3x3 confusion matrix.
    """
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)

    if len(y_t) == 0:
        raise ValueError("Label arrays cannot be empty.")
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: y_true={len(y_t)}, y_pred={len(y_p)}")

    acc = float(accuracy_score(y_t, y_p))
    macro_prec = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))

    # Per-class metrics
    labels = [LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_NEUTRAL]
    p_per_class = precision_score(y_t, y_p, labels=labels, average=None, zero_division=0)
    r_per_class = recall_score(y_t, y_p, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_t, y_p, labels=labels, average=None, zero_division=0)

    per_class: Dict[str, Dict[str, Any]] = {}
    for idx, name in CLASS_NAMES.items():
        support = int(np.sum(y_t == idx))
        per_class[name] = {
            "label_id": idx,
            "precision": round(float(p_per_class[idx]), 4),
            "recall": round(float(r_per_class[idx]), 4),
            "f1": round(float(f1_per_class[idx]), 4),
            "support": support,
        }

    # 3x3 Confusion Matrix
    cm = confusion_matrix(y_t, y_p, labels=labels)

    return {
        "total_samples": int(len(y_t)),
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": [CLASS_NAMES[i] for i in labels],
        },
    }


def compute_binary_subset_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Dict[str, Any]:
    """Compute secondary binary analysis strictly on the Positive-vs-Negative subset.

    IMPORTANT:
    Neutral instances are explicitly EXCLUDED from both true and predicted sets.
    This analysis is secondary and must NOT be substituted for primary 3-class results.
    """
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)

    # Filter strictly for samples where ground truth is binary (positive=0, negative=1)
    mask = (y_t != LABEL_NEUTRAL) & (y_p != LABEL_NEUTRAL)
    if not np.any(mask):
        return {
            "is_secondary_analysis": True,
            "error": "No non-neutral binary sample pairs available.",
            "total_samples": 0,
        }

    sub_true = y_t[mask]
    sub_pred = y_p[mask]

    acc = float(accuracy_score(sub_true, sub_pred))
    macro_f1 = float(f1_score(sub_true, sub_pred, average="macro", zero_division=0))

    return {
        "is_secondary_analysis": True,
        "description": "Secondary Positive-vs-Negative evaluation excluding Neutral",
        "total_samples": int(len(sub_true)),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
    }
