"""Model evaluation metrics calculation adhering to Overview Section 18.2 (T4-005).

Provides scikit-learn backed metric computation for classification models:
- Accuracy, macro-precision, macro-recall, macro-F1.
- Per-class precision, recall, F1, and support with fixed, explicit class order.
- Multi-class confusion matrix.
- Strict input validation rejecting unknown labels not in the model's class list.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def calculate_metrics(
    y_true: Sequence[str | int] | np.ndarray,
    y_pred: Sequence[str | int] | np.ndarray,
    labels: Sequence[str | int],
    zero_division: float = 0.0,
) -> dict[str, Any]:
    """Calculate multi-class classification metrics per Overview Section 18.2.

    Parameters
    ----------
    y_true:
        Ground truth target values.
    y_pred:
        Estimated targets returned by a classifier.
    labels:
        Fixed and explicit ordered list of class labels defining the model's taxonomy.
        Every label in y_true and y_pred must belong to this sequence.
    zero_division:
        Value to return when there is a zero division in precision/recall/F1 (default 0.0).

    Returns
    -------
    dict[str, Any]:
        Dictionary with all Section 18.2 required metric fields:
        - "accuracy": float
        - "macro_precision": float
        - "macro_recall": float
        - "macro_f1": float
        - "per_class": dict mapping class label -> dict with precision, recall, f1, support
        - "confusion_matrix": np.ndarray (k x k)
        - "confusion_matrix_list": list[list[int]]
        - "labels": list of labels in explicit order
        - "class_order": list of labels in explicit order
        - "total_samples": int

    Raises
    ------
    ValueError:
        If y_true and y_pred have different lengths, if either is empty,
        if labels is empty or has duplicates, or if unknown labels exist.
    """
    if labels is None or len(labels) == 0:
        raise ValueError("labels list must be provided, non-empty, and explicit.")

    ordered_labels = list(labels)
    if len(ordered_labels) != len(set(ordered_labels)):
        raise ValueError(f"Duplicate labels detected in class list: {ordered_labels}")

    # Standardize to flat sequences/arrays
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_t)}) and y_pred ({len(y_p)}) must have identical length."
        )

    if len(y_t) == 0:
        raise ValueError("Cannot calculate metrics on empty input (0 samples).")

    # Strict taxonomy check: reject inputs with unknown labels
    allowed_set = set(ordered_labels)
    unknown_true = set(y_t) - allowed_set
    unknown_pred = set(y_p) - allowed_set
    if unknown_true or unknown_pred:
        unknown_all = sorted(str(x) for x in (unknown_true | unknown_pred))
        raise ValueError(
            f"Unknown label(s) detected not in the model's class list {ordered_labels}: {unknown_all}"
        )

    # Compute metrics using scikit-learn
    acc = float(accuracy_score(y_t, y_p))
    cm = confusion_matrix(y_t, y_p, labels=ordered_labels)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_t,
        y_p,
        labels=ordered_labels,
        average="macro",
        zero_division=zero_division,
    )

    prec_per, rec_per, f1_per, sup_per = precision_recall_fscore_support(
        y_t,
        y_p,
        labels=ordered_labels,
        average=None,
        zero_division=zero_division,
    )

    per_class: dict[str, dict[str, float | int]] = {
        str(lbl): {
            "precision": float(prec_per[i]),
            "recall": float(rec_per[i]),
            "f1": float(f1_per[i]),
            "support": int(sup_per[i]),
        }
        for i, lbl in enumerate(ordered_labels)
    }

    return {
        "accuracy": acc,
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_list": cm.tolist(),
        "labels": [str(x) for x in ordered_labels],
        "class_order": [str(x) for x in ordered_labels],
        "total_samples": len(y_t),
    }


def to_serializable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert metrics dictionary to pure JSON-serializable types.

    Converts any np.ndarray or np.number types into standard Python lists, floats, ints.
    """
    serializable = {}
    for key, val in metrics.items():
        if isinstance(val, np.ndarray):
            serializable[key] = val.tolist()
        elif isinstance(val, (np.floating, float)):
            serializable[key] = float(val)
        elif isinstance(val, (np.integer, int)):
            serializable[key] = int(val)
        elif isinstance(val, dict):
            serializable[key] = {
                k: {
                    sub_k: (
                        int(sub_v)
                        if sub_k == "support"
                        else float(sub_v)
                    )
                    for sub_k, sub_v in v.items()
                }
                if isinstance(v, dict)
                else v
                for k, v in val.items()
            }
        else:
            serializable[key] = val
    return serializable


# Alias for explicit naming
calculate_classification_metrics = calculate_metrics
