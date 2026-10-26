"""Unit tests for from-scratch metrics notebook and concept note (T4-004)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "03_metrics_from_scratch.ipynb"
CONCEPT_NOTE_PATH = ROOT / "docs" / "learning" / "confusion-matrix-precisionrecall-macrof1.md"


def confusion_matrix_from_scratch(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    labels: list | None = None,
) -> tuple[np.ndarray, list]:
    """Pure NumPy multi-class confusion matrix implementation."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_pred must have identical length.")

    ordered_labels = list(labels) if labels is not None else sorted(set(y_t).union(set(y_p)))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(ordered_labels)}
    k = len(ordered_labels)
    cm = np.zeros((k, k), dtype=np.int64)

    for t, p in zip(y_t, y_p):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    return cm, ordered_labels


def calculate_metrics_from_scratch(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    labels: list | None = None,
    zero_division: float = 0.0,
) -> dict:
    """Pure NumPy precision, recall, F1, accuracy, and macro metrics."""
    cm, ordered_labels = confusion_matrix_from_scratch(y_true, y_pred, labels=labels)
    k = len(ordered_labels)

    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)
    n = len(y_true)

    prec_denom = tp + fp
    precision = np.divide(tp, prec_denom, out=np.full(k, zero_division), where=prec_denom > 0)

    rec_denom = tp + fn
    recall = np.divide(tp, rec_denom, out=np.full(k, zero_division), where=rec_denom > 0)

    f1_denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, f1_denom, out=np.full(k, zero_division), where=f1_denom > 0)

    accuracy = float(tp.sum()) / float(n) if n > 0 else zero_division
    macro_precision = float(np.mean(precision)) if k > 0 else zero_division
    macro_recall = float(np.mean(recall)) if k > 0 else zero_division
    macro_f1 = float(np.mean(f1)) if k > 0 else zero_division

    per_class = {
        ordered_labels[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(k)
    }

    return {
        "confusion_matrix": cm,
        "labels": ordered_labels,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def test_notebook_file_exists_and_executed() -> None:
    """Verify notebook exists, is non-empty, and has executed code cells."""
    assert NOTEBOOK_PATH.is_file(), f"Missing notebook: {NOTEBOOK_PATH}"
    data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert "cells" in data
    assert len(data["cells"]) > 5

    # Check for executed code cells
    code_cells = [c for c in data["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) >= 3
    for cell in code_cells:
        assert cell.get("execution_count") is not None, "Code cell was not executed!"


def test_concept_note_exists_and_complete() -> None:
    """Verify concept note exists and includes all required sections."""
    assert CONCEPT_NOTE_PATH.is_file(), f"Missing concept note: {CONCEPT_NOTE_PATH}"
    text = CONCEPT_NOTE_PATH.read_text(encoding="utf-8")
    assert "## 1. Intuition" in text
    assert "## 2. Mathematical Formulation" in text
    assert "## 3. From-Scratch Implementation" in text
    assert "## 4. Production & Library Usage" in text
    assert "## 5. Failure Modes, Edge Cases & Debugging" in text
    assert "## 6. Where It Appears in This Project" in text
    assert "## 7. Self-Check Questions" in text
    assert "Sushil" in text
    assert "Yashwant" in text


def test_metrics_match_sklearn_exact() -> None:
    """Verify confusion matrix and metrics match scikit-learn on multi-class example with zero predictions."""
    classes = ["EC_High", "EC_Low", "Multiple_Stress", "Normal", "WaterTemp_High", "pH_Low"]
    y_true = [
        "Normal", "Normal", "Normal", "Normal", "Normal",
        "pH_Low", "pH_Low", "pH_Low",
        "EC_High", "EC_High", "EC_High",
        "WaterTemp_High", "WaterTemp_High",
        "Multiple_Stress", "Multiple_Stress",  # Present in ground truth, never predicted
    ]
    y_pred = [
        "Normal", "Normal", "Normal", "Normal", "EC_High",
        "pH_Low", "pH_Low", "Normal",
        "EC_High", "EC_High", "Normal",
        "WaterTemp_High", "Normal",
        "Normal", "Normal",  # Multiple_Stress has ZERO predictions
    ]

    res = calculate_metrics_from_scratch(y_true, y_pred, labels=classes, zero_division=0.0)

    # Scikit-learn counterparts
    cm_sk = confusion_matrix(y_true, y_pred, labels=classes)
    acc_sk = accuracy_score(y_true, y_pred)
    prec_macro_sk = precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0.0)
    rec_macro_sk = recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0.0)
    f1_macro_sk = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0.0)

    np.testing.assert_array_equal(res["confusion_matrix"], cm_sk)
    np.testing.assert_allclose(res["accuracy"], acc_sk, atol=1e-9)
    np.testing.assert_allclose(res["macro_precision"], prec_macro_sk, atol=1e-9)
    np.testing.assert_allclose(res["macro_recall"], rec_macro_sk, atol=1e-9)
    np.testing.assert_allclose(res["macro_f1"], f1_macro_sk, atol=1e-9)

    # Per class checks
    prec_per_sk = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0.0)
    rec_per_sk = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0.0)
    f1_per_sk = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0.0)

    for i, cls in enumerate(classes):
        np.testing.assert_allclose(res["per_class"][cls]["precision"], prec_per_sk[i], atol=1e-9)
        np.testing.assert_allclose(res["per_class"][cls]["recall"], rec_per_sk[i], atol=1e-9)
        np.testing.assert_allclose(res["per_class"][cls]["f1"], f1_per_sk[i], atol=1e-9)


def test_accuracy_paradox_demonstration() -> None:
    """Verify that overall accuracy masks failure on rare classes."""
    y_true = ["Normal"] * 960 + ["pH_Low"] * 40
    y_pred = ["Normal"] * 1000

    res = calculate_metrics_from_scratch(y_true, y_pred, labels=["Normal", "pH_Low"])

    # Accuracy is high (>90%), but minority recall is 0.0 and Macro-F1 is low (<0.60)
    assert res["accuracy"] == 0.96
    assert res["per_class"]["pH_Low"]["recall"] == 0.0
    assert res["per_class"]["pH_Low"]["precision"] == 0.0
    assert res["per_class"]["pH_Low"]["f1"] == 0.0
    assert res["macro_f1"] < 0.50
