"""Unit tests for ml.evaluation.metrics adhering to Overview Section 18.2 (T4-005)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ml.evaluation import (
    calculate_classification_metrics,
    calculate_metrics,
    to_serializable_metrics,
)
from ml.labels import LETTUCE_SENSOR_CLASSES, SPINACH_IMAGE_CLASSES


def test_all_section_18_2_fields_present() -> None:
    """Acceptance Criterion 1: All Overview Section 18.2 metric fields present."""
    labels = list(LETTUCE_SENSOR_CLASSES)
    y_true = ["Normal", "EC_High", "pH_Low", "WaterTemp_High", "Normal"]
    y_pred = ["Normal", "EC_High", "Normal", "WaterTemp_High", "Normal"]

    metrics = calculate_metrics(y_true, y_pred, labels=labels)

    # Mandatory Section 18.2 fields
    required_fields = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "per_class",
        "confusion_matrix",
        "confusion_matrix_list",
        "labels",
        "class_order",
        "total_samples",
    ]
    for field in required_fields:
        assert field in metrics, f"Missing required metric field: {field}"

    # Per-class mandatory fields
    for lbl in labels:
        assert lbl in metrics["per_class"], f"Missing class in per_class metrics: {lbl}"
        class_entry = metrics["per_class"][lbl]
        assert "precision" in class_entry
        assert "recall" in class_entry
        assert "f1" in class_entry
        assert "support" in class_entry
        assert isinstance(class_entry["precision"], float)
        assert isinstance(class_entry["recall"], float)
        assert isinstance(class_entry["f1"], float)
        assert isinstance(class_entry["support"], int)

    # Check confusion matrix dimensions
    assert isinstance(metrics["confusion_matrix"], np.ndarray)
    assert metrics["confusion_matrix"].shape == (len(labels), len(labels))
    assert metrics["total_samples"] == len(y_true)


def test_class_order_is_fixed_and_explicit() -> None:
    """Acceptance Criterion 2: Class order is fixed, explicit, and governs confusion matrix."""
    labels_a = ["Normal", "EC_High", "pH_Low"]
    labels_b = ["pH_Low", "Normal", "EC_High"]

    y_true = ["Normal", "EC_High", "pH_Low"]
    y_pred = ["Normal", "Normal", "pH_Low"]

    metrics_a = calculate_metrics(y_true, y_pred, labels=labels_a)
    metrics_b = calculate_metrics(y_true, y_pred, labels=labels_b)

    assert metrics_a["labels"] == labels_a
    assert metrics_a["class_order"] == labels_a
    assert list(metrics_a["per_class"].keys()) == labels_a

    assert metrics_b["labels"] == labels_b
    assert metrics_b["class_order"] == labels_b
    assert list(metrics_b["per_class"].keys()) == labels_b

    # Confusion matrix indices map to explicit order
    # In labels_a: Normal is index 0. Normal true -> Normal pred is cm[0, 0] = 1
    # EC_High true -> Normal pred is cm[1, 0] = 1
    cm_a = metrics_a["confusion_matrix"]
    assert cm_a[0, 0] == 1  # Normal -> Normal
    assert cm_a[1, 0] == 1  # EC_High -> Normal

    # In labels_b: Normal is index 1. Normal true -> Normal pred is cm[1, 1] = 1
    # EC_High is index 2. EC_High true -> Normal pred is cm[2, 1] = 1
    cm_b = metrics_b["confusion_matrix"]
    assert cm_b[1, 1] == 1
    assert cm_b[2, 1] == 1


def test_unknown_label_raises_error() -> None:
    """Acceptance Criterion 3: Unknown label raises a clear error."""
    labels = ["Healthy", "Deficient"]

    # Unknown label in y_true
    with pytest.raises(ValueError, match="Unknown label.*Pest_Infestation"):
        calculate_metrics(["Healthy", "Pest_Infestation"], ["Healthy", "Deficient"], labels=labels)

    # Unknown label in y_pred
    with pytest.raises(ValueError, match="Unknown label.*Alien_Class"):
        calculate_metrics(["Healthy", "Deficient"], ["Healthy", "Alien_Class"], labels=labels)


def test_input_validation_errors() -> None:
    """Verify robust input validation on malformed inputs."""
    # Empty labels list
    with pytest.raises(ValueError, match="labels list must be provided, non-empty"):
        calculate_metrics(["A"], ["A"], labels=[])

    # Duplicate labels
    with pytest.raises(ValueError, match="Duplicate labels"):
        calculate_metrics(["A"], ["A"], labels=["A", "B", "A"])

    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_metrics(["A", "B"], ["A"], labels=["A", "B"])

    # Empty inputs
    with pytest.raises(ValueError, match="Cannot calculate metrics on empty input"):
        calculate_metrics([], [], labels=["A", "B"])


def test_exact_scikit_learn_parity_and_edge_cases() -> None:
    """Verify numerical parity with scikit-learn including zero-support and zero-prediction classes."""
    classes = list(SPINACH_IMAGE_CLASSES)
    y_true = [
        "Spinach_Healthy", "Spinach_Healthy", "Spinach_Healthy",
        "Spinach_downy mildew", "Spinach_downy mildew",
        "Spinach_anthracnose",
        # Notice: Spinach_pest damage has 0 ground truth samples
    ]
    y_pred = [
        "Spinach_Healthy", "Spinach_downy mildew", "Spinach_Healthy",
        "Spinach_downy mildew", "Spinach_downy mildew",
        "Spinach_Healthy",  # Spinach_anthracnose predicted as Spinach_Healthy
        # Notice: Spinach_Non fresh and Spinach_bacterial spot have 0 predictions
    ]

    res = calculate_metrics(y_true, y_pred, labels=classes, zero_division=0.0)

    # Direct scikit-learn calculations
    cm_sk = confusion_matrix(y_true, y_pred, labels=classes)
    acc_sk = accuracy_score(y_true, y_pred)
    prec_macro_sk = precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0.0)
    rec_macro_sk = recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0.0)
    f1_macro_sk = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0.0)

    np.testing.assert_array_equal(res["confusion_matrix"], cm_sk)
    assert res["accuracy"] == pytest.approx(acc_sk)
    assert res["macro_precision"] == pytest.approx(prec_macro_sk)
    assert res["macro_recall"] == pytest.approx(rec_macro_sk)
    assert res["macro_f1"] == pytest.approx(f1_macro_sk)


def test_json_serializability() -> None:
    """Verify to_serializable_metrics produces valid JSON with pure Python primitives."""
    labels = ["A", "B"]
    y_true = ["A", "B", "A"]
    y_pred = ["A", "A", "A"]

    metrics = calculate_classification_metrics(y_true, y_pred, labels=labels)

    # Without serialization helper, numpy ndarray fails standard json.dumps
    with pytest.raises(TypeError):
        json.dumps(metrics)

    # With helper, it serializes flawlessly
    clean = to_serializable_metrics(metrics)
    dumped = json.dumps(clean, indent=2)
    assert isinstance(dumped, str)
    loaded = json.loads(dumped)
    assert loaded["confusion_matrix"] == [[2, 0], [1, 0]]
    assert loaded["accuracy"] == pytest.approx(2 / 3)
