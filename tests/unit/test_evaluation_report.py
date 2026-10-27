"""Unit tests for ml.evaluation.report writer adhering to Overview Section 18.2 (T4-006)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.evaluation import (
    calculate_metrics,
    load_evaluation_report,
    write_evaluation_report,
)
from ml.labels import LETTUCE_SENSOR_CLASSES


@pytest.fixture
def sample_metrics() -> dict:
    """Fixture providing valid metrics dictionary from calculate_metrics."""
    labels = list(LETTUCE_SENSOR_CLASSES)
    y_true = ["Normal", "EC_High", "pH_Low", "WaterTemp_High", "Normal"]
    y_pred = ["Normal", "EC_High", "Normal", "WaterTemp_High", "Normal"]
    return calculate_metrics(y_true, y_pred, labels=labels)


def test_every_overview_18_2_item_has_a_field(tmp_path: Path, sample_metrics: dict) -> None:
    """Acceptance Criterion 1: Every Overview Section 18.2 item has a field."""
    json_path, md_path = write_evaluation_report(
        crop="lettuce",
        model_name="random_forest_baseline",
        split_evaluated="validation",
        metrics=sample_metrics,
        dataset_manifest="processed_data/manifests/lettuce_sensor.json",
        split_protocol="Preserved calendar day holdout (day % 4 == 0) across experiments",
        split_counts={"train": 1680, "validation": 200, "test": 162},
        class_counts_per_split={
            "train": {"Normal": 240, "EC_High": 240},
            "validation": {"Normal": 87, "EC_High": 9},
            "test": {"Normal": 87, "EC_High": 9},
        },
        label_generation_method="Threshold rule-based generation mapped from physical greenhouse bounds",
        model_family="RandomForestClassifier",
        hyperparameters={"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 1},
        limitations=[
            "Labels are rule-derived from thresholds; model learns rules rather than independent biology.",
            "Synthetic oversampling applied to minority classes in training partition.",
            "Intra-day readings from the same tank can be correlated.",
        ],
        seeds={"random_state": 42},
        device_used="cpu",
        artifact_version="1.0.0",
        preprocessing_version="1.0.0",
        output_dir=tmp_path,
    )

    assert json_path.is_file()
    assert md_path.is_file()

    report = load_evaluation_report(json_path)

    # Overview §18.2 required items:
    # 1. accuracy, macro precision, macro recall, macro-F1
    assert "accuracy" in report["metrics"]
    assert "macro_precision" in report["metrics"]
    assert "macro_recall" in report["metrics"]
    assert "macro_f1" in report["metrics"]

    # 2. per-class precision, recall, F1, and support
    assert "per_class" in report["metrics"]
    for cls_name in report["class_order"]:
        entry = report["metrics"]["per_class"][cls_name]
        assert "precision" in entry
        assert "recall" in entry
        assert "f1" in entry
        assert "support" in entry

    # 3. confusion matrix
    assert "confusion_matrix" in report["metrics"] or "confusion_matrix_list" in report["metrics"]

    # 4. train/validation/test counts
    assert report["split_counts"]["train"] == 1680
    assert report["split_counts"]["validation"] == 200
    assert report["split_counts"]["test"] == 162

    # 5. class counts per split
    assert "class_counts_per_split" in report
    assert "train" in report["class_counts_per_split"]
    assert "validation" in report["class_counts_per_split"]
    assert "test" in report["class_counts_per_split"]

    # 6. split/grouping protocol
    assert "split_protocol" in report
    assert "calendar day" in report["split_protocol"]

    # 7. source dataset/version and preprocessing version
    assert report["source_dataset_manifest"] == "processed_data/manifests/lettuce_sensor.json"
    assert report["preprocessing_version"] == "1.0.0"

    # 8. label-generation method
    assert "Threshold rule-based" in report["label_generation_method"]

    # 9. model family, hyperparameters, artifact version
    assert report["model_family"] == "RandomForestClassifier"
    assert report["hyperparameters"]["n_estimators"] == 200
    assert report["artifact_version"] == "1.0.0"

    # 10. evaluation date and software/runtime versions
    assert "evaluation_date" in report
    assert "software_versions" in report
    assert "python" in report["software_versions"]
    assert "scikit-learn" in report["software_versions"]
    assert "numpy" in report["software_versions"]


def test_validation_and_test_metrics_are_never_mixed(tmp_path: Path, sample_metrics: dict) -> None:
    """Acceptance Criterion 2: Validation and test metrics are never mixed or aggregated."""
    # Attempting to use an invalid, combined, or ambiguous split name raises an error
    with pytest.raises(ValueError, match="Invalid split_evaluated 'combined'"):
        write_evaluation_report(
            crop="lettuce",
            model_name="rf",
            split_evaluated="combined",
            metrics=sample_metrics,
            dataset_manifest="dummy_manifest",
            split_protocol="dummy",
            split_counts={"train": 100, "validation": 20, "test": 20},
            class_counts_per_split={"train": {}, "validation": {}, "test": {}},
            label_generation_method="dummy",
            model_family="RF",
            hyperparameters={},
            limitations=["Rule-derived targets"],
            output_dir=tmp_path,
        )

    # Valid validation run
    json_val, md_val = write_evaluation_report(
        crop="lettuce",
        model_name="rf",
        split_evaluated="validation",
        metrics=sample_metrics,
        dataset_manifest="dummy_manifest",
        split_protocol="dummy",
        split_counts={"train": 100, "validation": 20, "test": 20},
        class_counts_per_split={"train": {}, "validation": {}, "test": {}},
        label_generation_method="dummy",
        model_family="RF",
        hyperparameters={},
        limitations=["Rule-derived targets"],
        output_dir=tmp_path,
    )

    # Valid test run
    json_test, md_test = write_evaluation_report(
        crop="lettuce",
        model_name="rf",
        split_evaluated="test",
        metrics=sample_metrics,
        dataset_manifest="dummy_manifest",
        split_protocol="dummy",
        split_counts={"train": 100, "validation": 20, "test": 20},
        class_counts_per_split={"train": {}, "validation": {}, "test": {}},
        label_generation_method="dummy",
        model_family="RF",
        hyperparameters={},
        limitations=["Rule-derived targets"],
        output_dir=tmp_path,
    )

    # Files must be strictly segregated by split name
    assert json_val.name == "rf_validation.json"
    assert md_val.name == "rf_validation.md"
    assert json_test.name == "rf_test.json"
    assert md_test.name == "rf_test.md"

    # Contents must declare explicit single split
    val_data = json.loads(json_val.read_text(encoding="utf-8"))
    test_data = json.loads(json_test.read_text(encoding="utf-8"))
    assert val_data["split_evaluated"] == "validation"
    assert test_data["split_evaluated"] == "test"


def test_limitations_block_is_always_included(tmp_path: Path, sample_metrics: dict) -> None:
    """Acceptance Criterion 3: Limitations block is always included and mandatory."""
    # Empty limitations list raises ValueError
    with pytest.raises(ValueError, match="non-empty 'limitations' sequence is mandatory"):
        write_evaluation_report(
            crop="lettuce",
            model_name="rf",
            split_evaluated="validation",
            metrics=sample_metrics,
            dataset_manifest="dummy_manifest",
            split_protocol="dummy",
            split_counts={"train": 100, "validation": 20, "test": 20},
            class_counts_per_split={"train": {}, "validation": {}, "test": {}},
            label_generation_method="dummy",
            model_family="RF",
            hyperparameters={},
            limitations=[],
            output_dir=tmp_path,
        )

    # Whitespace-only limitations raise ValueError
    with pytest.raises(ValueError, match="non-empty 'limitations' sequence is mandatory"):
        write_evaluation_report(
            crop="lettuce",
            model_name="rf",
            split_evaluated="validation",
            metrics=sample_metrics,
            dataset_manifest="dummy_manifest",
            split_protocol="dummy",
            split_counts={"train": 100, "validation": 20, "test": 20},
            class_counts_per_split={"train": {}, "validation": {}, "test": {}},
            label_generation_method="dummy",
            model_family="RF",
            hyperparameters={},
            limitations=["   ", ""],
            output_dir=tmp_path,
        )

    # Valid limitations are rendered in both JSON and Markdown
    json_path, md_path = write_evaluation_report(
        crop="lettuce",
        model_name="rf",
        split_evaluated="validation",
        metrics=sample_metrics,
        dataset_manifest="dummy_manifest",
        split_protocol="dummy",
        split_counts={"train": 100, "validation": 20, "test": 20},
        class_counts_per_split={"train": {}, "validation": {}, "test": {}},
        label_generation_method="dummy",
        model_family="RF",
        hyperparameters={},
        limitations=["Rule-based synthetic labels", "Oversampled training split"],
        output_dir=tmp_path,
    )

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "limitations" in data
    assert len(data["limitations"]) == 2
    assert "Rule-based synthetic labels" in data["limitations"]

    md_text = md_path.read_text(encoding="utf-8")
    assert "## 7. Mandatory Limitations & Disclosures" in md_text
    assert "- Rule-based synthetic labels" in md_text
    assert "- Oversampled training split" in md_text
