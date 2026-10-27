"""Unit tests for sensor model selection by validation macro-F1 (T4-023).

Acceptance Criteria:
1. Script fails if test metrics are supplied.
2. Selection JSON lists runners-up.
3. Per-class weaknesses are noted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.select_model import (
    analyze_per_class_weaknesses,
    run_selection,
    select_best_model_for_crop,
    validate_report_is_validation_only,
)

ROOT = Path(__file__).resolve().parent.parent.parent


# ==============================================================================
# Criterion 1: Script fails if test metrics are supplied
# ==============================================================================


def test_fails_when_test_split_report_supplied(tmp_path: Path) -> None:
    """Criterion 1: Reject any report indicating split_evaluated='test'."""
    bad_report = {
        "crop": "lettuce",
        "model_name": "test_model_v1.0.0",
        "model_family": "RandomForestClassifier",
        "split_evaluated": "test",
        "metrics": {"macro_f1": 0.85, "accuracy": 0.90},
    }
    report_file = tmp_path / "lettuce_sensor_rf_v1.0.0_test.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(bad_report, f)

    with pytest.raises(ValueError, match="Criterion 1 Violation"):
        validate_report_is_validation_only(bad_report, report_file)


def test_fails_when_filename_indicates_test_report(tmp_path: Path) -> None:
    """Criterion 1: Reject files whose filename contains '_test.json'."""
    report = {
        "crop": "lettuce",
        "model_name": "test_model_v1.0.0",
        "model_family": "RandomForestClassifier",
        "split_evaluated": "validation",
        "metrics": {"macro_f1": 0.85, "accuracy": 0.90},
    }
    report_file = tmp_path / "lettuce_sensor_rf_v1.0.0_test.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f)

    with pytest.raises(ValueError, match="Criterion 1 Violation: Test split report"):
        validate_report_is_validation_only(report, report_file)


def test_fails_when_forbidden_test_metric_keys_present(tmp_path: Path) -> None:
    """Criterion 1: Reject reports containing test metric keys in top-level or metrics dict."""
    bad_report_top = {
        "crop": "lettuce",
        "model_name": "test_model_v1.0.0",
        "split_evaluated": "validation",
        "test_macro_f1": 0.82,
        "metrics": {"macro_f1": 0.85},
    }
    f1 = tmp_path / "rep1_validation.json"
    with pytest.raises(ValueError, match="Forbidden test metric keys"):
        validate_report_is_validation_only(bad_report_top, f1)

    bad_report_sub = {
        "crop": "lettuce",
        "model_name": "test_model_v1.0.0",
        "split_evaluated": "validation",
        "metrics": {"macro_f1": 0.85, "test_metrics": {"loss": 0.1}},
    }
    f2 = tmp_path / "rep2_validation.json"
    with pytest.raises(ValueError, match="Forbidden test metric keys"):
        validate_report_is_validation_only(bad_report_sub, f2)


# ==============================================================================
# Criterion 2: Selection JSON lists runners-up
# ==============================================================================


def _create_mock_validation_report(
    path: Path,
    crop: str,
    model_name: str,
    family: str,
    macro_f1: float,
    macro_recall: float,
    accuracy: float,
    per_class: dict[str, Any] | None = None,
) -> None:
    payload = {
        "crop": crop,
        "model_name": model_name,
        "model_family": family,
        "split_evaluated": "validation",
        "artifact_version": "1.0.0",
        "device_used": "cpu",
        "hyperparameters": {"param": 1},
        "metrics": {
            "macro_f1": macro_f1,
            "macro_recall": macro_recall,
            "accuracy": accuracy,
            "macro_precision": accuracy,
            "per_class": per_class
            or {
                "Normal": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 50},
                "EC_High": {
                    "precision": 0.70,
                    "recall": 0.60,
                    "f1": 0.65,
                    "support": 10,
                },
            },
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def test_selection_json_lists_runners_up_and_ranks(tmp_path: Path) -> None:
    """Criterion 2: Selection chooses highest macro-F1 and documents ranked runners-up with deltas."""
    _create_mock_validation_report(
        tmp_path / "lettuce_sensor_rf_v1.0.0_validation.json",
        crop="lettuce",
        model_name="lettuce_sensor_rf_v1.0.0",
        family="RandomForestClassifier",
        macro_f1=0.85,
        macro_recall=0.86,
        accuracy=0.92,
    )
    _create_mock_validation_report(
        tmp_path / "lettuce_sensor_dt_v1.0.0_validation.json",
        crop="lettuce",
        model_name="lettuce_sensor_dt_v1.0.0",
        family="DecisionTreeClassifier",
        macro_f1=0.80,
        macro_recall=0.81,
        accuracy=0.88,
    )
    _create_mock_validation_report(
        tmp_path / "lettuce_sensor_knn_v1.0.0_validation.json",
        crop="lettuce",
        model_name="lettuce_sensor_knn_v1.0.0",
        family="KNeighborsClassifier",
        macro_f1=0.72,
        macro_recall=0.73,
        accuracy=0.79,
    )

    result = select_best_model_for_crop("lettuce", reports_dir=tmp_path)

    # Selected model must be RF (highest macro-F1: 0.85)
    assert result["selected_model"]["model_name"] == "lettuce_sensor_rf_v1.0.0"
    assert result["selected_model"]["metrics"]["macro_f1"] == 0.85

    # Runners up must be listed in rank order (DT rank 2, KNN rank 3)
    runners_up = result["runners_up"]
    assert len(runners_up) == 2

    r2 = runners_up[0]
    assert r2["rank"] == 2
    assert r2["model_name"] == "lettuce_sensor_dt_v1.0.0"
    assert r2["val_macro_f1"] == 0.80
    assert r2["macro_f1_delta_from_selected"] == -0.05

    r3 = runners_up[1]
    assert r3["rank"] == 3
    assert r3["model_name"] == "lettuce_sensor_knn_v1.0.0"
    assert r3["val_macro_f1"] == 0.72
    assert r3["macro_f1_delta_from_selected"] == -0.13


def test_tie_break_logic_applied(tmp_path: Path) -> None:
    """Criterion 2: When macro-F1 is tied, secondary metrics and parsimony break the tie."""
    # DT and RF tied at 0.90 macro-F1, identical recall & accuracy -> DecisionTree wins on parsimony
    _create_mock_validation_report(
        tmp_path / "lettuce_sensor_dt_v1.0.0_validation.json",
        crop="lettuce",
        model_name="lettuce_sensor_dt_v1.0.0",
        family="DecisionTreeClassifier",
        macro_f1=0.90,
        macro_recall=0.90,
        accuracy=0.95,
    )
    _create_mock_validation_report(
        tmp_path / "lettuce_sensor_rf_v1.0.0_validation.json",
        crop="lettuce",
        model_name="lettuce_sensor_rf_v1.0.0",
        family="RandomForestClassifier",
        macro_f1=0.90,
        macro_recall=0.90,
        accuracy=0.95,
    )

    result = select_best_model_for_crop("lettuce", reports_dir=tmp_path)
    assert result["selected_model"]["model_name"] == "lettuce_sensor_dt_v1.0.0"
    assert result["tie_break_applied"] is True
    assert "tied at 0.9000" in result["tie_break_notes"]


# ==============================================================================
# Criterion 3: Per-class weaknesses are noted
# ==============================================================================


def test_per_class_weaknesses_captured() -> None:
    """Criterion 3: Identify classes with zero support, low recall, low precision, and assign hazards."""
    mock_metrics = {
        "per_class": {
            "Normal": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 50},
            "EC_High": {"precision": 0.65, "recall": 0.95, "f1": 0.77, "support": 20},
            "pH_Low": {"precision": 0.90, "recall": 0.50, "f1": 0.64, "support": 15},
            "WaterTemp_Low": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
        }
    }

    weaknesses = analyze_per_class_weaknesses(mock_metrics, crop="lettuce")

    assert "WaterTemp_Low" in weaknesses["unsupported_classes_in_val"]

    low_recall = weaknesses["low_recall_classes"]
    assert any(item["class"] == "pH_Low" for item in low_recall)
    ph_low_entry = next(item for item in low_recall if item["class"] == "pH_Low")
    assert (
        "lockout" in ph_low_entry["hazard"].lower()
        or "acid" in ph_low_entry["hazard"].lower()
    )

    low_prec = weaknesses["low_precision_classes"]
    assert any(item["class"] == "EC_High" for item in low_prec)

    notes = weaknesses["diagnostic_notes"]
    assert len(notes) >= 3


# ==============================================================================
# End-to-End Verification on Real Repository Reports
# ==============================================================================


@pytest.mark.parametrize("crop", ["lettuce", "spinach", "tomato"])
def test_end_to_end_selection_real_crops(crop: str, tmp_path: Path) -> None:
    """Verify that selection executes smoothly on active repository reports for all 3 crops."""
    out_file = tmp_path / f"{crop}_selection.json"
    res_path = run_selection(crop=crop, output_path=out_file)

    assert res_path.exists()
    with open(res_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["crop"] == crop
    assert "selected_model" in data
    assert "runners_up" in data
    assert len(data["runners_up"]) == 4  # 5 models trained per crop, 4 runners up
    assert "per_class_weaknesses" in data
    assert data["selection_metric"] == "validation_macro_f1"
