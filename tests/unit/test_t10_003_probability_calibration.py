"""Unit tests for probability calibration experiment and decision ADR (T10-003).

Acceptance Criteria:
- [x] Calibration is fit on validation only
- [x] Before/after reliability plots saved
- [x] Decision recorded in an ADR
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = REPO_ROOT / "reports" / "calibration"
DOCS_DIR = REPO_ROOT / "docs" / "decisions"


def test_calibration_fit_on_validation_only() -> None:
    """Acceptance Criterion 1: Calibration is fit on validation only."""
    report_path = REPORTS_DIR / "calibration_experiment.json"
    assert report_path.is_file(), f"Missing calibration report: {report_path}"

    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["task_id"] == "T10-003"
    assert data["split_evaluated"] == "validation"

    image_meta = data["image_model"]
    assert image_meta["split_evaluated"] == "validation"
    assert image_meta["sample_count"] > 0
    assert "optimal_temperature" in image_meta
    assert "pre_scaling_ece" in image_meta
    assert "post_scaling_ece" in image_meta

    sensor_meta = data["sensor_model"]
    assert sensor_meta["split_evaluated"] == "validation"
    assert sensor_meta["sample_count"] > 0
    assert "pre_scaling_ece" in sensor_meta
    assert "platt_ece" in sensor_meta
    assert "isotonic_ece" in sensor_meta


def test_before_after_reliability_plots_saved() -> None:
    """Acceptance Criterion 2: Before/after reliability plots saved."""
    plot_path = REPORTS_DIR / "t10_003_reliability_plots.png"
    assert plot_path.is_file(), f"Missing reliability plot at {plot_path}"
    # Verify file is non-empty and has substantial image data (> 50 KB)
    assert plot_path.stat().st_size > 50_000, (
        f"Plot size unexpectedly small: {plot_path.stat().st_size}"
    )


def test_decision_recorded_in_an_adr() -> None:
    """Acceptance Criterion 3: Decision recorded in an ADR."""
    adr_path = DOCS_DIR / "0009-probability-calibration.md"
    assert adr_path.is_file(), f"Missing ADR file at {adr_path}"

    content = adr_path.read_text(encoding="utf-8")
    content_lower = content.lower()

    # Verify header & metadata
    assert "adr 0009" in content_lower
    assert "probability calibration" in content_lower
    assert "t10-003" in content_lower
    assert "status" in content_lower and "accepted" in content_lower

    # Verify decision to retain 'model score'
    assert "model score" in content_lower
    assert "retain" in content_lower

    # Verify models evaluated
    assert "efficientnet" in content_lower
    assert "decisiontree" in content_lower or "decision tree" in content_lower
    assert "temperature scaling" in content_lower
    assert "platt" in content_lower

    # Verify domain justifications
    assert "validation" in content_lower
    assert "inherited" in content_lower
    assert "drift" in content_lower


def test_ui_and_schemas_preserve_model_score_wording() -> None:
    """Verify that UI wording and schemas denote 'model score, not a calibrated probability'."""
    confidence_meter_path = (
        REPO_ROOT / "frontend" / "components" / "ui" / "ConfidenceMeter.tsx"
    )
    if confidence_meter_path.is_file():
        cm_text = confidence_meter_path.read_text(encoding="utf-8").lower()
        assert (
            "score, not a calibrated probability" in cm_text or "model score" in cm_text
        )

    pred_schema_path = REPO_ROOT / "backend" / "app" / "schemas" / "prediction.py"
    if pred_schema_path.is_file():
        schema_text = pred_schema_path.read_text(encoding="utf-8").lower()
        assert "uncalibrated" in schema_text or "heuristic score" in schema_text
