"""Unit tests for Tomato Sensor Experiments (T4-019).

Verifies Acceptance Criteria:
- Month-based assignment is used (89 train, 16 val, 27 test disjoint months).
- Metadata has experimental=true on all trained tomato artifacts.
- Report documents transformed and synthetic features (all 8 proxy formulas).
- No test metrics appear in comparison reports.
- All five model families evaluated with required metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_tomato_experiments import (
    AUTHORIZED_FEATURES,
    MODEL_FAMILIES,
    verify_tomato_split_integrity,
)

ROOT = Path(__file__).resolve().parent.parent.parent
TOMATO_DATA_DIR = ROOT / "processed_data" / "sensor" / "tomato"
REPORTS_DIR = ROOT / "reports" / "tomato"
EXP_JSON_PATH = REPORTS_DIR / "tomato_sensor_experiments.json"
EXP_MD_PATH = REPORTS_DIR / "tomato_sensor_experiments.md"


def test_month_based_assignment_is_used() -> None:
    """Acceptance Criterion 1: Month-based assignment is used with zero leakage."""
    split_info = verify_tomato_split_integrity(TOMATO_DATA_DIR)
    assert split_info["group_separation_verified"] is True
    assert split_info["overlap_months"] == 0
    assert split_info["overlap_sample_ids"] == 0
    assert split_info["train_months"] == 89
    assert split_info["val_months"] == 16
    assert split_info["test_months"] == 27
    assert split_info["total_months"] == 132

    # Check that report JSON and Markdown reflect month-based partitioning
    if EXP_JSON_PATH.is_file():
        with open(EXP_JSON_PATH, encoding="utf-8") as f:
            data = json.load(f)
        assert data["split_integrity"]["train_months"] == 89
        assert data["split_integrity"]["val_months"] == 16
        assert data["split_integrity"]["test_months"] == 27


def test_metadata_has_experimental_true() -> None:
    """Acceptance Criterion 2: Metadata has experimental=true on all tomato models."""
    assert EXP_JSON_PATH.is_file(), f"Missing report at {EXP_JSON_PATH}"

    with open(EXP_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("experimental") is True

    for m in data.get("models", []):
        assert m.get("experimental") is True, (
            f"Model {m.get('model_type')} does not have experimental=True in summary"
        )
        metadata_path = Path(m["metadata_path"])
        if metadata_path.is_file():
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            assert meta.get("experimental") is True, (
                f"Model artifact {m.get('model_type')} has experimental={meta.get('experimental')}, expected True"
            )


def test_report_documents_transformed_and_synthetic_features() -> None:
    """Acceptance Criterion 3: Report documents transformed and synthetic features."""
    assert EXP_JSON_PATH.is_file()
    assert EXP_MD_PATH.is_file()

    with open(EXP_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    md_text = EXP_MD_PATH.read_text(encoding="utf-8")

    # Verify proxy_transforms in JSON
    assert "proxy_transforms" in data
    for feat in AUTHORIZED_FEATURES:
        assert feat in data["proxy_transforms"], (
            f"Feature {feat} missing from proxy_transforms documentation"
        )
        assert feat in md_text, f"Feature {feat} missing from Markdown report table"

    # Verify formulas mentioned
    assert "Ambient_Temp" in md_text
    assert "Humidity + 15.0" in md_text
    assert "Soil_pH - 1.0" in md_text
    assert "EC_mS_cm * 1.4" in md_text
    assert "Nitrogen_Level * 4.0" in md_text


def test_no_test_metrics_appear() -> None:
    """Acceptance Criterion: Zero test metrics appear in comparison report or JSON."""
    assert EXP_JSON_PATH.is_file()
    assert EXP_MD_PATH.is_file()

    with open(EXP_JSON_PATH, encoding="utf-8") as f:
        json_text = f.read().lower()

    md_text = EXP_MD_PATH.read_text(encoding="utf-8").lower()

    for forbidden in [
        "test_macro_f1",
        "test_accuracy",
        "test_precision",
        "test_recall",
    ]:
        assert forbidden not in json_text, f"Forbidden key '{forbidden}' found in JSON"
        assert forbidden not in md_text, (
            f"Forbidden key '{forbidden}' found in Markdown"
        )


def test_all_five_model_families_evaluated() -> None:
    """All 5 model families evaluated with required Section 18.2 validation metrics."""
    assert EXP_JSON_PATH.is_file()

    with open(EXP_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    families = {m["model_type"] for m in data.get("models", [])}
    assert families == set(MODEL_FAMILIES)

    for m in data.get("models", []):
        assert m["val_macro_f1"] > 0.0
        assert m["val_accuracy"] > 0.0
        assert "val_macro_precision" in m
        assert "val_macro_recall" in m
        assert "cv_macro_f1" in m
        assert "cv_std_macro_f1" in m
        assert "best_hyperparameters" in m
