"""Unit tests for Spinach Sensor Experiments (T4-018).

Verifies Acceptance Criteria:
- Report lists excluded fields (tds_ppm, label_id, humidity_source, sample_id, source_timestamp_EDT, bench).
- Group separation test passes (6-hour temporal blocks disjoint, 160/40 per-class counts preserved).
- Limitations block states mixed real/synthetic features.
- All five model families evaluated with validation metrics isolated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.labels import SENSOR_CLASSES
from ml.sensor_models.schemas import SPINACH_ORDERED_FEATURE_NAMES
from scripts.run_spinach_experiments import (
    AUTHORIZED_FEATURES,
    MODEL_FAMILIES,
    compute_spinach_blocks,
    verify_spinach_split_integrity,
)

ROOT = Path(__file__).resolve().parent.parent.parent
SPINACH_DATA_DIR = ROOT / "processed_data" / "sensor" / "spinach"
REPORTS_DIR = ROOT / "reports" / "spinach"
EXP_JSON_PATH = REPORTS_DIR / "spinach_sensor_experiments.json"
EXP_MD_PATH = REPORTS_DIR / "spinach_sensor_experiments.md"

EXPECTED_EXCLUDED_FIELDS = {
    "tds_ppm",
    "label_id",
    "humidity_source",
    "sample_id",
    "source_timestamp_EDT",
    "bench",
}


def test_report_lists_excluded_fields() -> None:
    """Acceptance Criterion 1: Report lists excluded fields."""
    assert EXP_MD_PATH.is_file(), f"Missing report at {EXP_MD_PATH}"
    assert EXP_JSON_PATH.is_file(), f"Missing JSON report at {EXP_JSON_PATH}"

    md_content = EXP_MD_PATH.read_text(encoding="utf-8")
    with open(EXP_JSON_PATH, encoding="utf-8") as f:
        json_data = json.load(f)

    # Check all 6 excluded fields in markdown and json
    for field in EXPECTED_EXCLUDED_FIELDS:
        assert f"`{field}`" in md_content or field in md_content, (
            f"Excluded field '{field}' not mentioned in Markdown report"
        )
        assert field in json_data["excluded_fields"], (
            f"Excluded field '{field}' missing from JSON report excluded_fields"
        )

    # Check authorized features
    assert json_data["authorized_features"] == list(SPINACH_ORDERED_FEATURE_NAMES)
    for feat in AUTHORIZED_FEATURES:
        assert feat not in EXPECTED_EXCLUDED_FIELDS


def test_spinach_group_separation_and_split_counts() -> None:
    """Acceptance Criterion 2: Group separation test passes.

    - 6-hour temporal block separation (bench | date | hour//6).
    - Zero block or sample ID overlap across train, val, and test splits.
    - Preserves 160/40 per-class split (160 train+val, 40 test per class).
    """
    split_info = verify_spinach_split_integrity(SPINACH_DATA_DIR)
    assert split_info["group_separation_verified"] is True
    assert split_info["overlap_blocks"] == 0
    assert split_info["overlap_sample_ids"] == 0
    assert split_info["per_class_train_val"] == 160
    assert split_info["per_class_test"] == 40
    assert split_info["total_rows"] == 1400

    train_df = pd.read_csv(SPINACH_DATA_DIR / "train.csv")
    val_df = pd.read_csv(SPINACH_DATA_DIR / "val.csv")
    test_df = pd.read_csv(SPINACH_DATA_DIR / "test.csv")

    train_blocks = set(compute_spinach_blocks(train_df))
    val_blocks = set(compute_spinach_blocks(val_df))
    test_blocks = set(compute_spinach_blocks(test_df))

    # Strict disjointness
    assert len(train_blocks & val_blocks) == 0
    assert len(train_blocks & test_blocks) == 0
    assert len(val_blocks & test_blocks) == 0

    # Sample ID disjointness
    assert len(set(train_df["sample_id"]) & set(val_df["sample_id"])) == 0
    assert len(set(train_df["sample_id"]) & set(test_df["sample_id"])) == 0
    assert len(set(val_df["sample_id"]) & set(test_df["sample_id"])) == 0

    # Stratified 160/40 per class counts
    spinach_classes = SENSOR_CLASSES["spinach"]
    train_val = pd.concat([train_df, val_df])
    for cls in spinach_classes:
        assert (train_val["label"] == cls).sum() == 160
        assert (test_df["label"] == cls).sum() == 40


def test_limitations_block_states_mixed_real_synthetic_features() -> None:
    """Acceptance Criterion 3: Limitations block states mixed real/synthetic features."""
    assert EXP_MD_PATH.is_file()
    md_content = EXP_MD_PATH.read_text(encoding="utf-8")

    with open(EXP_JSON_PATH, encoding="utf-8") as f:
        json_data = json.load(f)

    # Markdown Limitations check
    assert "## 5. Mandatory Limitations & Disclosures" in md_content
    assert (
        "mixed real and synthetic" in md_content.lower()
        or "mixed real/synthetic" in md_content.lower()
    )

    # JSON Limitations check
    limitations = json_data.get("limitations", [])
    assert len(limitations) > 0
    assert any("mixed real" in lim.lower() for lim in limitations)

    # Check that individual sensor model validation reports also include this disclosure
    for model_type in MODEL_FAMILIES:
        m_reports = list(REPORTS_DIR.glob(f"*sensor*{model_type}*_validation.md"))
        if m_reports:
            content = m_reports[0].read_text(encoding="utf-8")
            assert (
                "mixed real and synthetic" in content.lower()
                or "mixed real/synthetic" in content.lower()
            )


def test_all_five_model_families_evaluated() -> None:
    """Verify all five model families are present in the comparison report."""
    with open(EXP_JSON_PATH, encoding="utf-8") as f:
        json_data = json.load(f)

    models_in_report = [m["model_type"] for m in json_data["models"]]
    for family in MODEL_FAMILIES:
        assert family in models_in_report, (
            f"Model family '{family}' missing from experiment report"
        )

    # Verify no test metrics leaked into validation reports
    for m in json_data["models"]:
        assert "val_macro_f1" in m
        assert "cv_macro_f1" in m
        assert "test_macro_f1" not in m
        assert "test_accuracy" not in m
