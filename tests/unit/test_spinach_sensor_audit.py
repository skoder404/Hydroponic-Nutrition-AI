"""Unit tests for Spinach Sensor Dataset Audit (T2-006)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPINACH_DATA = REPO_ROOT / "sensor_data" / "Spinach_Sensor-Readings"
REPORT_PATH = REPO_ROOT / "reports" / "data" / "spinach_sensor_audit.md"

EXPECTED_CLASSES = {
    "Normal",
    "Water_Temp_High",
    "EC_High_Toxicity",
    "EC_Low_Deficiency",
    "pH_Imbalance",
    "Humidity_High_Mildew_Risk",
    "Humidity_Low_Stress",
}


def test_spinach_dataset_files_exist() -> None:
    """Verify all spinach sensor CSV files exist."""
    assert (SPINACH_DATA / "spinach_7class_full.csv").is_file()
    assert (SPINACH_DATA / "spinach_7class_train.csv").is_file()
    assert (SPINACH_DATA / "spinach_7class_test.csv").is_file()


def test_spinach_row_counts_and_distributions() -> None:
    """Verify exact row counts: full=1400 (200/class), train=1120 (160/class), test=280 (40/class)."""
    full = pd.read_csv(SPINACH_DATA / "spinach_7class_full.csv")
    train = pd.read_csv(SPINACH_DATA / "spinach_7class_train.csv")
    test = pd.read_csv(SPINACH_DATA / "spinach_7class_test.csv")

    assert len(full) == 1400
    assert len(train) == 1120
    assert len(test) == 280

    assert set(full["label"].unique()) == EXPECTED_CLASSES
    for count in full["label"].value_counts():
        assert count == 200

    for count in train["label"].value_counts():
        assert count == 160

    for count in test["label"].value_counts():
        assert count == 40


def test_spinach_six_hour_block_temporal_split_no_leakage() -> None:
    """Verify 6-hour temporal block holdout has zero leakage between train and test."""
    train = pd.read_csv(SPINACH_DATA / "spinach_7class_train.csv")
    test = pd.read_csv(SPINACH_DATA / "spinach_7class_test.csv")

    for df in (train, test):
        ts = pd.to_datetime(df["source_timestamp_EDT"])
        df["block"] = df["bench"] + "|" + ts.dt.strftime("%Y-%m-%d") + "|" + (ts.dt.hour // 6).astype(str)

    train_blocks = set(train["block"])
    test_blocks = set(test["block"])

    overlap = train_blocks.intersection(test_blocks)
    assert len(overlap) == 0, f"Overlapping 6-hour blocks found: {overlap}"

    id_overlap = set(train["sample_id"]).intersection(set(test["sample_id"]))
    assert len(id_overlap) == 0, f"Overlapping sample IDs: {id_overlap}"


def test_spinach_provenance_and_derived_tds() -> None:
    """Verify humidity_source counts and derived TDS formula."""
    full = pd.read_csv(SPINACH_DATA / "spinach_7class_full.csv")

    hum_counts = full["humidity_source"].value_counts().to_dict()
    assert hum_counts["real"] == 1200
    assert hum_counts["adjusted"] == 200

    # Humidity_High_Mildew_Risk should account for all 200 adjusted rows
    adjusted_labels = set(full[full["humidity_source"] == "adjusted"]["label"])
    assert adjusted_labels == {"Humidity_High_Mildew_Risk"}

    # TDS = EC * 700 (+- 2% noise)
    ratio = full["tds_ppm"] / (full["ec_mS_cm"] * 700)
    assert 0.98 <= float(ratio.mean()) <= 1.02


def test_spinach_audit_report_content() -> None:
    """Verify generated audit report exists and confirms passing status."""
    assert REPORT_PATH.is_file()
    content = REPORT_PATH.read_text(encoding="utf-8")
    assert "ALL CHECKS PASSED" in content
    assert "Zero temporal blocks are shared between train and test splits" in content
    assert "tds_ppm" in content
    assert "Derived (Excluded)" in content
