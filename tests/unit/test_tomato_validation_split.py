"""Unit tests for Tomato Sensor Validation Set Split (T2-012)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ORIG_DATA_DIR = ROOT / "sensor_data" / "Tomato_Sensor-Readings"
V2_DATA_DIR = ROOT / "processed_data" / "sensor" / "tomato"
MANIFEST_FILE = V2_DATA_DIR / "manifest.json"

EXPECTED_CLASSES = [
    "EC_High_Toxicity",
    "EC_Low_Deficiency",
    "Humidity_High_Mildew_Risk",
    "Humidity_Low_Stress",
    "K_Deficiency",
    "N_Deficiency",
    "Normal",
    "P_Deficiency",
    "Water_Temp_High",
    "pH_Imbalance",
]


def test_v2_tomato_files_exist() -> None:
    """Verify v2 train, val, test CSVs and manifest exist."""
    assert (V2_DATA_DIR / "train.csv").is_file()
    assert (V2_DATA_DIR / "val.csv").is_file()
    assert (V2_DATA_DIR / "test.csv").is_file()
    assert MANIFEST_FILE.is_file()


def test_tomato_test_set_identical_to_v1() -> None:
    """Verify v2 test set is identical to original v1 test set."""
    orig = pd.read_csv(ORIG_DATA_DIR / "tomato_hydroponic_test.csv")
    v2 = pd.read_csv(V2_DATA_DIR / "test.csv")
    assert v2.equals(orig)
    assert len(v2) == 8830
    assert (v2["label"].value_counts() == 883).all()


def test_tomato_disjoint_calendar_months_and_ids() -> None:
    """Verify no calendar month or sample_id appears in more than one partition."""
    train = pd.read_csv(V2_DATA_DIR / "train.csv")
    val = pd.read_csv(V2_DATA_DIR / "val.csv")
    test = pd.read_csv(V2_DATA_DIR / "test.csv")

    train_months = set(pd.to_datetime(train["source_timestamp"]).dt.strftime("%Y-%m"))
    val_months = set(pd.to_datetime(val["source_timestamp"]).dt.strftime("%Y-%m"))
    test_months = set(pd.to_datetime(test["source_timestamp"]).dt.strftime("%Y-%m"))

    # Verify disjoint months
    assert len(train_months & val_months) == 0, f"Train and Val share months: {train_months & val_months}"
    assert len(train_months & test_months) == 0, f"Train and Test share months: {train_months & test_months}"
    assert len(val_months & test_months) == 0, f"Val and Test share months: {val_months & test_months}"

    assert len(test_months) == 27
    assert len(val_months) == 16
    assert len(train_months) == 89
    assert len(test_months | val_months | train_months) == 132

    # Verify disjoint sample IDs
    train_ids = set(train["sample_id"])
    val_ids = set(val["sample_id"])
    test_ids = set(test["sample_id"])

    assert len(train_ids & val_ids) == 0, "Train and Val share sample IDs!"
    assert len(train_ids & test_ids) == 0, "Train and Test share sample IDs!"
    assert len(val_ids & test_ids) == 0, "Val and Test share sample IDs!"

    # Total row count
    assert len(train) + len(val) + len(test) == 44150


def test_tomato_manifest_coverage_and_counts() -> None:
    """Verify manifest records disjoint month lists, per-class counts, and Normal counts."""
    data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))

    assert data["dataset_version"] == "v2"
    assert data["crop"] == "tomato"

    splits = data["splits"]
    assert splits["test_months_count"] == 27
    assert splits["validation_months_count"] == 16
    assert splits["train_months_count"] == 89
    assert splits["total_months_count"] == 132

    test_m = set(splits["test_months"])
    val_m = set(splits["validation_months"])
    train_m = set(splits["train_months"])

    assert len(test_m & val_m) == 0
    assert len(train_m & val_m) == 0
    assert len(train_m & test_m) == 0

    # Row counts
    row_counts = data["row_counts"]
    assert row_counts["test"] == 8830
    assert row_counts["val"] == 5377
    assert row_counts["train"] == 29943
    assert row_counts["full"] == 44150

    # Normal class counts
    per_class = data["row_counts_per_class"]
    assert per_class["test"]["Normal"] == 883
    assert per_class["val"]["Normal"] == 549
    assert per_class["train"]["Normal"] == 2983
    assert per_class["full"]["Normal"] == 4415

    # Check all classes present
    coverage = data["class_coverage"]
    for cls in EXPECTED_CLASSES:
        assert cls in coverage
        c = coverage[cls]
        assert c["total_rows"] == 4415
        assert c["train_rows"] > 0
        assert c["val_rows"] > 0
        assert c["test_rows"] == 883
