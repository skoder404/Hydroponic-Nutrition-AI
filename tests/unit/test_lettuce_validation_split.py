"""Unit tests for Lettuce Sensor Validation Set Split (T2-010)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ORIG_DATA_DIR = ROOT / "sensor_data" / "Lettuce_Sensor-Readings"
V2_DATA_DIR = ROOT / "processed_data" / "sensor" / "lettuce"
MANIFEST_FILE = V2_DATA_DIR / "manifest.json"

EXPECTED_CLASSES = [
    "EC_High",
    "Humidity_Low",
    "Multiple_Stress",
    "Normal",
    "WaterTemp_High",
    "pH_High",
    "pH_Low",
]


def test_v2_lettuce_files_exist() -> None:
    """Verify v2 train, val, test CSVs and manifest exist."""
    assert (V2_DATA_DIR / "train.csv").is_file()
    assert (V2_DATA_DIR / "val.csv").is_file()
    assert (V2_DATA_DIR / "test.csv").is_file()
    assert MANIFEST_FILE.is_file()


def test_test_set_identical_to_v1() -> None:
    """Verify v2 test set is identical to original v1 test set."""
    orig = pd.read_csv(ORIG_DATA_DIR / "hydroponic_lettuce_test.csv")
    v2 = pd.read_csv(V2_DATA_DIR / "test.csv")
    assert v2.equals(orig)
    assert len(v2) == 162


def test_disjoint_experiment_date_days() -> None:
    """Verify no (experiment, date) pair appears in more than one of train/val/test."""
    train = pd.read_csv(V2_DATA_DIR / "train.csv")
    val = pd.read_csv(V2_DATA_DIR / "val.csv")
    test = pd.read_csv(V2_DATA_DIR / "test.csv")

    train_keys = set(zip(train["experiment"], train["date"]))
    val_keys = set(zip(val["experiment"], val["date"]))
    test_keys = set(zip(test["experiment"], test["date"]))

    assert len(train_keys & val_keys) == 0
    assert len(train_keys & test_keys) == 0
    assert len(val_keys & test_keys) == 0


def test_manifest_coverage_and_counts() -> None:
    """Verify manifest records per-class real row counts and handles class flagging."""
    data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))

    assert data["dataset_version"] == "v2"
    assert data["crop"] == "lettuce"
    assert data["row_counts"]["test_real"] == 162
    assert data["row_counts"]["val_real"] == 125
    assert data["row_counts"]["train_balanced"] == 1295

    coverage = data["class_coverage"]
    assert len(coverage) == 7
    for cls in EXPECTED_CLASSES:
        assert cls in coverage
        assert coverage[cls]["balanced_train_rows"] == 185

    # Check pH_Low
    assert coverage["pH_Low"]["real_train_rows"] == 7
    assert coverage["pH_Low"]["real_val_rows"] == 1
    assert coverage["pH_Low"]["real_test_rows"] == 2
