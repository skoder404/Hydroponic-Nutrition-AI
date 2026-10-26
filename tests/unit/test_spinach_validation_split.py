"""Unit tests for Spinach Sensor Validation Set Split (T2-011)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.make_sensor_validation import (
    compute_spinach_blocks,
    make_spinach_validation,
)

ORIG_DATA_DIR = ROOT / "sensor_data" / "Spinach_Sensor-Readings"
V2_DATA_DIR = ROOT / "processed_data" / "sensor" / "spinach"
MANIFEST_FILE = V2_DATA_DIR / "manifest.json"

EXPECTED_CLASSES = [
    "EC_High_Toxicity",
    "EC_Low_Deficiency",
    "Humidity_High_Mildew_Risk",
    "Humidity_Low_Stress",
    "Normal",
    "Water_Temp_High",
    "pH_Imbalance",
]


def test_v2_spinach_files_exist() -> None:
    """Verify v2 train, val, test CSVs and manifest exist."""
    assert (V2_DATA_DIR / "train.csv").is_file()
    assert (V2_DATA_DIR / "val.csv").is_file()
    assert (V2_DATA_DIR / "test.csv").is_file()
    assert MANIFEST_FILE.is_file()


def test_spinach_test_set_identical_to_v1() -> None:
    """Verify v2 test set is identical to original v1 test set."""
    orig = pd.read_csv(ORIG_DATA_DIR / "spinach_7class_test.csv")
    v2 = pd.read_csv(V2_DATA_DIR / "test.csv")
    assert v2.equals(orig)
    assert len(v2) == 280
    assert (v2["label"].value_counts() == 40).all()


def test_spinach_disjoint_temporal_blocks_and_ids() -> None:
    """Verify no 6-hour temporal block or sample_id appears in more than one partition."""
    train = pd.read_csv(V2_DATA_DIR / "train.csv")
    val = pd.read_csv(V2_DATA_DIR / "val.csv")
    test = pd.read_csv(V2_DATA_DIR / "test.csv")

    train_blocks = set(compute_spinach_blocks(train))
    val_blocks = set(compute_spinach_blocks(val))
    test_blocks = set(compute_spinach_blocks(test))

    assert len(train_blocks & val_blocks) == 0
    assert len(train_blocks & test_blocks) == 0
    assert len(val_blocks & test_blocks) == 0

    assert len(train_blocks) == 77
    assert len(val_blocks) == 11
    assert len(test_blocks) == 22

    # Sample ID disjointness
    train_ids = set(train["sample_id"])
    val_ids = set(val["sample_id"])
    test_ids = set(test["sample_id"])

    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0


def test_spinach_manifest_coverage_and_counts() -> None:
    """Verify manifest records per-class counts across all 7 classes and partitions."""
    data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))

    assert data["dataset_version"] == "v2"
    assert data["crop"] == "spinach"
    assert data["row_counts"]["train"] == 958
    assert data["row_counts"]["val"] == 162
    assert data["row_counts"]["test"] == 280
    assert data["row_counts"]["total_v2"] == 1400

    coverage = data["class_coverage"]
    assert len(coverage) == 7
    for cls in EXPECTED_CLASSES:
        assert cls in coverage
        c = coverage[cls]
        assert c["total_rows"] == 200
        assert c["test_rows"] == 40
        assert c["train_rows"] + c["val_rows"] == 160
        assert c["train_rows"] > 120
        assert c["val_rows"] > 10


def test_make_spinach_validation_reproducible(tmp_path: Path) -> None:
    """Verify make_spinach_validation executes cleanly to custom output directory."""
    code, manifest = make_spinach_validation(output_dir=tmp_path, seed=42)
    assert code == 0
    assert (tmp_path / "train.csv").is_file()
    assert (tmp_path / "val.csv").is_file()
    assert (tmp_path / "test.csv").is_file()
    assert (tmp_path / "manifest.json").is_file()
    assert manifest["row_counts"]["val"] == 162
