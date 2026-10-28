"""Unit tests for spinach OpenCV-feature classifiers (T5-015)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ml.image_models.train_features as tf_mod  # noqa: E402
from ml.features.spinach import SPINACH_FEATURE_NAMES  # noqa: E402

# Enable unpickling of FeaturePipeline if pickled under __main__
sys.modules["__main__"].FeaturePipeline = tf_mod.FeaturePipeline

EXPECTED_SPINACH_CLASSES = [
    "Spinach_Healthy",
    "Spinach_Non fresh",
    "Spinach_anthracnose",
    "Spinach_bacterial spot",
    "Spinach_downy mildew",
    "Spinach_pest damage",
]


def test_spinach_feature_dataset() -> None:
    feat_dir = PROJECT_ROOT / "processed_data/features/spinach"
    assert feat_dir.exists()

    meta_file = feat_dir / "metadata.json"
    assert meta_file.exists()
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["crop"] == "spinach"
    assert meta["schema_version"] == "1.0.0"
    assert list(meta["feature_names"]) == list(SPINACH_FEATURE_NAMES)
    assert meta["total_images"] == 18302
    assert "mask_config" in meta
    assert meta["failed_images_count"] >= 0

    train_df = pd.read_csv(feat_dir / "train.csv")
    val_df = pd.read_csv(feat_dir / "val.csv")
    test_df = pd.read_csv(feat_dir / "test.csv")

    assert len(train_df) == 12812
    assert len(val_df) == 2745
    assert len(test_df) == 2745

    for fname in SPINACH_FEATURE_NAMES:
        assert not train_df[fname].isna().any()
        assert not val_df[fname].isna().any()
        assert not test_df[fname].isna().any()


@pytest.mark.parametrize("model_family", ["random_forest", "svm"])
def test_spinach_model_artifacts(model_family: str) -> None:
    model_dir = PROJECT_ROOT / f"models/image/spinach/{model_family}/v1.0.0"
    model_path = model_dir / "model.joblib"
    meta_path = model_dir / "metadata.json"

    assert model_path.exists()
    assert meta_path.exists()

    pipeline = joblib.load(model_path)
    assert "scaler" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["crop"] == "spinach"
    assert meta["feature_order"] == list(SPINACH_FEATURE_NAMES)
    assert "mask_config" in meta
    assert "segmentation_config" in meta
    assert "excluded_images_count" in meta
    assert len(meta["class_support_validation"]) == 6
    assert meta["val_macro_f1"] > 0.0


def test_spinach_reports() -> None:
    reports_dir = PROJECT_ROOT / "reports/spinach"
    comp_json = reports_dir / "spinach_feature_experiments.json"
    comp_md = reports_dir / "spinach_feature_experiments.md"

    assert comp_json.exists()
    assert comp_md.exists()

    with open(comp_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["evaluation_split"] == "validation"
    assert data["schema_version"] == "1.0.0"
    assert len(data["models"]) >= 2

    for model_info in data["models"]:
        per_class = model_info["per_class"]
        assert len(per_class) == 6
        for cls_name in EXPECTED_SPINACH_CLASSES:
            assert cls_name in per_class
            assert per_class[cls_name]["support"] > 0
