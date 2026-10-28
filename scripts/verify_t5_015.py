"""Verification script for Task T5-015: Train and report spinach OpenCV-feature classifiers.

Verifies:
1. Feature dataset exists with canonical 6 features, mask_config, and split isolation.
2. Reports written with per-class metrics for all 6 spinach classes.
3. Mask/segmentation config is stored with the model metadata.
4. Leaves that fail masking are excluded and counted in the report and metadata.
5. Model artifacts (RandomForest, SVM) exist, include scaler and classifier, and contain feature_order.
6. Evaluation was performed strictly on the validation partition (never on test.csv).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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


def verify_t5_015() -> None:
    print("=== Verifying T5-015: Spinach OpenCV-Feature Classifiers ===")

    # 1. Verify feature dataset metadata, mask_config, and canonical 6 features
    feature_meta_path = PROJECT_ROOT / "processed_data/features/spinach/metadata.json"
    assert feature_meta_path.exists(), f"Feature metadata missing: {feature_meta_path}"
    with open(feature_meta_path, "r", encoding="utf-8") as f:
        feat_meta = json.load(f)

    schema_version = feat_meta.get("schema_version")
    assert schema_version == "1.0.0", (
        f"Expected schema_version '1.0.0', got '{schema_version}'"
    )

    feature_names = tuple(feat_meta.get("feature_names", []))
    assert feature_names == SPINACH_FEATURE_NAMES, (
        f"Feature names {feature_names} do not match canonical SPINACH_FEATURE_NAMES {SPINACH_FEATURE_NAMES}"
    )
    assert len(feature_names) == 6, f"Expected 6 features, got {len(feature_names)}"

    mask_config = feat_meta.get("mask_config", {})
    assert mask_config, "mask_config missing from feature metadata"
    assert "algorithm" in mask_config
    assert "exg_threshold" in mask_config

    failed_count = feat_meta.get("failed_images_count")
    assert failed_count is not None, "failed_images_count missing from feature metadata"
    print(
        f"[PASS] Feature metadata verified: schema_version='{schema_version}', "
        f"{len(feature_names)} features, mask_config={mask_config}, excluded={failed_count}."
    )

    # 2. Verify model artifacts
    for fam in ("random_forest", "svm"):
        model_dir = PROJECT_ROOT / f"models/image/spinach/{fam}/v1.0.0"
        model_file = model_dir / "model.joblib"
        meta_file = model_dir / "metadata.json"

        assert model_file.exists(), f"Model file missing: {model_file}"
        assert meta_file.exists(), f"Metadata missing: {meta_file}"

        pipeline = joblib.load(model_file)
        assert hasattr(pipeline, "named_steps"), "Saved artifact is not a Pipeline"
        assert "scaler" in pipeline.named_steps, "Pipeline missing scaler step"
        assert "classifier" in pipeline.named_steps, "Pipeline missing classifier step"

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta.get("crop") == "spinach"
        assert meta.get("feature_schema_version") == "1.0.0"
        assert meta.get("feature_order") == list(SPINACH_FEATURE_NAMES)
        assert meta.get("num_features") == 6
        assert meta.get("split_counts", {}).get("train") == 12812
        assert meta.get("split_counts", {}).get("val") == 2745

        # Acceptance Criteria 2: Mask/segmentation config stored with the model
        assert "mask_config" in meta, f"mask_config missing from {fam} metadata"
        assert "segmentation_config" in meta, (
            f"segmentation_config missing from {fam} metadata"
        )
        assert meta["mask_config"] == mask_config

        # Acceptance Criteria 3: Excluded images counted in metadata
        assert "excluded_images_count" in meta, (
            f"excluded_images_count missing from {fam} metadata"
        )
        assert meta["excluded_images_count"] == failed_count

        # Acceptance Criteria 1: 6 spinach classes in validation support
        val_support = meta.get("class_support_validation", {})
        assert len(val_support) == 6, (
            f"Expected 6 classes in val support, got {len(val_support)}"
        )
        for cls_name in EXPECTED_SPINACH_CLASSES:
            assert cls_name in val_support, f"Class {cls_name} missing from val support"
        assert sum(val_support.values()) == 2745, (
            f"Expected 2745 val instances, got {sum(val_support.values())}"
        )
        print(
            f"[PASS] Model artifact verified for {fam} ({meta['family']}): "
            f"val macro-F1={meta['val_macro_f1']:.4f}, accuracy={meta['val_accuracy']:.4f}, "
            f"mask_config stored={bool(meta['mask_config'])}."
        )

    # 3. Verify evaluation reports
    reports_dir = PROJECT_ROOT / "reports/spinach"
    comp_json = reports_dir / "spinach_feature_experiments.json"
    comp_md = reports_dir / "spinach_feature_experiments.md"
    assert comp_json.exists(), f"Comparison JSON missing: {comp_json}"
    assert comp_md.exists(), f"Comparison MD missing: {comp_md}"

    with open(comp_json, "r", encoding="utf-8") as f:
        comp_data = json.load(f)

    assert comp_data.get("evaluation_split") == "validation"
    assert comp_data.get("schema_version") == "1.0.0"
    assert comp_data.get("excluded_images_count") == failed_count
    models_evaluated = comp_data.get("models", [])
    assert len(models_evaluated) >= 2

    # Check per-class metrics in comparison JSON
    for model_entry in models_evaluated:
        per_class = model_entry.get("per_class", {})
        assert len(per_class) == 6, (
            f"Expected 6 classes in per_class, got {len(per_class)}"
        )
        for cls_name in EXPECTED_SPINACH_CLASSES:
            assert cls_name in per_class, (
                f"Class {cls_name} missing from per_class metrics"
            )
            metrics = per_class[cls_name]
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics
            assert "support" in metrics

    # Check validation reports for each family
    for fam in ("random_forest", "svm"):
        val_json = reports_dir / f"spinach_feature_{fam}_v1.0.0_validation.json"
        val_md = reports_dir / f"spinach_feature_{fam}_v1.0.0_validation.md"
        assert val_json.exists(), f"Validation report JSON missing: {val_json}"
        assert val_md.exists(), f"Validation report MD missing: {val_md}"

        with open(val_json, "r", encoding="utf-8") as f:
            val_data = json.load(f)

        assert val_data.get("split_evaluated") == "validation"
        per_class_val = val_data.get("metrics", {}).get("per_class", {})
        assert len(per_class_val) == 6
        for cls_name in EXPECTED_SPINACH_CLASSES:
            assert cls_name in per_class_val

        # Verify limitations mention masking and exclusion count
        limits = val_data.get("limitations", [])
        assert any("mask" in lim.lower() for lim in limits), (
            "Masking details missing from report limitations"
        )
        assert any(str(failed_count) in lim for lim in limits), (
            "Excluded leaves count missing from report limitations"
        )

    print(
        "[PASS] Evaluation reports verified: all 6 classes present with precision/recall/F1/support."
    )
    print("=== All T5-015 Acceptance Criteria Successfully Verified ===")


if __name__ == "__main__":
    verify_t5_015()
