"""Verification script for Task T5-016: Train and report tomato OpenCV-feature classifiers.

Verifies:
1. Reports written with per-class metrics for all 11 tomato classes.
2. Feature schema version strictly matches ADR 0007 ("1.0.0", 4 features).
3. Excluded images are accurately counted and documented (1 corrupt image HL_(336)).
4. Model artifacts (RandomForest, SVM) exist, include scaler and classifier, and contain feature_order.
5. Evaluation was performed strictly on the validation partition (never on test.csv).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features.tomato import TOMATO_FEATURE_NAMES
import ml.image_models.train_features as tf_mod

# Enable unpickling of FeaturePipeline if pickled under __main__
sys.modules["__main__"].FeaturePipeline = tf_mod.FeaturePipeline


def verify_t5_016() -> None:
    print("=== Verifying T5-016: Tomato OpenCV-Feature Classifiers ===")

    # 1. Verify feature dataset metadata and ADR 0007 compliance
    feature_meta_path = PROJECT_ROOT / "processed_data/features/tomato/metadata.json"
    assert feature_meta_path.exists(), f"Feature metadata missing: {feature_meta_path}"
    with open(feature_meta_path, "r", encoding="utf-8") as f:
        feat_meta = json.load(f)

    schema_version = feat_meta.get("schema_version")
    assert schema_version == "1.0.0", f"Expected schema_version '1.0.0', got '{schema_version}'"

    feature_names = tuple(feat_meta.get("feature_names", []))
    assert feature_names == TOMATO_FEATURE_NAMES, (
        f"Feature names {feature_names} do not match canonical TOMATO_FEATURE_NAMES {TOMATO_FEATURE_NAMES}"
    )
    assert "droop_angle" not in feature_names, "ADR 0007 violation: droop_angle must not be present"
    assert len(feature_names) == 4, f"Expected 4 features, got {len(feature_names)}"

    # Check excluded images
    failed_count = feat_meta.get("failed_images_count", 0)
    assert failed_count == 1, f"Expected 1 failed image, got {failed_count}"
    failed_images = feat_meta.get("failed_images", [])
    assert len(failed_images) == 1
    assert failed_images[0]["image_id"] == "HL_(336)"
    print(f"[PASS] Feature metadata verified: schema_version='{schema_version}', {len(feature_names)} features, {failed_count} excluded image ({failed_images[0]['image_id']}).")

    # 2. Verify model artifacts
    for fam in ("random_forest", "svm"):
        model_dir = PROJECT_ROOT / f"models/image/tomato/{fam}/v1.0.0"
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

        assert meta.get("crop") == "tomato"
        assert meta.get("feature_schema_version") == "1.0.0"
        assert meta.get("feature_order") == list(TOMATO_FEATURE_NAMES)
        assert meta.get("excluded_images_count") == 1
        assert meta.get("split_counts", {}).get("train") == 23495
        assert meta.get("split_counts", {}).get("val") == 5034

        # Validate 11 classes in support
        val_support = meta.get("class_support_validation", {})
        assert len(val_support) == 11, f"Expected 11 classes in val support, got {len(val_support)}"
        assert sum(val_support.values()) == 5034, f"Expected 5034 val instances, got {sum(val_support.values())}"
        print(f"[PASS] Model artifact verified for {fam} ({meta['family']}): val macro-F1={meta['val_macro_f1']:.4f}, accuracy={meta['val_accuracy']:.4f}.")

    # 3. Verify evaluation reports
    reports_dir = PROJECT_ROOT / "reports/tomato"
    comp_json = reports_dir / "tomato_feature_experiments.json"
    comp_md = reports_dir / "tomato_feature_experiments.md"
    assert comp_json.exists(), f"Comparison JSON missing: {comp_json}"
    assert comp_md.exists(), f"Comparison MD missing: {comp_md}"

    with open(comp_json, "r", encoding="utf-8") as f:
        comp_data = json.load(f)

    assert comp_data.get("evaluation_split") == "validation"
    assert comp_data.get("schema_version") == "1.0.0"
    assert comp_data.get("excluded_images_count") == 1
    models_evaluated = comp_data.get("models", [])
    assert len(models_evaluated) >= 2

    for m in models_evaluated:
        per_class = m.get("per_class", {})
        assert len(per_class) == 11, f"Expected 11 classes in per-class metrics, got {len(per_class)}"
        for cls_name, p_metrics in per_class.items():
            assert "precision" in p_metrics
            assert "recall" in p_metrics
            assert "f1" in p_metrics
            assert "support" in p_metrics
            assert p_metrics["support"] > 0

    md_content = comp_md.read_text(encoding="utf-8")
    assert "droop_angle" in md_content and "removed" in md_content
    assert "HL_(336)" in md_content
    assert "Excluded Images Count:" in md_content

    # Individual Section 18.2 reports
    for fam in ("random_forest", "svm"):
        val_rep_md = reports_dir / f"tomato_feature_{fam}_v1.0.0_validation.md"
        val_rep_json = reports_dir / f"tomato_feature_{fam}_v1.0.0_validation.json"
        assert val_rep_md.exists(), f"Validation report MD missing: {val_rep_md}"
        assert val_rep_json.exists(), f"Validation report JSON missing: {val_rep_json}"

        rep_text = val_rep_md.read_text(encoding="utf-8")
        assert "## 7. Mandatory Limitations & Disclosures" in rep_text
        assert "ADR 0007" in rep_text
        assert "HL_(336)" in rep_text

    print("[PASS] All Section 18.2 and comparative reports verified.")
    print("\n>>> All T5-016 acceptance criteria successfully verified! <<<")


if __name__ == "__main__":
    verify_t5_016()
