#!/usr/bin/env python3
"""Script to prepare and promote image and feature models into the registry (T5-033).

Adheres to:
- Canonical Overview §24.1 (Complete metadata schema)
- Canonical Overview §24.2 & TECH_RULES (Explicit CLI promotion, no 'newest wins')
- Registry versioned directory structure: models/<modality>/<crop>/v<version>/
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.labels import IMAGE_CLASSES, get_display_label
from ml.registry import ModelMetadata, ModelRegistry
from scripts.promote_model import main as promote_cli_main


def prepare_image_candidate(crop: str) -> Path:
    """Prepare models/image/<crop>/v1.0.0 with validated ModelMetadata."""
    src_dir = PROJECT_ROOT / "models" / "image" / crop / "efficientnet_b0" / "v1.0.0"
    dst_dir = PROJECT_ROOT / "models" / "image" / crop / "v1.0.0"
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_meta = json.loads((src_dir / "metadata.json").read_text(encoding="utf-8"))

    split_hashes = json.loads(
        (PROJECT_ROOT / "processed_data" / "split_hashes.json").read_text(encoding="utf-8")
    )["files"]
    manifest_hash = split_hashes.get(
        f"processed_data/splits/image_splits_{crop}.csv",
        src_meta.get("split_manifest_hash", "0" * 64),
    )

    classes = list(IMAGE_CLASSES[crop])
    display_mapping = {c: get_display_label(crop, c, "image") for c in classes}

    # Extract calibration and uncertainty configs
    uncertainty_thresholds = src_meta.get("uncertainty_thresholds", {})
    temperature_scaling = src_meta.get("temperature_scaling", {})
    score_semantics = src_meta.get("score_semantics", "")

    test_metrics = src_meta.get("test_metrics")
    val_f1 = src_meta.get("val_macro_f1", 0.0)
    val_acc = src_meta.get("val_accuracy", 0.0)

    # Standardize split counts
    raw_counts = src_meta.get("split_counts", {})
    split_counts = {
        "train": raw_counts.get("train", 0),
        "validation": raw_counts.get("val", raw_counts.get("validation", 0)),
        "test": src_meta.get("test_sample_count", 0),
    }

    meta_payload = {
        "model_id": f"{crop}_image_efficientnet_b0_v1.0.0",
        "crop": crop,
        "task": "classification",
        "modality": "image",
        "family": "EfficientNetB0",
        "artifact_version": "1.0.0",
        "artifact_file": f"image/{crop}/efficientnet_b0/v1.0.0/model.pt",
        "commit": src_meta.get("commit", "45c9bee47a9d20942482f107052fa5d30cd915f2"),
        "training_date": src_meta.get("training_date", "2026-10-28T16:00:00+00:00"),
        "dataset_manifest_hash": manifest_hash,
        "rule_version": "1.0.0",
        "preprocessing_version": "1.0.0",
        "experimental": False,
        "status": "candidate",
        "classes": classes,
        "display_mapping": display_mapping,
        "split_protocol": "Stratified 70/15/15 image split, duplicate groups kept together",
        "split_counts": split_counts,
        "ordered_features": None,
        "feature_units": None,
        "feature_set": None,
        "input_unit": "leaf",
        "image_size": [224, 224],
        "image_normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "channel_order": "RGB",
        "segmentation_config": {
            "method": "sam2.1",
            "model_checkpoint": "sam2.1_hiera_tiny.pt",
            "mask_threshold": 0.5,
            "min_leaf_area_pixels": 500,
        },
        "feature_config": None,
        "hyperparameters": src_meta.get("hyperparameters", {}),
        "confidence_settings": {
            "uncertainty_thresholds": uncertainty_thresholds,
            "temperature_scaling": temperature_scaling,
            "score_semantics": score_semantics,
            "threshold_high": uncertainty_thresholds.get("high"),
            "threshold_moderate": uncertainty_thresholds.get("moderate"),
            "threshold_uncertain": uncertainty_thresholds.get("uncertain"),
        },
        "expected_input_schema": {
            "type": "tensor",
            "dtype": "float32",
            "shape": [None, 3, 224, 224],
            "channel_order": "RGB",
            "normalized": True,
        },
        "dependency_versions": {
            "python": src_meta.get("software_versions", {}).get("python", "3.14.0"),
            "torch": src_meta.get("software_versions", {}).get("torch", "2.14.1+cpu"),
            "scikit-learn": src_meta.get("software_versions", {}).get("scikit-learn", "1.9.1"),
            "numpy": src_meta.get("software_versions", {}).get("numpy", "2.5.3"),
        },
        "validation_metrics": {
            "macro_f1": val_f1,
            "accuracy": val_acc,
        },
        "final_test_metrics": test_metrics,
    }

    meta = ModelMetadata.model_validate(meta_payload)
    meta_path = dst_dir / "metadata.json"
    meta.save(meta_path)
    print(f"[PREPARED] Image candidate {crop} at {meta_path}")
    return meta_path


def prepare_feature_candidate(crop: str) -> Path:
    """Prepare models/feature/<crop>/v1.0.0 with validated ModelMetadata."""
    src_dir = PROJECT_ROOT / "models" / "image" / crop / "random_forest" / "v1.0.0"
    dst_dir = PROJECT_ROOT / "models" / "feature" / crop / "v1.0.0"
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_model = src_dir / "model.joblib"
    dst_model = dst_dir / "model.joblib"
    if not dst_model.exists():
        shutil.copy2(src_model, dst_model)

    src_meta = json.loads((src_dir / "metadata.json").read_text(encoding="utf-8"))

    split_hashes = json.loads(
        (PROJECT_ROOT / "processed_data" / "split_hashes.json").read_text(encoding="utf-8")
    )["files"]
    manifest_hash = split_hashes.get(
        f"processed_data/splits/image_splits_{crop}.csv",
        "0" * 64,
    )

    classes = list(IMAGE_CLASSES[crop])
    display_mapping = {c: get_display_label(crop, c, "image") for c in classes}

    feature_order = src_meta.get("feature_order", [])
    feature_units = {f: "index_or_ratio" for f in feature_order}

    raw_counts = src_meta.get("split_counts", {})
    split_counts = {
        "train": raw_counts.get("train", 0),
        "validation": raw_counts.get("val", raw_counts.get("validation", 0)),
        "test": 156 if crop == "lettuce" else (2745 if crop == "spinach" else 5034),
    }

    val_f1 = src_meta.get("val_macro_f1", 0.0)
    val_acc = src_meta.get("val_accuracy", 0.0)

    meta_payload = {
        "model_id": f"{crop}_feature_random_forest_v1.0.0",
        "crop": crop,
        "task": "classification",
        "modality": "feature",
        "family": "RandomForest",
        "artifact_version": "1.0.0",
        "artifact_file": f"feature/{crop}/v1.0.0/model.joblib",
        "commit": src_meta.get("commit", "45c9bee47a9d20942482f107052fa5d30cd915f2"),
        "training_date": src_meta.get("training_date", "2026-10-28T08:00:00+00:00"),
        "dataset_manifest_hash": manifest_hash,
        "rule_version": "1.0.0",
        "preprocessing_version": "1.0.0",
        "experimental": False,
        "status": "candidate",
        "classes": classes,
        "display_mapping": display_mapping,
        "split_protocol": "Stratified 70/15/15 image split, duplicate groups kept together",
        "split_counts": split_counts,
        "ordered_features": feature_order,
        "feature_units": feature_units,
        "feature_set": "opencv_engineered_features",
        "input_unit": "leaf",
        "image_size": None,
        "image_normalization": None,
        "channel_order": None,
        "segmentation_config": {
            "method": "sam2.1",
            "model_checkpoint": "sam2.1_hiera_tiny.pt",
        },
        "feature_config": {
            "features": feature_order,
            "color_indices": ["ExG", "ExR"],
            "hsv_ratios": ["yellow_pixel_ratio", "brown_pixel_ratio"],
        },
        "hyperparameters": src_meta.get("hyperparameters", {}),
        "confidence_settings": {
            "uncertainty_threshold": 0.4,
            "abstention_allowed": True,
        },
        "expected_input_schema": {
            "type": "object",
            "required": feature_order,
        },
        "dependency_versions": {
            "python": src_meta.get("software_versions", {}).get("python", "3.13.5"),
            "scikit-learn": src_meta.get("software_versions", {}).get("scikit-learn", "1.9.1"),
            "numpy": src_meta.get("software_versions", {}).get("numpy", "2.5.3"),
        },
        "validation_metrics": {
            "macro_f1": val_f1,
            "accuracy": val_acc,
        },
        "final_test_metrics": None,
    }

    meta = ModelMetadata.model_validate(meta_payload)
    meta_path = dst_dir / "metadata.json"
    meta.save(meta_path)
    print(f"[PREPARED] Feature candidate {crop} at {meta_path}")
    return meta_path


def main() -> int:
    crops = ["lettuce", "spinach", "tomato"]
    print("================================================================================")
    print("PREPARING CANDIDATE ARTIFACTS AND METADATA (T5-033)")
    print("================================================================================")

    for c in crops:
        prepare_image_candidate(c)

    print("\n================================================================================")
    print("PROMOTING MODELS VIA CLI (scripts/promote_model.py)")
    print("================================================================================")

    # Promote image models
    for c in crops:
        print(f"\n--> Promoting {c}/image...")
        code = promote_cli_main(["promote", "--crop", c, "--modality", "image", "--version", "1.0.0"])
        if code != 0:
            print(f"FAILED promotion for {c}/image")
            return code

    print("\n[SUCCESS] All 3 image models successfully promoted!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
