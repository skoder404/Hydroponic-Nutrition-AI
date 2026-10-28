"""Unit tests for T5-033: Promote image and feature models to the registry.

Acceptance Criteria:
1. Registry validation passes for all promoted image models.
2. Missing combinations are listed in models/manifests/COVERAGE.md.
3. Rollback tested cleanly restoring previous known-good version.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from backend.app.core.paths import validate_model_path
from ml.labels import IMAGE_CLASSES
from ml.registry import ModelMetadata, ModelRegistry
from scripts.promote_model import main as promote_cli_main

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_promoted_image_models_registry_validation():
    """Acceptance Criterion 1: Registry validation passes for all promoted image models."""
    registry = ModelRegistry()
    expected_crops = ["lettuce", "spinach", "tomato"]

    for crop in expected_crops:
        # Verify active pointer exists
        pointer = registry.get_active_pointer(crop, "image")
        assert pointer["crop"] == crop
        assert pointer["modality"] == "image"
        assert pointer["active_version"] == "1.0.0"

        # Verify metadata parses and conforms to strict ModelMetadata schema
        metadata = registry.get_active_metadata(crop, "image")
        assert isinstance(metadata, ModelMetadata)
        assert metadata.status == "promoted"
        assert metadata.crop == crop
        assert metadata.modality == "image"
        assert metadata.family == "EfficientNetB0"
        assert metadata.input_unit == "leaf"
        assert metadata.channel_order == "RGB"
        assert metadata.image_size == [224, 224]
        assert metadata.classes == list(IMAGE_CLASSES[crop])
        assert metadata.validation_metrics["macro_f1"] > 0.70
        assert metadata.final_test_metrics is not None
        assert metadata.final_test_metrics["macro_f1"] > 0.70

        # Verify uncertainty thresholds and calibration settings are present
        assert "confidence_settings" in metadata.model_dump()
        conf = metadata.confidence_settings
        assert "uncertainty_thresholds" in conf
        assert "temperature_scaling" in conf
        assert conf["uncertainty_thresholds"]["high"] > 0.5

        # Verify model artifact exists and passes trusted path validation
        artifact_path = PROJECT_ROOT / "models" / pointer["active_artifact_path"]
        validate_model_path(artifact_path, trusted_root=registry.root_dir, must_exist=True)

        # Verify artifact loads safely
        model_obj, loaded_meta = registry.load_model(crop, "image")
        assert loaded_meta.artifact_version == "1.0.0"
        assert model_obj is not None


def test_coverage_manifest_documents_missing_combinations():
    """Acceptance Criterion 2: Missing combinations are listed in models/manifests/COVERAGE.md."""
    coverage_path = PROJECT_ROOT / "models" / "manifests" / "COVERAGE.md"
    assert coverage_path.is_file(), f"COVERAGE.md missing at {coverage_path}"

    content = coverage_path.read_text(encoding="utf-8")

    # Verify coverage table
    assert "Modality & Crop Coverage Matrix" in content
    assert "lettuce_image_efficientnet_b0_v1.0.0" in content
    assert "spinach_image_efficientnet_b0_v1.0.0" in content
    assert "tomato_image_efficientnet_b0_v1.0.0" in content

    # Verify explicitly listed missing/unsupported combinations
    assert "Tomato Live Image & Feature Models" in content or "tomato_live / image" in content
    assert "Spinach Live & Lettuce Live Telemetry" in content or "spinach / live" in content
    assert "Classical Handcrafted Feature Models" in content or "modality=\"feature\"" in content
    assert "Direct Whole-Canopy Unsegmented Classification Models" in content
    assert "Alternative Heavy Deep Learning Architectures" in content or "ResNet-50" in content


def test_model_rollback_via_cli(tmp_path: Path):
    """Acceptance Criterion 3: Rollback tested via scripts/promote_model.py."""
    # Create isolated test registry
    test_models_root = tmp_path / "models"
    test_models_root.mkdir(parents=True)
    manifests_dir = test_models_root / "manifests"
    manifests_dir.mkdir(parents=True)

    test_reg = ModelRegistry(root_dir=test_models_root)

    import torch

    # 1. Create v1.0.0 candidate
    v1_dir = test_models_root / "image" / "lettuce" / "v1.0.0"
    v1_dir.mkdir(parents=True)
    torch.save({"weight": 1.0}, v1_dir / "model.pt")

    meta_v1 = {
        "model_id": "lettuce_image_v1.0.0",
        "crop": "lettuce",
        "task": "classification",
        "modality": "image",
        "family": "EfficientNetB0",
        "artifact_version": "1.0.0",
        "artifact_file": "image/lettuce/v1.0.0/model.pt",
        "commit": "45c9bee47a9d20942482f107052fa5d30cd915f2",
        "training_date": "2026-10-28T16:00:00+00:00",
        "dataset_manifest_hash": "a" * 64,
        "rule_version": "1.0.0",
        "preprocessing_version": "1.0.0",
        "experimental": False,
        "status": "candidate",
        "classes": list(IMAGE_CLASSES["lettuce"]),
        "display_mapping": {c: f"disp_{c}" for c in IMAGE_CLASSES["lettuce"]},
        "split_protocol": "Stratified 70/15/15",
        "split_counts": {"train": 703, "validation": 128, "test": 156},
        "input_unit": "leaf",
        "image_size": [224, 224],
        "channel_order": "RGB",
        "hyperparameters": {"batch_size": 32},
        "confidence_settings": {"uncertainty_thresholds": {"high": 0.8}},
        "expected_input_schema": {"type": "tensor"},
        "dependency_versions": {"python": "3.14.0"},
        "validation_metrics": {"macro_f1": 0.91},
    }
    ModelMetadata.model_validate(meta_v1).save(v1_dir / "metadata.json")

    # Promote v1.0.0 via CLI
    code = promote_cli_main([
        "promote",
        "--crop", "lettuce",
        "--modality", "image",
        "--version", "1.0.0",
        "--root", str(test_models_root),
    ])
    assert code == 0
    pointer = test_reg.get_active_pointer("lettuce", "image")
    assert pointer["active_version"] == "1.0.0"
    assert pointer["previous_version"] is None

    # 2. Create v1.1.0 candidate
    v2_dir = test_models_root / "image" / "lettuce" / "v1.1.0"
    v2_dir.mkdir(parents=True)
    torch.save({"weight": 2.0}, v2_dir / "model.pt")

    meta_v2 = dict(meta_v1)
    meta_v2["artifact_version"] = "1.1.0"
    meta_v2["artifact_file"] = "image/lettuce/v1.1.0/model.pt"
    meta_v2["validation_metrics"] = {"macro_f1": 0.93}
    ModelMetadata.model_validate(meta_v2).save(v2_dir / "metadata.json")

    # Promote v1.1.0 via CLI
    code = promote_cli_main([
        "promote",
        "--crop", "lettuce",
        "--modality", "image",
        "--version", "1.1.0",
        "--root", str(test_models_root),
    ])
    assert code == 0
    pointer = test_reg.get_active_pointer("lettuce", "image")
    assert pointer["active_version"] == "1.1.0"
    assert pointer["previous_version"] == "1.0.0"

    # 3. Rollback to previous version (v1.0.0) via CLI
    code = promote_cli_main([
        "rollback",
        "--crop", "lettuce",
        "--modality", "image",
        "--root", str(test_models_root),
    ])
    assert code == 0
    pointer = test_reg.get_active_pointer("lettuce", "image")
    assert pointer["active_version"] == "1.0.0"
    assert pointer["previous_version"] == "1.1.0"

    # Verify active loaded metadata is restored to v1.0.0
    _, loaded_meta = test_reg.load_model("lettuce", "image")
    assert loaded_meta.artifact_version == "1.0.0"
    assert loaded_meta.validation_metrics["macro_f1"] == 0.91
