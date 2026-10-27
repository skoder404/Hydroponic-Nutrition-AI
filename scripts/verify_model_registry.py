#!/usr/bin/env python3
"""Validation script for ModelRegistry and Promoted Models (T4-025).

Acceptance Criteria:
1. Three promoted entries exist (lettuce, spinach, tomato sensor models).
2. Registry validation script passes (valid metadata, artifact, experimental flag, schema guard).
3. Rollback tested once on a dummy version (verifying clean restoration of previous known-good version).

Usage:
    python scripts/verify_model_registry.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.registry import ModelMetadata, ModelRegistry, SchemaMismatchError


def verify_three_promoted_entries_exist(registry: ModelRegistry) -> tuple[bool, str]:
    """Criterion 1: Three promoted entries exist in models/manifests/."""
    expected_crops = ["lettuce", "spinach", "tomato"]
    missing = []
    pointers = {}

    for crop in expected_crops:
        pointer_file = registry._get_active_pointer_file(crop, "sensor")
        if not pointer_file.is_file():
            missing.append(str(pointer_file.relative_to(PROJECT_ROOT)))
            continue
        try:
            data = json.loads(pointer_file.read_text(encoding="utf-8"))
            pointers[crop] = data
        except Exception as exc:
            return False, f"Failed to parse JSON in pointer '{pointer_file}': {exc}"

    if missing:
        return False, f"Missing active pointer files for: {missing}"

    for crop in expected_crops:
        p = pointers[crop]
        if p.get("crop") != crop or p.get("modality") != "sensor" or not p.get("active_version"):
            return False, f"Invalid pointer structure for {crop}: {p}"

    return True, (
        f"Three promoted entries exist in models/manifests/ "
        f"(Lettuce v{pointers['lettuce']['active_version']}, "
        f"Spinach v{pointers['spinach']['active_version']}, "
        f"Tomato v{pointers['tomato']['active_version']})."
    )


def verify_promoted_models_integrity(registry: ModelRegistry) -> tuple[bool, str]:
    """Criterion 2: Metadata validity, artifact existence, experimental flags, schema compatibility."""
    expected = {
        "lettuce": {"experimental": False, "family": "DecisionTreeClassifier"},
        "spinach": {"experimental": False, "family": "DecisionTreeClassifier"},
        "tomato": {"experimental": True, "family": "RandomForestClassifier"},
    }

    details = []

    for crop, exp_spec in expected.items():
        # 1. Load active metadata
        try:
            metadata = registry.get_active_metadata(crop, "sensor")
        except Exception as exc:
            return False, f"Failed to load active metadata for {crop}/sensor: {exc}"

        # 2. Verify status is promoted
        if metadata.status != "promoted":
            return False, f"Model status for {crop}/sensor is '{metadata.status}', expected 'promoted'."

        # 3. Verify experimental flag (Tomato MUST be experimental=True per Section 15 / ADR 0005)
        if metadata.experimental != exp_spec["experimental"]:
            return False, (
                f"Experimental flag mismatch for {crop}: got {metadata.experimental}, "
                f"expected {exp_spec['experimental']}."
            )

        # 4. Verify model family
        if metadata.family != exp_spec["family"]:
            return False, (
                f"Model family mismatch for {crop}: got {metadata.family}, "
                f"expected {exp_spec['family']}."
            )

        # 5. Verify artifact loading through trusted path guard
        try:
            model_obj, loaded_meta = registry.load_model(crop, "sensor")
            if model_obj is None:
                return False, f"Loaded model object is None for {crop}/sensor."
        except Exception as exc:
            return False, f"Failed to safely load model artifact for {crop}/sensor: {exc}"

        # 6. Verify feature schema compatibility guard
        try:
            registry.load_model(crop, "sensor", expected_features=metadata.ordered_features)
        except Exception as exc:
            return False, f"Schema validation failed for valid feature list on {crop}/sensor: {exc}"

        # 7. Verify corrupted/mismatched feature list triggers SchemaMismatchError
        try:
            registry.load_model(crop, "sensor", expected_features=["invalid_dummy_feature"])
            return False, f"Failed to raise SchemaMismatchError on mismatched features for {crop}/sensor."
        except SchemaMismatchError:
            pass  # Expected

        details.append(
            f"{crop.capitalize()} (v{metadata.artifact_version}, {metadata.family}, "
            f"experimental={metadata.experimental}, status={metadata.status})"
        )

    return True, f"Promoted models verified: {'; '.join(details)}."


def verify_rollback_on_dummy_version() -> tuple[bool, str]:
    """Criterion 3: Rollback tested once on an isolated dummy version."""
    temp_dir = Path(tempfile.mkdtemp(prefix="dummy_registry_"))
    try:
        dummy_reg = ModelRegistry(root_dir=temp_dir)

        # Build dummy metadata v1.0.0
        meta_dict_v1 = {
            "model_id": "dummy_crop_sensor_dummy_v1.0.0",
            "crop": "lettuce",
            "task": "classification",
            "modality": "sensor",
            "family": "DecisionTreeClassifier",
            "artifact_version": "1.0.0",
            "artifact_file": "sensor/lettuce/v1.0.0/dummy_v1.joblib",
            "commit": "testcommitdummy01",
            "training_date": "2026-10-27T00:00:00+00:00",
            "dataset_manifest_hash": "c62bf6a127fce1c4efdfa88d752ad755a7201b2fba9eb73b306b9b1e92d6e3c5",
            "rule_version": "1.0.0",
            "preprocessing_version": "1.0.0",
            "experimental": False,
            "status": "candidate",
            "classes": ["Normal", "EC_High"],
            "display_mapping": {"Normal": "healthy", "EC_High": "unhealthy/EC_High"},
            "split_protocol": "Calendar holdout",
            "split_counts": {"train": 100, "validation": 20, "test": 20},
            "ordered_features": ["air_temp", "ph"],
            "feature_units": {"air_temp": "degC", "ph": "pH"},
            "input_unit": "sensor_reading",
            "hyperparameters": {"max_depth": 3},
            "confidence_settings": {"uncertainty_threshold": 0.4},
            "expected_input_schema": {"type": "object"},
            "dependency_versions": {"python": "3.13.5"},
            "validation_metrics": {"accuracy": 0.95, "macro_f1": 0.94},
        }

        # Save and promote v1.0.0
        dummy_reg.save_model({"weight": 1.0}, ModelMetadata.model_validate(meta_dict_v1))
        res_p1 = dummy_reg.promote("lettuce", "sensor", "1.0.0", called_from_cli=True)
        if res_p1["active_version"] != "1.0.0" or res_p1["previous_version"] is not None:
            return False, f"Dummy promotion v1.0.0 failed: {res_p1}"

        # Build dummy metadata v1.1.0
        meta_dict_v2 = dict(meta_dict_v1)
        meta_dict_v2["model_id"] = "dummy_crop_sensor_dummy_v1.1.0"
        meta_dict_v2["artifact_version"] = "1.1.0"
        meta_dict_v2["artifact_file"] = "sensor/lettuce/v1.1.0/dummy_v2.joblib"

        # Save and promote v1.1.0
        dummy_reg.save_model({"weight": 2.0}, ModelMetadata.model_validate(meta_dict_v2))
        res_p2 = dummy_reg.promote("lettuce", "sensor", "1.1.0", called_from_cli=True)
        if res_p2["active_version"] != "1.1.0" or res_p2["previous_version"] != "1.0.0":
            return False, f"Dummy promotion v1.1.0 failed: {res_p2}"

        # Rollback to v1.0.0
        res_rb = dummy_reg.rollback("lettuce", "sensor", called_from_cli=True)
        if res_rb["active_version"] != "1.0.0" or res_rb["previous_version"] != "1.1.0":
            return False, f"Dummy rollback failed: {res_rb}"

        # Verify active model loaded is actually v1.0.0
        loaded_obj, loaded_meta = dummy_reg.load_model("lettuce", "sensor")
        if loaded_meta.artifact_version != "1.0.0" or loaded_obj.get("weight") != 1.0:
            return False, f"Loaded model after rollback is not v1.0.0: meta={loaded_meta.artifact_version}"

        return True, "Rollback cleanly restored active model to v1.0.0 on dummy test registry."
    except Exception as exc:
        return False, f"Exception during rollback test: {exc}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    registry = ModelRegistry()

    checks = [
        ("Three promoted entries exist in models/manifests/", lambda: verify_three_promoted_entries_exist(registry)),
        ("Registry validation: metadata, artifact integrity, experimental flag, schema guard", lambda: verify_promoted_models_integrity(registry)),
        ("Rollback tested once on a dummy version", verify_rollback_on_dummy_version),
    ]

    print("================================================================================")
    print("VERIFY MODEL REGISTRY AND PROMOTION (T4-025)")
    print("================================================================================\n")

    all_passed = True
    for title, check_fn in checks:
        ok, msg = check_fn()
        status = "[PASS]" if ok else "[FAIL]"
        print(f"{status} {title}\n       {msg}")
        if not ok:
            all_passed = False

    print("\n--------------------------------------------------------------------------------")
    if all_passed:
        print("All T4-025 acceptance criteria successfully verified!")
        return 0
    print("Some criteria checks failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
