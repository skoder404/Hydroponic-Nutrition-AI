"""Verification script for promoted model loading and GET /api/v1/models endpoint (T6-002).

Acceptance Criteria:
1. Removing one artifact breaks only that crop (reported as MODEL_UNAVAILABLE).
2. Incompatible feature list is reported as SCHEMA_MISMATCH.
3. /health reflects registry state ('ok', 'degraded', 'unavailable').
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient

from backend.app.core.errors import ModelUnavailableError, SchemaMismatchError
from backend.app.main import create_app
from backend.app.services.model_service import (
    ModelRegistryService,
    get_model_registry_service,
    reset_model_registry_service,
)


def verify_active_models_exposed() -> None:
    """Verify live app exposes all promoted models with complete fields."""
    print("[1/4] Verifying promoted models exposed via GET /api/v1/models...")
    reset_model_registry_service()
    app = create_app(configure_logging=False)
    client = TestClient(app)

    resp = client.get("/api/v1/models")
    if resp.status_code != 200:
        raise RuntimeError(f"Expected 200 from GET /api/v1/models, got {resp.status_code}: {resp.text}")

    models = resp.json()
    crops = {m["crop"]: m for m in models}
    expected_crops = {"lettuce", "spinach", "tomato"}
    if not expected_crops.issubset(crops.keys()):
        raise RuntimeError(f"Expected crops {expected_crops}, found {set(crops.keys())}")

    for crop in expected_crops:
        m = crops[crop]
        if m["status"] != "promoted":
            raise RuntimeError(f"Crop {crop} expected status 'promoted', got {m['status']}")
        if m["task"] != "classification":
            raise RuntimeError(f"Crop {crop} expected task 'classification', got {m['task']}")
        if not isinstance(m["validation_macro_f1"], float) or m["validation_macro_f1"] <= 0:
            raise RuntimeError(f"Crop {crop} invalid validation_macro_f1: {m['validation_macro_f1']}")

    if crops["tomato"]["experimental"] is not True:
        raise RuntimeError("Tomato model must have experimental=True per §24.1")
    if crops["lettuce"]["experimental"] is not False or crops["spinach"]["experimental"] is not False:
        raise RuntimeError("Lettuce and spinach models must have experimental=False")

    print("  -> Lettuce, spinach, and tomato models exposed with status=promoted and validation macro-F1.")


def verify_criterion_1_isolation() -> None:
    """Acceptance Criterion 1: Removing one artifact breaks only that crop."""
    print("[2/4] Testing Criterion 1: Removing one artifact breaks only that crop...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mock_root = tmp_path / "models"
        shutil.copytree(Path("models").resolve(), mock_root)

        # Delete lettuce artifact
        lettuce_pointer = json.loads(
            (mock_root / "manifests" / "active_lettuce_sensor.json").read_text(encoding="utf-8")
        )
        lettuce_art = mock_root / lettuce_pointer["active_artifact_path"]
        lettuce_art.unlink()

        service = ModelRegistryService(registry_root=mock_root)
        service.load_promoted_models()

        summaries = {s.crop: s for s in service.get_model_summaries()}
        if summaries["lettuce"].status != "MODEL_UNAVAILABLE":
            raise RuntimeError(f"Expected lettuce status MODEL_UNAVAILABLE, got {summaries['lettuce'].status}")

        if summaries["spinach"].status != "promoted":
            raise RuntimeError(f"Expected spinach to remain promoted, got {summaries['spinach'].status}")
        if summaries["tomato"].status != "promoted":
            raise RuntimeError(f"Expected tomato to remain promoted, got {summaries['tomato'].status}")

        # Check inference failure for lettuce
        try:
            service.get_model("lettuce", "sensor")
            raise RuntimeError("Expected ModelUnavailableError for lettuce, but call succeeded")
        except ModelUnavailableError as exc:
            if exc.code != "MODEL_UNAVAILABLE":
                raise RuntimeError(f"Expected error code MODEL_UNAVAILABLE, got {exc.code}")

        # Check inference success for spinach
        spin_obj, spin_meta = service.get_model("spinach", "sensor")
        if spin_obj is None or spin_meta.crop != "spinach":
            raise RuntimeError("Spinach model failed to load despite isolated lettuce failure")

        print("  -> Criterion 1 PASSED: Removing lettuce artifact sets it to MODEL_UNAVAILABLE while spinach and tomato operate normally.")


def verify_criterion_2_schema_mismatch() -> None:
    """Acceptance Criterion 2: Incompatible feature list is reported as SCHEMA_MISMATCH."""
    print("[3/4] Testing Criterion 2: Incompatible feature list reported as SCHEMA_MISMATCH...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mock_root = tmp_path / "models"
        shutil.copytree(Path("models").resolve(), mock_root)

        # Alter spinach ordered_features in metadata
        spinach_meta_path = mock_root / "sensor" / "spinach" / "v1.0.0" / "metadata.json"
        meta = json.loads(spinach_meta_path.read_text(encoding="utf-8"))
        meta["ordered_features"] = ["ph", "ec", "invalid_sensor_column"]
        spinach_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        service = ModelRegistryService(registry_root=mock_root)
        service.load_promoted_models()

        summaries = {s.crop: s for s in service.get_model_summaries()}
        if summaries["spinach"].status != "SCHEMA_MISMATCH":
            raise RuntimeError(f"Expected spinach status SCHEMA_MISMATCH, got {summaries['spinach'].status}")

        if summaries["lettuce"].status != "promoted":
            raise RuntimeError(f"Expected lettuce to remain promoted, got {summaries['lettuce'].status}")

        try:
            service.get_model("spinach", "sensor")
            raise RuntimeError("Expected SchemaMismatchError for spinach, but call succeeded")
        except SchemaMismatchError as exc:
            if exc.code != "SCHEMA_MISMATCH":
                raise RuntimeError(f"Expected error code SCHEMA_MISMATCH, got {exc.code}")

        # Lettuce operates normally
        let_obj, let_meta = service.get_model("lettuce", "sensor")
        if let_obj is None or let_meta.crop != "lettuce":
            raise RuntimeError("Lettuce model failed to load despite isolated spinach schema mismatch")

        print("  -> Criterion 2 PASSED: Incompatible feature list triggers SCHEMA_MISMATCH exclusively for the affected crop.")


def verify_criterion_3_health() -> None:
    """Acceptance Criterion 3: /health reflects registry state."""
    print("[4/4] Testing Criterion 3: /health reflects registry state...")
    # 1. Fully healthy
    reset_model_registry_service()
    app = create_app(configure_logging=False)
    client = TestClient(app)
    h_ok = client.get("/api/v1/health").json()
    if h_ok["status"] != "ok" or h_ok["registry"] != "ok":
        raise RuntimeError(f"Expected status=ok, registry=ok; got {h_ok}")

    # 2. Degraded state
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mock_root = tmp_path / "models"
        shutil.copytree(Path("models").resolve(), mock_root)

        lettuce_pointer = json.loads(
            (mock_root / "manifests" / "active_lettuce_sensor.json").read_text(encoding="utf-8")
        )
        (mock_root / lettuce_pointer["active_artifact_path"]).unlink()

        reset_model_registry_service()
        app_deg = create_app(configure_logging=False, model_registry_root=str(mock_root))
        c_deg = TestClient(app_deg)
        h_deg = c_deg.get("/api/v1/health").json()
        if h_deg["status"] != "degraded" or h_deg["registry"] != "degraded":
            raise RuntimeError(f"Expected status=degraded, registry=degraded; got {h_deg}")

    # 3. Empty / unavailable state
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        empty_root = tmp_path / "models"
        (empty_root / "manifests").mkdir(parents=True)

        reset_model_registry_service()
        app_unavail = create_app(configure_logging=False, model_registry_root=str(empty_root))
        c_unavail = TestClient(app_unavail)
        h_unavail = c_unavail.get("/api/v1/health").json()
        if h_unavail["status"] != "degraded" or h_unavail["registry"] != "unavailable":
            raise RuntimeError(f"Expected status=degraded, registry=unavailable; got {h_unavail}")

    print("  -> Criterion 3 PASSED: /health correctly reflects ok, degraded, and unavailable states.")


def main() -> None:
    """Run all verification checks for T6-002."""
    print("================================================================================")
    print("Verifying T6-002: Promoted Model Loading and GET /api/v1/models")
    print("================================================================================")
    try:
        verify_active_models_exposed()
        verify_criterion_1_isolation()
        verify_criterion_2_schema_mismatch()
        verify_criterion_3_health()
        print("\nSUCCESS: All T6-002 acceptance criteria verified successfully.")
    except Exception as exc:
        print(f"\nFAILURE: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
