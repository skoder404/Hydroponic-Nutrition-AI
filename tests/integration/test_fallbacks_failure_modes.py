"""Integration tests: fallbacks and failure modes (T8-006).

Acceptance Criteria:
- [x] Each failure returns the documented code
- [x] DB outage never reports success
- [x] Gemini case is skipped with a clear marker until Phase 10

Failure modes covered:
1. Corrupt/missing model artifact -> HTTP 503 MODEL_UNAVAILABLE
2. Corrupt image upload -> HTTP 415 UNSUPPORTED_MEDIA
3. Database outage on save -> HTTP 503 PERSISTENCE_FAILED ('Result was not saved')
4. Device unavailable / stale -> HTTP 503 DEVICE_UNAVAILABLE to manual path
5. GPU unavailable (FORCE_CPU) & simulated OOM -> Fallback to CPU with honest device reporting
6. Gemini unavailable -> rules fallback, with full Gemini live API test skipped until Phase 10
"""

from __future__ import annotations

import io
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.exc import OperationalError

from backend.app.core.device_manager import (
    DeviceManager,
    SimulatedCudaOOMError,
    get_device_manager,
)
from backend.app.core.rate_limit import get_rate_limiter
from backend.app.database.session import get_db
from backend.app.devices.adapter import (
    MockAdapter,
    reset_device_adapter,
)
from backend.app.main import create_app
from backend.app.services.image_service import reset_image_service
from backend.app.services.model_service import (
    reset_model_registry_service,
)
from backend.app.services.recommendation_service import (
    RecommendationService,
    reset_recommendation_service,
)
from backend.app.services.sensor_service import reset_sensor_service


def _create_synthetic_plant_image_bytes(format_name: str = "JPEG") -> bytes:
    """Generate in-memory byte stream of a synthetic plant image."""
    import cv2
    import numpy as np

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (60, 60), 30, (25, 200, 25), -1)
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format=format_name)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def clean_environment():
    """Reset singletons before and after each test."""
    reset_model_registry_service()
    reset_sensor_service()
    reset_image_service()
    reset_recommendation_service()
    reset_device_adapter()
    get_device_manager.cache_clear()
    get_rate_limiter().reset()
    yield
    reset_model_registry_service()
    reset_sensor_service()
    reset_image_service()
    reset_recommendation_service()
    reset_device_adapter()
    get_device_manager.cache_clear()
    get_rate_limiter().reset()


@pytest.fixture
def client():
    """Create standard FastAPI test client."""
    app = create_app(configure_logging=False)
    with TestClient(app) as test_client:
        yield test_client


# -----------------------------------------------------------------------------
# 1. Corrupt / Missing Artifact Failures (Documented Code: MODEL_UNAVAILABLE)
# -----------------------------------------------------------------------------


def test_missing_model_artifact_returns_model_unavailable(client: TestClient):
    """Failure Mode 1a: Requesting prediction when model is missing returns 503 MODEL_UNAVAILABLE."""
    img_bytes = _create_synthetic_plant_image_bytes()
    files = {"file": ("plant.jpg", img_bytes, "image/jpeg")}
    data = {
        "crop": "lettuce",
        "input_mode": "plant",
        "model_family": "nonexistent_model_family",
    }
    response = client.post("/api/v1/predict/image", data=data, files=files)
    assert response.status_code == 503
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "MODEL_UNAVAILABLE"
    assert body["error"]["retryable"] is False
    assert "nonexistent_model_family" in body["error"]["message"]


def test_corrupt_image_upload_returns_unsupported_media(client: TestClient):
    """Failure Mode 1b: Uploading a corrupted/non-image file returns 415 UNSUPPORTED_MEDIA."""
    corrupt_bytes = b"not a real image binary data at all"
    files = {"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")}
    data = {"crop": "lettuce", "input_mode": "plant"}

    response = client.post("/api/v1/predict/image", data=data, files=files)
    assert response.status_code == 415
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "UNSUPPORTED_MEDIA"


# -----------------------------------------------------------------------------
# 2. Database Outage on Save (AC: DB outage never reports success)
# -----------------------------------------------------------------------------


def test_db_outage_never_reports_success_and_returns_persistence_failed():
    """Failure Mode 2: DB outage raises PersistenceFailedError ('Result was not saved') and never reports success."""
    app = create_app(configure_logging=False)

    # Simulate database outage by raising OperationalError on DB session operations
    def broken_db_session():
        mock_session = MagicMock()
        mock_session.add.side_effect = OperationalError(
            statement="INSERT INTO records",
            params={},
            orig=Exception("connection refused: database server down"),
        )
        mock_session.commit.side_effect = OperationalError(
            statement="COMMIT",
            params={},
            orig=Exception("connection refused: database server down"),
        )
        yield mock_session

    app.dependency_overrides[get_db] = broken_db_session

    with TestClient(app) as test_client:
        record_payload = {
            "crop": "lettuce",
            "mode": "sensor",
            "observed_at": "2026-10-29T08:00:00Z",
            "observed_tz": "UTC",
            "source": "manual",
            "inputs": {"ph": 6.2, "ec": 1.6},
            "predictions": [
                {
                    "crop": "lettuce",
                    "modality": "sensor",
                    "predicted_class": "Normal",
                    "display_label": "Optimal",
                    "confidence": 0.95,
                    "confidence_semantics": "multinomial_probability",
                    "uncertain": False,
                    "model_id": "lettuce_sensor_dt",
                    "model_version": "1.0.0",
                    "timestamp": "2026-10-29T08:00:00Z",
                    "provenance": "manual",
                }
            ],
        }

        response = test_client.post("/api/v1/history", json=record_payload)

        # Acceptance Criterion 2: DB outage never reports success
        assert response.status_code != 200
        assert response.status_code != 201
        assert response.status_code == 503

        body = response.json()
        assert "error" in body
        assert body["error"]["code"] == "PERSISTENCE_FAILED"
        # Must clearly indicate that the result was not saved
        assert "not saved" in body["error"]["message"].lower()

    app.dependency_overrides.clear()


# -----------------------------------------------------------------------------
# 3. Device Unavailable / Stale to Manual Path
# -----------------------------------------------------------------------------


def test_device_unavailable_returns_503_and_routes_to_manual_path():
    """Failure Mode 3: Disconnected campus device returns 503 DEVICE_UNAVAILABLE."""
    app = create_app(configure_logging=False)

    # Inject mock adapter with disconnected state
    from backend.app.api.v1.device import get_device_adapter

    disconnected_adapter = MockAdapter()
    disconnected_adapter.set_simulate_disconnected(True)
    app.dependency_overrides[get_device_adapter] = lambda: disconnected_adapter

    with TestClient(app) as test_client:
        # Status endpoint shows status='disconnected'
        status_resp = test_client.get("/api/v1/device/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "disconnected"

        # Attempting to read telemetry when disconnected returns 503 DEVICE_UNAVAILABLE
        sensor_resp = test_client.get("/api/v1/device/sensors/latest?crop=lettuce")
        assert sensor_resp.status_code == 503
        body = sensor_resp.json()
        assert "error" in body
        assert body["error"]["code"] == "DEVICE_UNAVAILABLE"
        assert body["error"]["retryable"] is True

        # System falls back to manual entry: user supplies readings directly
        manual_predict_resp = test_client.post(
            "/api/v1/predict/sensor",
            json={
                "crop": "lettuce",
                "readings": {
                    "ph": 6.2,
                    "ec": 1.6,
                    "water_temp": 21.0,
                    "humidity": 65.0,
                    "air_temp": 23.5,
                },
                "provenance": "manual",
            },
        )
        assert manual_predict_resp.status_code == 200
        assert manual_predict_resp.json()["provenance"] == "manual"

    app.dependency_overrides.clear()


# -----------------------------------------------------------------------------
# 4. GPU Unavailable (FORCE_CPU) and Simulated OOM
# -----------------------------------------------------------------------------


def test_force_cpu_fallback_and_device_honesty():
    """Failure Mode 4a: Setting FORCE_CPU forces CPU execution and honestly reports device."""
    with patch.dict(os.environ, {"FORCE_CPU": "1"}):
        dm = DeviceManager()
        assert dm.force_cpu is True
        assert dm.get_target_device() == "cpu"
        assert dm.get_device_name() == "CPU"
        assert dm.get_device_actually_used() == "cpu"


def test_simulated_oom_ladder_falls_back_to_cpu():
    """Failure Mode 4b: Simulated CUDA OOM triggers smaller batch and then CPU fallback."""
    dm = DeviceManager(force_cpu=False)

    with patch.object(dm, "is_cuda_available", return_value=True):
        attempts = []

        def memory_heavy_work(device: str, batch: int):
            attempts.append((device, batch))
            if "cuda" in device:
                raise SimulatedCudaOOMError("Simulated out of memory error on GPU")
            return f"Completed on {device} with batch size {batch}"

        result, meta = dm.execute_with_retry(memory_heavy_work, initial_batch_size=4)

        assert meta["fallback_to_cpu"] is True
        assert meta["device_used"] == "cpu"
        assert result == "Completed on cpu with batch size 2"
        # Sequence: cuda:0 (batch 4) -> cuda:0 (batch 2) -> cpu (batch 2)
        assert len(attempts) == 3
        assert attempts[-1] == ("cpu", 2)


# -----------------------------------------------------------------------------
# 5. Gemini Unavailable -> Rules Fallback & Phase 10 Skip Marker
# -----------------------------------------------------------------------------


def test_gemini_slot_falls_back_to_rules_when_unconfigured():
    """Failure Mode 5a: Gemini slot without API key falls back to rules with source='rules'."""
    rec_service = RecommendationService(provider="gemini")
    rec = rec_service.recommend(
        crop="lettuce",
        predicted_class="NitrogenDeficiency",
        modality="sensor",
        confidence=0.88,
    )
    assert rec.source == "rules"
    assert rec.rules_version.startswith("v")
    assert "content" in rec.model_dump()


@pytest.mark.skip(
    reason="Gemini LLM narrative recommendations scheduled for Phase 10 (T10-001)"
)
def test_gemini_live_api_phase_10_marker():
    """AC 3: Gemini case is skipped with a clear marker until Phase 10."""
