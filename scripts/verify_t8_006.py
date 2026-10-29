"""Verification script for T8-006: Integration tests: fallbacks and failure modes.

Acceptance Criteria:
- Each failure returns the documented code
- DB outage never reports success
- Gemini case is skipped with a clear marker until Phase 10
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.exc import OperationalError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.device_manager import get_device_manager
from backend.app.core.rate_limit import get_rate_limiter
from backend.app.database.session import get_db
from backend.app.devices.adapter import MockAdapter, reset_device_adapter
from backend.app.main import create_app
from backend.app.services.image_service import reset_image_service
from backend.app.services.model_service import reset_model_registry_service
from backend.app.services.recommendation_service import reset_recommendation_service
from backend.app.services.sensor_service import reset_sensor_service


def create_sample_image() -> bytes:
    import cv2
    import numpy as np

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (60, 60), 30, (25, 200, 25), -1)
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return buf.getvalue()


def main() -> int:
    print("Verifying T8-006: Fallbacks and failure modes...")

    reset_model_registry_service()
    reset_sensor_service()
    reset_image_service()
    reset_recommendation_service()
    reset_device_adapter()
    get_device_manager.cache_clear()
    get_rate_limiter().reset()

    app = create_app(configure_logging=False)
    client = TestClient(app)

    # 1. AC 1: Each failure returns the documented code
    # 1a. Missing/unsupported model family -> 503 MODEL_UNAVAILABLE
    img_bytes = create_sample_image()
    resp_unavail = client.post(
        "/api/v1/predict/image",
        data={
            "crop": "lettuce",
            "input_mode": "plant",
            "model_family": "nonexistent_family",
        },
        files={"file": ("plant.jpg", img_bytes, "image/jpeg")},
    )
    assert resp_unavail.status_code == 503, (
        f"Expected 503, got {resp_unavail.status_code}"
    )
    assert resp_unavail.json()["error"]["code"] == "MODEL_UNAVAILABLE"
    print("  [PASS] AC 1a: Missing model family returns 503 MODEL_UNAVAILABLE")

    # 1b. Corrupt upload -> 415 UNSUPPORTED_MEDIA
    resp_corrupt = client.post(
        "/api/v1/predict/image",
        data={"crop": "lettuce", "input_mode": "plant"},
        files={"file": ("corrupt.jpg", b"fake binary data", "image/jpeg")},
    )
    assert resp_corrupt.status_code == 415, (
        f"Expected 415, got {resp_corrupt.status_code}"
    )
    assert resp_corrupt.json()["error"]["code"] == "UNSUPPORTED_MEDIA"
    print("  [PASS] AC 1b: Corrupt image returns 415 UNSUPPORTED_MEDIA")

    # 2. AC 2: DB outage never reports success
    def broken_db():
        mock_session = MagicMock()
        mock_session.add.side_effect = OperationalError(
            statement="INSERT", params={}, orig=Exception("DB down")
        )
        mock_session.commit.side_effect = OperationalError(
            statement="COMMIT", params={}, orig=Exception("DB down")
        )
        yield mock_session

    app.dependency_overrides[get_db] = broken_db
    resp_db = client.post(
        "/api/v1/history",
        json={
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
        },
    )
    assert resp_db.status_code != 200 and resp_db.status_code != 201, (
        f"DB outage must never report success, got {resp_db.status_code}"
    )
    assert resp_db.status_code == 503
    assert resp_db.json()["error"]["code"] == "PERSISTENCE_FAILED"
    assert "not saved" in resp_db.json()["error"]["message"].lower()
    app.dependency_overrides.clear()
    print(
        "  [PASS] AC 2: DB outage never reports success (returns 503 PERSISTENCE_FAILED with 'not saved')"
    )

    # 3. AC 1c: Device disconnected -> 503 DEVICE_UNAVAILABLE
    from backend.app.api.v1.device import get_device_adapter

    disconnected = MockAdapter()
    disconnected.set_simulate_disconnected(True)
    app.dependency_overrides[get_device_adapter] = lambda: disconnected
    resp_dev = client.get("/api/v1/device/sensors/latest?crop=lettuce")
    assert resp_dev.status_code == 503, f"Expected 503, got {resp_dev.status_code}"
    assert resp_dev.json()["error"]["code"] == "DEVICE_UNAVAILABLE"
    app.dependency_overrides.clear()
    print("  [PASS] AC 1c: Device disconnected returns 503 DEVICE_UNAVAILABLE")

    # 4. AC 3: Gemini case is skipped with a clear marker until Phase 10
    from tests.integration.test_fallbacks_failure_modes import (
        test_gemini_live_api_phase_10_marker,
    )

    skip_marker = getattr(test_gemini_live_api_phase_10_marker, "pytestmark", [])
    has_skip = any(m.name == "skip" for m in skip_marker)
    assert has_skip, (
        "Expected test_gemini_live_api_phase_10_marker to be marked with pytest.mark.skip"
    )
    print("  [PASS] AC 3: Gemini case is skipped with a clear marker until Phase 10")

    print("ALL ACs for T8-006 PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
