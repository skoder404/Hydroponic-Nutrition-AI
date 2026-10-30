"""Verification script for T9-004: Health details, fallback counters, and metrics.

Acceptance Criteria:
- [x] Counters increment in a simulated failure test
- [x] Health reflects a stopped DB
- [x] No sensitive data in output
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from backend.app.core.device import SimulatedCudaOOMError, get_device_manager
from backend.app.core.telemetry import get_fallback_tracker
from backend.app.main import create_app
from backend.app.schemas.recommendation import RecommendationRead
from backend.app.services.image_service import ImageService
from backend.app.services.recommendation_service import GeminiRecommendationProvider


def verify_t9_004() -> None:
    print("=== T9-004 Health Details and Fallback Counters Verification ===")
    tracker = get_fallback_tracker()
    tracker.reset()

    app = create_app(configure_logging=False)
    client = TestClient(app)

    # --------------------------------------------------------------------------
    # Check 1: Health reflects a stopped / unavailable DB
    # --------------------------------------------------------------------------
    print("\n[Check 1] Verifying health reflects stopped database...")
    with patch("backend.app.api.v1.health.get_engine") as mock_get_engine:
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = ConnectionRefusedError(
            "Connection refused to PostgreSQL host on port 5432"
        )
        mock_get_engine.return_value = mock_engine

        resp = client.get("/api/v1/health")
        assert resp.status_code == 200, (
            f"Expected 200 from /health, got {resp.status_code}"
        )
        data = resp.json()
        assert data["db"] == "unavailable", (
            f"Expected db='unavailable', got {data['db']}"
        )
        assert data["status"] == "degraded", (
            f"Expected status='degraded', got {data['status']}"
        )
        print(
            "  ✓ Stopped DB correctly reported as db='unavailable' and status='degraded'"
        )

    # --------------------------------------------------------------------------
    # Check 2: Health reflects connected DB
    # --------------------------------------------------------------------------
    print("\n[Check 2] Verifying health reflects connected database...")
    with patch("backend.app.api.v1.health.get_engine") as mock_get_engine:
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_get_engine.return_value = mock_engine

        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["db"] == "connected", f"Expected db='connected', got {data['db']}"
        assert data["status"] == "ok", f"Expected status='ok', got {data['status']}"
        print("  ✓ Connected DB correctly reported as db='connected' and status='ok'")

    # --------------------------------------------------------------------------
    # Check 3: Counters increment in simulated failures
    # --------------------------------------------------------------------------
    print("\n[Check 3] Verifying fallback counters increment in simulated failures...")
    assert tracker.get_counters() == {
        "gemini_to_rules": 0,
        "gpu_to_cpu": 0,
        "segmentation_failure": 0,
    }

    # Simulate Gemini failure
    provider = GeminiRecommendationProvider(
        api_key="mock-key-for-test",
        model_name="gemini-2.5-flash",
    )
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_post.return_value = mock_resp

        rec = provider.recommend(
            crop="lettuce",
            predicted_class="NitrogenDeficiency",
            modality="sensor",
        )
        assert isinstance(rec, RecommendationRead)
        assert rec.source == "rules"
    assert tracker.get_counters()["gemini_to_rules"] == 1
    print("  ✓ gemini_to_rules counter incremented on Gemini fallback")

    # Simulate GPU OOM
    dm = get_device_manager()
    try:
        with dm.retry_on_oom():
            raise SimulatedCudaOOMError("OOM during inference")
    except SimulatedCudaOOMError:
        pass
    assert tracker.get_counters()["gpu_to_cpu"] == 1
    print("  ✓ gpu_to_cpu counter incremented on CUDA OOM fallback")

    # Simulate Segmentation failure
    image_service = ImageService()
    mock_empty = MagicMock()
    mock_empty.leaves = []
    mock_empty.rejected = [MagicMock(candidate_id="leaf_0", reason="blurred")]
    import numpy as np

    blank_img = np.zeros((64, 64, 3), dtype=np.uint8)

    with patch(
        "ml.segmentation.leaf_extraction.extract_leaves", return_value=mock_empty
    ):
        from backend.app.core.errors import SegmentationFailedError

        try:
            image_service.predict(image=blank_img, crop="spinach")
        except SegmentationFailedError:
            pass
    assert tracker.get_counters()["segmentation_failure"] == 1
    print("  ✓ segmentation_failure counter incremented on leaf extraction failure")

    # Check /health and /metrics report counters
    health_resp = client.get("/api/v1/health")
    counters = health_resp.json()["fallback_counters"]
    assert counters["gemini_to_rules"] == 1
    assert counters["gpu_to_cpu"] == 1
    assert counters["segmentation_failure"] == 1

    metrics_resp = client.get("/api/v1/metrics")
    assert "gemini_to_rules_total 1" in metrics_resp.text
    assert "gpu_to_cpu_total 1" in metrics_resp.text
    assert "segmentation_failure_total 1" in metrics_resp.text
    print("  ✓ /health and /metrics endpoints reflect updated counters")

    # --------------------------------------------------------------------------
    # Check 4: No sensitive data in output
    # --------------------------------------------------------------------------
    print("\n[Check 4] Scanning responses for sensitive data leaks...")
    responses_to_scan = [
        client.get("/api/v1/health").text,
        client.get("/api/v1/health/details").text,
        client.get("/api/v1/metrics").text,
        client.get("/api/v1/metrics?format=json").text,
    ]
    sensitive_patterns = [
        re.compile(r"postgresql://", re.IGNORECASE),
        re.compile(r"password\s*[:=]", re.IGNORECASE),
        re.compile(r"gemini[_-]?api[_-]?key", re.IGNORECASE),
        re.compile(r"secret", re.IGNORECASE),
        re.compile(r"AIzaSy[A-Za-z0-9_-]{33}"),
        re.compile(r"bearer\s+[A-Za-z0-9_-]+", re.IGNORECASE),
    ]

    for body in responses_to_scan:
        for pat in sensitive_patterns:
            assert pat.search(body) is None, (
                f"Sensitive pattern {pat.pattern} found in: {body}"
            )
    print("  ✓ Zero sensitive data leaks found across all health and metrics endpoints")

    print("\n=== All T9-004 verification checks passed successfully! ===")


if __name__ == "__main__":
    verify_t9_004()
