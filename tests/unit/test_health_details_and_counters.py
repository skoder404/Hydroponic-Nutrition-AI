"""Unit tests for health diagnostics, fallback counters, and metrics (T9-004).

Acceptance Criteria verified:
- [x] Counters increment in a simulated failure test (Gemini, GPU OOM, Segmentation)
- [x] Health reflects a stopped DB (db='unavailable', status='degraded')
- [x] No sensitive data in output (credentials, tokens, connection strings redacted)
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.core.device import SimulatedCudaOOMError, get_device_manager
from backend.app.core.telemetry import get_fallback_tracker
from backend.app.main import create_app
from backend.app.schemas.recommendation import RecommendationRead
from backend.app.services.image_service import ImageService
from backend.app.services.recommendation_service import GeminiRecommendationProvider


@pytest.fixture(autouse=True)
def reset_telemetry_tracker():
    """Reset counters before and after each test."""
    tracker = get_fallback_tracker()
    tracker.reset()
    yield
    tracker.reset()


@pytest.fixture
def client() -> TestClient:
    """TestClient instance for FastAPI application."""
    app = create_app(configure_logging=False)
    return TestClient(app)


def test_health_reflects_stopped_db(client: TestClient):
    """AC 2: Health endpoint accurately reflects an unavailable or stopped database."""
    with patch("backend.app.api.v1.health.get_engine") as mock_get_engine:
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = ConnectionRefusedError(
            "Connection to PostgreSQL server on 127.0.0.1:5432 refused."
        )
        mock_get_engine.return_value = mock_engine

        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()

        # Database must show unavailable and overall status degraded
        assert data["db"] == "unavailable"
        assert data["status"] == "degraded"
        assert "device" in data
        assert "registry" in data
        assert "gpu" in data
        assert "fallback_counters" in data


def test_health_reflects_connected_db(client: TestClient):
    """Verify health returns status 'ok' and db 'connected' when database connection succeeds."""
    with patch("backend.app.api.v1.health.get_engine") as mock_get_engine:
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_get_engine.return_value = mock_engine

        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()

        assert data["db"] == "connected"
        assert data["status"] == "ok"


def test_counters_increment_in_simulated_failures(client: TestClient):
    """AC 1: Telemetry counters increment when simulated failures occur."""
    tracker = get_fallback_tracker()
    assert tracker.get_counters()["gemini_to_rules"] == 0
    assert tracker.get_counters()["gpu_to_cpu"] == 0
    assert tracker.get_counters()["segmentation_failure"] == 0

    # 1. Simulate Gemini failure -> gemini_to_rules increments
    gemini_provider = GeminiRecommendationProvider(
        api_key="simulated-mock-key-for-test",
        model_name="gemini-2.5-flash",
    )
    with patch("requests.post") as mock_post:
        # Simulate HTTP 503 Service Unavailable
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "Service Overloaded"
        mock_post.return_value = mock_resp

        rec = gemini_provider.recommend(
            crop="lettuce",
            predicted_class="NitrogenDeficiency",
            modality="sensor",
        )
        # Verify fallback occurred to deterministic rules
        assert isinstance(rec, RecommendationRead)
        assert rec.source == "rules"
        assert tracker.get_counters()["gemini_to_rules"] == 1

    # 2. Simulate GPU OOM -> gpu_to_cpu increments
    dm = get_device_manager()
    try:
        with dm.retry_on_oom():
            raise SimulatedCudaOOMError("Simulated CUDA OOM in unit test")
    except SimulatedCudaOOMError:
        pass
    assert tracker.get_counters()["gpu_to_cpu"] == 1

    # 3. Simulate Segmentation failure -> segmentation_failure increments
    image_service = ImageService()
    # Mock extract_leaves returning 0 leaves
    mock_empty_result = MagicMock()
    mock_empty_result.leaves = []
    mock_rej = MagicMock()
    mock_rej.candidate_id = "candidate_0"
    mock_rej.reason = "Area too small"
    mock_empty_result.rejected = [mock_rej]

    import numpy as np

    blank_img = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch(
        "ml.segmentation.leaf_extraction.extract_leaves", return_value=mock_empty_result
    ):
        from backend.app.core.errors import SegmentationFailedError

        with pytest.raises(SegmentationFailedError):
            image_service.predict(
                image=blank_img,
                crop="spinach",
            )
    assert tracker.get_counters()["segmentation_failure"] == 1

    # 4. Verify /health and /health/details reflect updated counters
    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == 200
    counters_data = health_resp.json()["fallback_counters"]
    assert counters_data["gemini_to_rules"] == 1
    assert counters_data["gpu_to_cpu"] == 1
    assert counters_data["segmentation_failure"] == 1

    # 5. Verify /metrics exposes counters in Prometheus format
    metrics_resp = client.get("/api/v1/metrics")
    assert metrics_resp.status_code == 200
    metrics_text = metrics_resp.text
    assert "gemini_to_rules_total 1" in metrics_text
    assert "gpu_to_cpu_total 1" in metrics_text
    assert "segmentation_failure_total 1" in metrics_text


def test_no_sensitive_data_in_output(client: TestClient):
    """AC 3: Ensure no sensitive information is leaked in /health or /metrics endpoints."""
    # Test GET /health
    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == 200
    raw_health = health_resp.text

    # Test GET /health/details
    details_resp = client.get("/api/v1/health/details")
    assert details_resp.status_code == 200
    raw_details = details_resp.text

    # Test GET /metrics
    metrics_resp = client.get("/api/v1/metrics")
    assert metrics_resp.status_code == 200
    raw_metrics = metrics_resp.text

    all_outputs = [raw_health, raw_details, raw_metrics]

    # Sensitive patterns that should NEVER appear in output
    forbidden_patterns = [
        re.compile(r"postgresql://", re.IGNORECASE),
        re.compile(r"password\s*[:=]", re.IGNORECASE),
        re.compile(r"gemini[_-]?api[_-]?key", re.IGNORECASE),
        re.compile(r"secret", re.IGNORECASE),
        re.compile(r"AIzaSy[A-Za-z0-9_-]{33}"),  # Google API key format
        re.compile(r"token\s*[:=]", re.IGNORECASE),
    ]

    for output in all_outputs:
        for pattern in forbidden_patterns:
            match = pattern.search(output)
            assert match is None, (
                f"Sensitive data pattern '{pattern.pattern}' detected in response: {output}"
            )


def test_metrics_json_and_prometheus_negotiation(client: TestClient):
    """Verify format negotiation on /metrics endpoint."""
    tracker = get_fallback_tracker()
    tracker.increment_gemini_to_rules("test_reason")

    # Prometheus default format
    resp_prom = client.get("/api/v1/metrics")
    assert resp_prom.status_code == 200
    assert "text/plain" in resp_prom.headers["content-type"]
    assert "gemini_to_rules_total 1" in resp_prom.text

    # JSON format via query param
    resp_json = client.get("/api/v1/metrics?format=json")
    assert resp_json.status_code == 200
    assert "application/json" in resp_json.headers["content-type"]
    json_data = resp_json.json()
    assert json_data["fallback_counters"]["gemini_to_rules"] == 1

    # JSON format via Accept header
    resp_header = client.get("/api/v1/metrics", headers={"Accept": "application/json"})
    assert resp_header.status_code == 200
    assert "application/json" in resp_header.headers["content-type"]
    assert resp_header.json()["fallback_counters"]["gemini_to_rules"] == 1


def test_structured_logging_on_fallback(caplog):
    """Verify structured log records are emitted when fallback occurs."""
    import logging

    tracker = get_fallback_tracker()
    with caplog.at_level(logging.WARNING):
        tracker.increment_gemini_to_rules(reason="RateLimitExceeded")
        tracker.increment_gpu_to_cpu(reason="OOMDetected")
        tracker.increment_segmentation_failure(reason="NoMasksFound")

    log_text = caplog.text
    assert "gemini_to_rules" in log_text
    assert "RateLimitExceeded" in log_text
    assert "gpu_to_cpu" in log_text
    assert "OOMDetected" in log_text
    assert "segmentation_failure" in log_text
    assert "NoMasksFound" in log_text
