"""Unit tests for FastAPI app factory, /health, request-ID middleware, and JSON logging (T1-009)."""

from __future__ import annotations

import io
import json
import logging

from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.core.device import DeviceManager, get_device_manager
from backend.app.core.logging import setup_logging
from backend.app.main import create_app


def test_health_endpoint_contract():
    """Verify GET /api/v1/health returns 200 and documented fields."""
    app = create_app(configure_logging=False)
    client = TestClient(app)

    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert data["db"] in ("connected", "unavailable", "unknown")
    assert data["registry"] in ("ok", "degraded", "unknown")
    assert "gpu" in data
    assert "fallback_counters" in data

    expected_device = get_device_manager().get_device_actually_used()
    assert data["device"] == expected_device

    # Check header
    assert "x-request-id" in [k.lower() for k in response.headers]


def test_device_manager_force_cpu():
    """Verify DeviceManager properly honors force_cpu."""
    cpu_manager = DeviceManager(force_cpu=True)
    assert cpu_manager.get_device_str() == "cpu"
    assert cpu_manager.get_device_name() == "CPU"
    status = cpu_manager.get_status()
    assert status["force_cpu"] is True
    assert status["target_device"] == "cpu"


def test_json_logging_and_request_id_middleware():
    """Verify every log line is JSON containing request_id."""
    log_stream = io.StringIO()
    setup_logging(log_level="INFO", stream=log_stream)

    app = create_app(configure_logging=False)
    client = TestClient(app)

    custom_id = "test-custom-unit-id-999"
    response = client.get("/api/v1/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_id

    # Emit a lifecycle log
    logger = logging.getLogger("unit_test")
    logger.info("Lifecycle check")

    log_output = log_stream.getvalue().strip()
    lines = [line for line in log_output.split("\n") if line.strip()]
    assert len(lines) > 0

    found_custom_id = False
    for line in lines:
        entry = json.loads(line)
        assert "event" in entry
        assert "request_id" in entry
        assert "level" in entry
        assert "timestamp" in entry
        if entry.get("request_id") == custom_id:
            found_custom_id = True

    assert found_custom_id, "Injected X-Request-ID was not recorded in logs"


def test_secret_redaction_in_logs():
    """Verify no secret appears in plaintext in logs."""
    log_stream = io.StringIO()
    setup_logging(log_level="INFO", stream=log_stream)

    logger = logging.getLogger("security_unit_test")
    raw_secret = "super_secret_database_password_42"
    raw_key = "AIzaSyDtestingGeminiApiKey12345678900"
    raw_bearer = "bearer_token_abc_xyz_9999"

    logger.info(
        f"Database connection string: postgresql://postgres:{raw_secret}@localhost:5432/db"
    )
    logger.info(f"Loaded gemini with api_key={raw_key}")
    logger.info(f"Authorization: Bearer {raw_bearer}")

    log_output = log_stream.getvalue()
    assert raw_secret not in log_output
    assert raw_key not in log_output
    assert raw_bearer not in log_output
    assert "***" in log_output


def test_settings_repr_redaction():
    """Verify settings repr redacts database password and secrets."""
    settings = get_settings()
    rendered = repr(settings)
    assert not any(f":{pwd}@" in rendered for pwd in ["postgres", "password", "secret"])
