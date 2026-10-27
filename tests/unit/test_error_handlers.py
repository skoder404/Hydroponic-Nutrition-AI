"""Unit tests for the shared error envelope and exception handlers (T3-004).

Acceptance Criteria:
1. Each code returns the documented HTTP status.
2. Unhandled error response contains no stack trace or path.
3. retryable is true only for segmentation failure and device states.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from backend.app.core.errors import (
    AppException,
    DeviceUnavailableError,
    InvalidInputError,
    MissingFieldError,
    ModelUnavailableError,
    PayloadTooLargeError,
    PersistenceFailedError,
    SchemaMismatchError,
    SegmentationFailedError,
    UnsupportedMediaError,
    register_error_handlers,
)

# Documented code to HTTP status and retryable expectation
DOCUMENTED_ERROR_SPECS: list[tuple[type[AppException], str, int, bool]] = [
    (InvalidInputError, "INVALID_INPUT", 422, False),
    (MissingFieldError, "MISSING_FIELD", 422, False),
    (UnsupportedMediaError, "UNSUPPORTED_MEDIA", 415, False),
    (PayloadTooLargeError, "PAYLOAD_TOO_LARGE", 413, False),
    (SegmentationFailedError, "SEGMENTATION_FAILED", 422, True),
    (SchemaMismatchError, "SCHEMA_MISMATCH", 409, False),
    (ModelUnavailableError, "MODEL_UNAVAILABLE", 503, False),
    (DeviceUnavailableError, "DEVICE_UNAVAILABLE", 503, True),
    (PersistenceFailedError, "PERSISTENCE_FAILED", 503, False),
]


class SamplePayload(BaseModel):
    crop: str
    reading: float


@pytest.fixture
def error_test_app() -> FastAPI:
    """Create test application with all error handlers registered."""
    app = FastAPI()
    register_error_handlers(app)

    # Routes to trigger specific typed exceptions
    @app.get("/trigger/unsupported-media")
    def trigger_unsupported_media() -> None:
        raise UnsupportedMediaError("Unsupported media type.")

    @app.get("/trigger/payload-too-large")
    def trigger_payload_too_large() -> None:
        raise PayloadTooLargeError("Payload too large.")

    @app.get("/trigger/invalid-input")
    def trigger_invalid_input() -> None:
        raise InvalidInputError("Invalid input provided.")

    @app.get("/trigger/missing-field")
    def trigger_missing_field() -> None:
        raise MissingFieldError("Field 'ph' is missing.")

    @app.get("/trigger/segmentation-failed")
    def trigger_segmentation_failed() -> None:
        raise SegmentationFailedError("Could not isolate leaf mask.")

    @app.get("/trigger/schema-mismatch")
    def trigger_schema_mismatch() -> None:
        raise SchemaMismatchError("Active model expects 5 features, got 4.")

    @app.get("/trigger/model-unavailable")
    def trigger_model_unavailable() -> None:
        raise ModelUnavailableError("No promoted tomato model found.")

    @app.get("/trigger/device-unavailable")
    def trigger_device_unavailable() -> None:
        raise DeviceUnavailableError("Campus sensor stream timed out.")

    @app.get("/trigger/persistence-failed")
    def trigger_persistence_failed() -> None:
        raise PersistenceFailedError("Failed to persist record.")

    @app.get("/trigger/unhandled")
    def trigger_unhandled() -> None:
        raise RuntimeError(
            "Fatal database internal error at /secret/db/connection.py line 89!"
        )

    @app.post("/trigger/validation")
    def trigger_validation(payload: SamplePayload) -> dict[str, str]:
        return {"crop": payload.crop}

    return app


# ------------------------------------------------------------------------------
# Criterion 1: Each Code Returns the Documented HTTP Status
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_cls,expected_code,expected_status,expected_retryable", DOCUMENTED_ERROR_SPECS
)
def test_each_typed_exception_class_specification(
    exc_cls: Any,
    expected_code: str,
    expected_status: int,
    expected_retryable: bool,
) -> None:
    """Verify class-level defaults match TECH_RULES specification."""
    instance = exc_cls()
    assert instance.code == expected_code
    assert instance.status_code == expected_status
    assert instance.retryable is expected_retryable

    envelope = instance.to_envelope()
    assert "error" in envelope
    assert envelope["error"]["code"] == expected_code
    assert envelope["error"]["retryable"] is expected_retryable


def test_each_code_returns_documented_http_status_via_api(
    error_test_app: FastAPI,
) -> None:
    """Acceptance Criterion 1: Verify all 9 endpoints return exact documented HTTP status codes."""
    client = TestClient(error_test_app)

    endpoint_mappings = [
        ("/trigger/invalid-input", "INVALID_INPUT", 422),
        ("/trigger/missing-field", "MISSING_FIELD", 422),
        ("/trigger/unsupported-media", "UNSUPPORTED_MEDIA", 415),
        ("/trigger/payload-too-large", "PAYLOAD_TOO_LARGE", 413),
        ("/trigger/segmentation-failed", "SEGMENTATION_FAILED", 422),
        ("/trigger/schema-mismatch", "SCHEMA_MISMATCH", 409),
        ("/trigger/model-unavailable", "MODEL_UNAVAILABLE", 503),
        ("/trigger/device-unavailable", "DEVICE_UNAVAILABLE", 503),
        ("/trigger/persistence-failed", "PERSISTENCE_FAILED", 503),
    ]

    for endpoint, expected_code, expected_status in endpoint_mappings:
        res = client.get(endpoint)
        assert res.status_code == expected_status, (
            f"Endpoint {endpoint} returned {res.status_code}, expected {expected_status}"
        )
        data = res.json()
        assert "error" in data
        assert data["error"]["code"] == expected_code


def test_request_validation_error_returns_422_with_envelope(
    error_test_app: FastAPI,
) -> None:
    """Request parameter validation failures return HTTP 422 in the unified envelope."""
    client = TestClient(error_test_app)

    # Missing required field
    res_missing = client.post("/trigger/validation", json={"crop": "lettuce"})
    assert res_missing.status_code == 422
    data_missing = res_missing.json()
    assert data_missing["error"]["code"] in ("MISSING_FIELD", "INVALID_INPUT")
    assert data_missing["error"]["retryable"] is False

    # Invalid type
    res_type = client.post(
        "/trigger/validation", json={"crop": "lettuce", "reading": "not-a-number"}
    )
    assert res_type.status_code == 422
    data_type = res_type.json()
    assert data_type["error"]["code"] == "INVALID_INPUT"
    assert data_type["error"]["retryable"] is False


# ------------------------------------------------------------------------------
# Criterion 2: Unhandled Error Response Contains No Stack Trace or Path
# ------------------------------------------------------------------------------


def test_unhandled_error_response_contains_no_stack_trace_or_path(
    error_test_app: FastAPI,
) -> None:
    """Acceptance Criterion 2: Unhandled exception response is generic and leaks zero paths or traces."""
    client = TestClient(error_test_app, raise_server_exceptions=False)

    res = client.get("/trigger/unhandled")
    assert res.status_code == 500

    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "INTERNAL_ERROR"
    assert data["error"]["message"] == "An unexpected internal server error occurred."
    assert data["error"]["retryable"] is False

    response_text = res.text
    # Must never leak sensitive details
    forbidden_tokens = [
        "Traceback",
        "RuntimeError",
        'File "',
        "/secret/db",
        "connection.py",
        "line 89",
    ]
    for token in forbidden_tokens:
        assert token not in response_text, (
            f"Leaked sensitive token in 500 response: '{token}'"
        )


# ------------------------------------------------------------------------------
# Criterion 3: Retryable is True ONLY for Segmentation Failure and Device States
# ------------------------------------------------------------------------------


def test_retryable_is_true_only_for_segmentation_failure_and_device_states(
    error_test_app: FastAPI,
) -> None:
    """Acceptance Criterion 3: retryable flag is strictly True for SEGMENTATION_FAILED and DEVICE_UNAVAILABLE."""
    client = TestClient(error_test_app, raise_server_exceptions=False)

    cases = [
        ("/trigger/segmentation-failed", True),
        ("/trigger/device-unavailable", True),
        ("/trigger/unsupported-media", False),
        ("/trigger/payload-too-large", False),
        ("/trigger/invalid-input", False),
        ("/trigger/missing-field", False),
        ("/trigger/schema-mismatch", False),
        ("/trigger/model-unavailable", False),
        ("/trigger/persistence-failed", False),
        ("/trigger/unhandled", False),
    ]

    for endpoint, expected_retryable in cases:
        res = client.get(endpoint)
        data = res.json()
        assert "error" in data, f"Missing error envelope on {endpoint}"
        retryable_val = data["error"]["retryable"]
        assert retryable_val is expected_retryable, (
            f"Endpoint {endpoint} has retryable={retryable_val}, expected {expected_retryable}"
        )
