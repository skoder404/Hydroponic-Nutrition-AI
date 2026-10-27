#!/usr/bin/env python3
"""Verification script for the shared error envelope and exception handlers (T3-004).

Acceptance Criteria:
1. Each code returns the documented HTTP status.
2. Unhandled error response contains no stack trace or path.
3. retryable is true only for segmentation failure and device states.
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, S110
        pass

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.core.errors import (
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


def _build_test_app() -> FastAPI:
    """Create test application mounting endpoints for all error scenarios."""
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/error/invalid-input")
    def r_invalid() -> None:
        raise InvalidInputError("Field validation failed.")

    @app.get("/error/missing-field")
    def r_missing() -> None:
        raise MissingFieldError("Field 'ec' is required.")

    @app.get("/error/unsupported-media")
    def r_media() -> None:
        raise UnsupportedMediaError("Invalid image signature.")

    @app.get("/error/payload-too-large")
    def r_payload() -> None:
        raise PayloadTooLargeError("File exceeds 15 MB cap.")

    @app.get("/error/segmentation-failed")
    def r_segmentation() -> None:
        raise SegmentationFailedError("SAM 2.1 failed to detect leaf.")

    @app.get("/error/schema-mismatch")
    def r_schema() -> None:
        raise SchemaMismatchError("Feature count mismatch.")

    @app.get("/error/model-unavailable")
    def r_model() -> None:
        raise ModelUnavailableError("No model promoted for crop.")

    @app.get("/error/device-unavailable")
    def r_device() -> None:
        raise DeviceUnavailableError("Campus server timed out.")

    @app.get("/error/persistence-failed")
    def r_persist() -> None:
        raise PersistenceFailedError("Database connection lost.")

    @app.get("/error/unhandled")
    def r_unhandled() -> None:
        raise RuntimeError(
            "Fatal connection syntax error at /opt/backend/secrets/db.py line 404"
        )

    return app


def verify_each_code_returns_documented_http_status() -> tuple[bool, str]:
    """Criterion 1: Each code returns the documented HTTP status."""
    app = _build_test_app()
    client = TestClient(app)

    expected_table = [
        ("/error/invalid-input", "INVALID_INPUT", 422),
        ("/error/missing-field", "MISSING_FIELD", 422),
        ("/error/unsupported-media", "UNSUPPORTED_MEDIA", 415),
        ("/error/payload-too-large", "PAYLOAD_TOO_LARGE", 413),
        ("/error/segmentation-failed", "SEGMENTATION_FAILED", 422),
        ("/error/schema-mismatch", "SCHEMA_MISMATCH", 409),
        ("/error/model-unavailable", "MODEL_UNAVAILABLE", 503),
        ("/error/device-unavailable", "DEVICE_UNAVAILABLE", 503),
        ("/error/persistence-failed", "PERSISTENCE_FAILED", 503),
    ]

    for path, code, expected_status in expected_table:
        res = client.get(path)
        if res.status_code != expected_status:
            return (
                False,
                f"Endpoint {path} returned HTTP {res.status_code}, expected {expected_status}",
            )
        data = res.json()
        if "error" not in data or data["error"].get("code") != code:
            return (
                False,
                f"Endpoint {path} returned code '{data.get('error', {}).get('code')}', expected '{code}'",
            )

    return (
        True,
        "Criterion 1 satisfied: All error codes return the exact HTTP status specified in TECH_RULES table.",
    )


def verify_unhandled_error_response_contains_no_stack_trace_or_path() -> tuple[
    bool, str
]:
    """Criterion 2: Unhandled error response contains no stack trace or path."""
    app = _build_test_app()
    client = TestClient(app, raise_server_exceptions=False)

    res = client.get("/error/unhandled")
    if res.status_code != 500:
        return False, f"Unhandled error returned HTTP {res.status_code}, expected 500"

    data = res.json()
    err_obj = data.get("error", {})
    if err_obj.get("code") != "INTERNAL_ERROR":
        return False, f"Expected code 'INTERNAL_ERROR', got '{err_obj.get('code')}'"

    raw_text = res.text
    leak_tokens = [
        "Traceback",
        "RuntimeError",
        "/opt/backend",
        "secrets",
        "db.py",
        "line 404",
    ]
    for token in leak_tokens:
        if token in raw_text:
            return False, f"Leaked sensitive token in 500 response: '{token}'"

    return (
        True,
        "Criterion 2 satisfied: Unhandled exceptions return generic message without stack traces or server paths.",
    )


def verify_retryable_is_true_only_for_segmentation_failure_and_device_states() -> tuple[
    bool, str
]:
    """Criterion 3: retryable is true only for segmentation failure and device states."""
    app = _build_test_app()
    client = TestClient(app, raise_server_exceptions=False)

    endpoints = [
        ("/error/segmentation-failed", True),
        ("/error/device-unavailable", True),
        ("/error/invalid-input", False),
        ("/error/missing-field", False),
        ("/error/unsupported-media", False),
        ("/error/payload-too-large", False),
        ("/error/schema-mismatch", False),
        ("/error/model-unavailable", False),
        ("/error/persistence-failed", False),
        ("/error/unhandled", False),
    ]

    for path, expected_retryable in endpoints:
        res = client.get(path)
        data = res.json()
        retryable_val = data.get("error", {}).get("retryable")
        if retryable_val is not expected_retryable:
            return (
                False,
                f"Endpoint {path} returned retryable={retryable_val}, expected {expected_retryable}",
            )

    return (
        True,
        "Criterion 3 satisfied: 'retryable' flag is strictly True only for SEGMENTATION_FAILED and DEVICE_UNAVAILABLE.",
    )


def main() -> int:
    """Run all verification checks for T3-004."""
    print("=" * 70)
    print("T3-004 Verification: Shared Error Envelope & Exception Handlers")
    print("=" * 70)

    verifiers = [
        (
            "Criterion 1: Each code returns documented HTTP status",
            verify_each_code_returns_documented_http_status,
        ),
        (
            "Criterion 2: Unhandled error leaks no trace/path",
            verify_unhandled_error_response_contains_no_stack_trace_or_path,
        ),
        (
            "Criterion 3: retryable is true only for segmentation & device",
            verify_retryable_is_true_only_for_segmentation_failure_and_device_states,
        ),
    ]

    all_passed = True
    for label, fn in verifiers:
        ok, msg = fn()
        if ok:
            print(f"[PASS] {label}")
            print(f"       {msg}\n")
        else:
            print(f"[FAIL] {label}")
            print(f"       {msg}\n")
            all_passed = False

    if all_passed:
        print("ALL CRITERIA PASSED! T3-004 is fully verified.")
        return 0
    else:
        print("VERIFICATION FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
