#!/usr/bin/env python3
"""Verification script for FastAPI app scaffold, /health, request-ID middleware, and JSON logging (T1-009).

Acceptance Criteria:
1. GET /api/v1/health returns 200 with documented fields (status, device, db, registry).
2. Every log line is JSON containing request_id, event, level.
3. No secret appears in logs.
"""
from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.core.device import get_device_manager
from backend.app.core.logging import setup_logging
from backend.app.main import create_app


def verify_health_endpoint() -> tuple[bool, str]:
    """Criterion 1: GET /api/v1/health returns 200 with documented fields."""
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/v1/health")
    if response.status_code != 200:
        return False, f"Expected 200 OK from /api/v1/health, got {response.status_code}"

    data = response.json()
    required_fields = ["status", "device", "db", "registry"]
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field in health response: '{field}'. Got: {data}"

    if data["db"] != "unknown":
        return False, f"Expected db='unknown', got {data['db']}"
    if data["registry"] != "unknown":
        return False, f"Expected registry='unknown', got {data['registry']}"

    expected_device = get_device_manager().get_device_str()
    if data["device"] != expected_device:
        return False, f"Expected device='{expected_device}', got '{data['device']}'"

    # Verify X-Request-ID response header
    req_id = response.headers.get("X-Request-ID")
    if not req_id:
        return False, "Response header missing 'X-Request-ID'"

    return True, f"Health returned 200 with documented fields: status='{data['status']}', device='{data['device']}', db='{data['db']}', registry='{data['registry']}'"


def verify_json_logging_and_request_id() -> tuple[bool, str]:
    """Criterion 2: Every log line is JSON containing request_id."""
    log_stream = io.StringIO()
    setup_logging(log_level="INFO", stream=log_stream)

    app = create_app()
    client = TestClient(app)

    custom_id = "test-custom-req-id-777"
    response = client.get("/api/v1/health", headers={"X-Request-ID": custom_id})
    if response.status_code != 200:
        return False, f"Failed health request during logging test: status {response.status_code}"

    # Also log a message outside of a request to test fallback request_id
    logger = logging.getLogger("test_logger")
    logger.info("Application lifecycle background log message")

    log_output = log_stream.getvalue().strip()
    lines = [line for line in log_output.split("\n") if line.strip()]

    if not lines:
        return False, "No log lines were captured."

    found_custom_req_id = False
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as err:
            return False, f"Log line is not valid JSON: '{line}', error: {err}"

        for req_key in ("event", "request_id", "level", "timestamp"):
            if req_key not in entry:
                return False, f"Log JSON missing required key '{req_key}': {line}"

        if entry.get("request_id") == custom_id:
            found_custom_req_id = True

    if not found_custom_req_id:
        return False, f"Did not find log entry with injected X-Request-ID '{custom_id}'."

    return True, f"Verified {len(lines)} log lines are valid JSON containing request_id and documented fields."


def verify_secret_redaction() -> tuple[bool, str]:
    """Criterion 3: No secret appears in logs."""
    log_stream = io.StringIO()
    setup_logging(log_level="INFO", stream=log_stream)

    logger = logging.getLogger("security_test")

    sensitive_tokens = [
        ("AIzaSyDfakeGeminiKeyForTesting1234567890", "AIzaSy key"),
        ("super_secret_db_password_999", "Database password in connection URL"),
        ("bearer_token_xyz_confidential_888", "Bearer token"),
        ("my_confidential_api_token_val", "Generic API key"),
    ]

    # Log entries simulating various accidental leak scenarios
    logger.info(f"Connecting to postgresql://postgres:{sensitive_tokens[1][0]}@localhost:5432/db")
    logger.info(f"Initialized Gemini model with key={sensitive_tokens[0][0]}")
    logger.info(f"Incoming Authorization: Bearer {sensitive_tokens[2][0]}")
    logger.info(f"Request with api_key='{sensitive_tokens[3][0]}'")

    log_output = log_stream.getvalue()

    for token, desc in sensitive_tokens:
        if token in log_output:
            return False, f"Secret leaked in log output for {desc}: '{token}' was found!"

    # Ensure settings repr also hides secrets
    settings = get_settings()
    settings_repr = repr(settings)
    if "postgres:" in settings_repr and "@" in settings_repr:
        # Check if database password was leaked in repr
        import re
        if re.search(r":([^:@*]+)@", settings_repr):
            return False, "Database password not masked in Settings.__repr__"

    return True, "All sensitive tokens, database passwords, and API keys are verified redacted from logs."


def main() -> int:
    checks = [
        ("Health endpoint returns 200 with documented fields", verify_health_endpoint),
        ("Every log line is JSON containing request_id", verify_json_logging_and_request_id),
        ("No secret appears in logs", verify_secret_redaction),
    ]

    print("================================================================================")
    print("VERIFY FASTAPI APP SCAFFOLD & LOGGING (T1-009)")
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
        print("All criteria successfully verified!")
        return 0
    print("Some criteria checks failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
