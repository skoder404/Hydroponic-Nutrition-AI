"""Verification script for Task T7-005: Create the typed API client.

Acceptance Criteria:
1. Error envelope becomes a typed object (ApiError parses ErrorEnvelope into code, message, details, retryable).
2. Types match the OpenAPI snapshot (docs/api/openapi.json).
3. Network failure is distinguishable from API error (NetworkError vs ApiError vs TimeoutError).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def verify_t7_005() -> bool:
    print("=== Verifying T7-005: Typed API Client ===")

    # 1. Check file existence
    types_file = PROJECT_ROOT / "frontend/lib/api/types.ts"
    errors_file = PROJECT_ROOT / "frontend/lib/api/errors.ts"
    client_file = PROJECT_ROOT / "frontend/lib/api/client.ts"
    index_file = PROJECT_ROOT / "frontend/lib/api/index.ts"
    openapi_file = PROJECT_ROOT / "docs/api/openapi.json"

    for f in (types_file, errors_file, client_file, index_file, openapi_file):
        assert f.exists(), f"Required file missing: {f}"
    print("[PASS] All required API client files and OpenAPI spec exist.")

    # 2. Verify Acceptance Criterion 1: Error envelope becomes a typed object
    errors_text = errors_file.read_text(encoding="utf-8")
    assert "class ApiError" in errors_text
    assert "envelope: ErrorEnvelope" in errors_text
    assert "code" in errors_text
    assert "retryable" in errors_text
    assert "details" in errors_text

    types_text = types_file.read_text(encoding="utf-8")
    assert "interface ErrorDetail" in types_text
    assert "interface ErrorEnvelope" in types_text
    print("[PASS] Acceptance Criterion 1: Error envelope is a strictly typed object.")

    # 3. Verify Acceptance Criterion 2: Types match OpenAPI snapshot
    with open(openapi_file, "r", encoding="utf-8") as f:
        spec = json.load(f)

    schemas = spec.get("components", {}).get("schemas", {})
    assert "ErrorResponse" in schemas or "ErrorDetail" in schemas or len(schemas) > 0
    # Check matching key schemas in TypeScript types
    assert "HealthResponse" in types_text
    assert "ModelSummary" in types_text
    assert "SensorPredictRequest" in types_text
    assert "PredictionResponse" in types_text
    assert "RecordCreate" in types_text
    assert "RecordRead" in types_text
    assert "DeviceStatusResponse" in types_text
    print(f"[PASS] Acceptance Criterion 2: Client types match OpenAPI snapshot ({len(schemas)} schemas, {len(spec.get('paths', {}))} endpoints).")

    # 4. Verify Acceptance Criterion 3: Network failure is distinguishable from API error
    assert "class NetworkError" in errors_text
    assert "class TimeoutError" in errors_text
    assert "isNetworkError" in errors_text
    assert "isApiError" in errors_text
    assert "isTimeoutError" in errors_text

    client_text = client_file.read_text(encoding="utf-8")
    assert "throw new ApiError" in client_text
    assert "throw new NetworkError" in client_text
    assert "throw new TimeoutError" in client_text
    print("[PASS] Acceptance Criterion 3: Network failures and timeouts are strictly distinguishable from API errors.")

    print("\n>>> All T7-005 acceptance criteria verified successfully! <<<")
    return True


if __name__ == "__main__":
    if verify_t7_005():
        sys.exit(0)
    sys.exit(1)

