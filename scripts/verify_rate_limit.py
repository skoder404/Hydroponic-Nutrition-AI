#!/usr/bin/env python3
"""Verification script for per-IP rate limiting and exception handlers (T3-006).

Acceptance Criteria:
1. Burst test returns 429 in the standard envelope.
2. Limits are configurable via env.
3. Health endpoint is exempt.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, S110
        pass

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.core.rate_limit import RateLimiter, RateLimitMiddleware


def _build_verification_app(limiter: RateLimiter) -> FastAPI:
    """Construct lightweight FastAPI test application."""
    app = FastAPI(title="Rate Limit Verification App")
    app.add_middleware(RateLimitMiddleware, limiter=limiter)

    @app.get("/health")
    def health_root() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/health")
    def health_v1() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/data")
    def default_endpoint() -> dict[str, str]:
        return {"status": "success"}

    @app.post("/predict/image")
    def predict_image_root() -> dict[str, str]:
        return {"prediction": "healthy"}

    @app.post("/api/v1/predict/image")
    def predict_image_v1() -> dict[str, str]:
        return {"prediction": "healthy"}

    @app.post("/api/v1/analyze")
    def analyze_v1() -> dict[str, str]:
        return {"analysis": "complete"}

    return app


def verify_criterion_1_burst_test_returns_429() -> None:
    """Criterion 1: Burst test returns 429 in the standard envelope."""
    os.environ["RATE_LIMIT_DEFAULT"] = "3/minute"
    os.environ["RATE_LIMIT_STRICT"] = "2/minute"
    os.environ["RATE_LIMIT_ENABLED"] = "true"
    get_settings.cache_clear()

    limiter = RateLimiter()
    app = _build_verification_app(limiter)
    client = TestClient(app)

    ip = "192.168.10.42"

    # Burst 3 requests on default tier -> should succeed (200)
    for i in range(3):
        res = client.get("/api/v1/data", headers={"X-Forwarded-For": ip})
        assert res.status_code == 200, (
            f"Request {i + 1} failed with status {res.status_code}"
        )
        assert res.headers.get("X-RateLimit-Limit") == "3"

    # 4th burst request must be rejected with 429
    blocked = client.get("/api/v1/data", headers={"X-Forwarded-For": ip})
    assert blocked.status_code == 429, f"Expected 429, got {blocked.status_code}"
    assert "Retry-After" in blocked.headers
    assert blocked.headers.get("X-RateLimit-Remaining") == "0"

    envelope = blocked.json()
    assert "error" in envelope, "Response missing 'error' key"
    err = envelope["error"]
    assert err["code"] == "RATE_LIMIT_EXCEEDED", f"Unexpected code: {err.get('code')}"
    assert err["retryable"] is False, "retryable must be false for rate limits"
    assert "limit" in err["details"]
    assert err["details"]["limit"] == 3
    assert err["details"]["window_seconds"] == 60
    assert err["details"]["client_ip"] == ip

    # Verify strict endpoint /api/v1/predict/image trips after 2 requests
    strict_ip = "192.168.10.43"
    client.post("/api/v1/predict/image", headers={"X-Forwarded-For": strict_ip})
    client.post("/api/v1/predict/image", headers={"X-Forwarded-For": strict_ip})
    strict_blocked = client.post(
        "/api/v1/predict/image", headers={"X-Forwarded-For": strict_ip}
    )
    assert strict_blocked.status_code == 429
    assert strict_blocked.json()["error"]["details"]["limit"] == 2

    print("[PASS] Criterion 1: Burst test returns 429 in the standard envelope")
    print(
        f"       Code: {err['code']}, Retryable: {err['retryable']}, Limit: {err['details']['limit']}"
    )


def verify_criterion_2_limits_configurable_via_env() -> None:
    """Criterion 2: Limits are configurable via env."""
    # Test setting custom limits via env
    os.environ["RATE_LIMIT_DEFAULT"] = "1/second"
    os.environ["RATE_LIMIT_STRICT"] = "1/second"
    os.environ["RATE_LIMIT_ENABLED"] = "true"
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.rate_limit_default == "1/second"
    assert settings.rate_limit_enabled is True

    limiter = RateLimiter()
    app = _build_verification_app(limiter)
    client = TestClient(app)

    ip = "172.16.0.5"
    r1 = client.get("/api/v1/data", headers={"X-Forwarded-For": ip})
    assert r1.status_code == 200

    r2 = client.get("/api/v1/data", headers={"X-Forwarded-For": ip})
    assert r2.status_code == 429
    assert r2.json()["error"]["details"]["limit"] == 1

    # Test disabling rate limit via env
    os.environ["RATE_LIMIT_ENABLED"] = "false"
    get_settings.cache_clear()
    limiter_disabled = RateLimiter()
    app_disabled = _build_verification_app(limiter_disabled)
    client_disabled = TestClient(app_disabled)

    for _ in range(10):
        r = client_disabled.get("/api/v1/data", headers={"X-Forwarded-For": ip})
        assert r.status_code == 200

    print("[PASS] Criterion 2: Limits are configurable via env")
    print(
        "       Verified custom threshold (1/second) and enable/disable toggle (RATE_LIMIT_ENABLED)."
    )


def verify_criterion_3_health_endpoint_is_exempt() -> None:
    """Criterion 3: Health endpoint is exempt."""
    os.environ["RATE_LIMIT_DEFAULT"] = "1/minute"
    os.environ["RATE_LIMIT_STRICT"] = "1/minute"
    os.environ["RATE_LIMIT_ENABLED"] = "true"
    get_settings.cache_clear()

    limiter = RateLimiter()
    app = _build_verification_app(limiter)
    client = TestClient(app)

    health_ip = "10.10.10.10"

    # Send 50 burst requests to /health
    for i in range(50):
        res = client.get("/health", headers={"X-Forwarded-For": health_ip})
        assert res.status_code == 200, (
            f"/health request {i + 1} was unexpectedly rate limited!"
        )

    # Send 50 burst requests to /api/v1/health
    for i in range(50):
        res = client.get("/api/v1/health", headers={"X-Forwarded-For": health_ip})
        assert res.status_code == 200, (
            f"/api/v1/health request {i + 1} was unexpectedly rate limited!"
        )

    # Non-exempt endpoint from same IP is blocked after 1 request
    r_item1 = client.get("/api/v1/data", headers={"X-Forwarded-For": health_ip})
    assert r_item1.status_code == 200
    r_item2 = client.get("/api/v1/data", headers={"X-Forwarded-For": health_ip})
    assert r_item2.status_code == 429

    print("[PASS] Criterion 3: Health endpoint is exempt")
    print(
        "       Verified 100 consecutive requests to /health and /api/v1/health without rate limiting."
    )


def main() -> None:
    """Run all T3-006 criteria verifications."""
    print("=" * 70)
    print("T3-006 Verification: Per-IP Rate Limiting & Exemption Suite")
    print("=" * 70)

    try:
        verify_criterion_1_burst_test_returns_429()
        verify_criterion_2_limits_configurable_via_env()
        verify_criterion_3_health_endpoint_is_exempt()
        print("\nALL CRITERIA PASSED! T3-006 is fully verified.")
    finally:
        # Cleanup environment variables
        os.environ.pop("RATE_LIMIT_DEFAULT", None)
        os.environ.pop("RATE_LIMIT_STRICT", None)
        os.environ.pop("RATE_LIMIT_ENABLED", None)
        get_settings.cache_clear()


if __name__ == "__main__":
    main()
