"""Unit tests for per-IP rate limiting middleware, configuration, and exemptions (T3-006).

Acceptance Criteria:
1. Burst test returns 429 in the standard envelope.
2. Limits are configurable via env.
3. Health endpoint is exempt.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.core.rate_limit import (
    RateLimiter,
    RateLimitMiddleware,
    parse_rate_limit,
)


@pytest.fixture(autouse=True)
def reset_rate_limiter() -> None:
    """Reset the rate limiter state before each test."""
    from backend.app.core.rate_limit import get_rate_limiter

    get_rate_limiter().reset()


def _build_test_app(custom_limiter: RateLimiter | None = None) -> FastAPI:
    """Create a minimalist FastAPI application wired with RateLimitMiddleware and test endpoints."""
    app = FastAPI()

    # Mount rate limiting middleware
    app.add_middleware(RateLimitMiddleware, limiter=custom_limiter)

    @app.get("/health")
    def health_root() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/health")
    def health_api_v1() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/items")
    def general_endpoint() -> dict[str, str]:
        return {"message": "success"}

    @app.post("/predict/image")
    def predict_image_root() -> dict[str, str]:
        return {"result": "prediction"}

    @app.post("/api/v1/predict/image")
    def predict_image_v1() -> dict[str, str]:
        return {"result": "prediction"}

    @app.post("/analyze")
    def analyze_root() -> dict[str, str]:
        return {"analysis": "complete"}

    @app.post("/api/v1/analyze")
    def analyze_v1() -> dict[str, str]:
        return {"analysis": "complete"}

    return app


# ==============================================================================
# 1. Parsing & Expression Unit Tests
# ==============================================================================


@pytest.mark.parametrize(
    "expr,expected_limit,expected_window",
    [
        ("60/minute", 60, 60.0),
        ("10/second", 10, 1.0),
        ("5/sec", 5, 1.0),
        ("100/hour", 100, 3600.0),
        ("500/day", 500, 86400.0),
        ("30/2minute", 30, 120.0),
        ("100", 100, 60.0),
        (50, 50, 60.0),
    ],
)
def test_parse_rate_limit_valid(
    expr: str | int, expected_limit: int, expected_window: float
) -> None:
    """Verify correct parsing of rate limit strings and numeric values."""
    limit, window = parse_rate_limit(expr)
    assert limit == expected_limit
    assert window == expected_window


def test_parse_rate_limit_invalid() -> None:
    """Verify ValueError is raised on malformed rate limit strings."""
    with pytest.raises(ValueError, match="Invalid rate limit"):
        parse_rate_limit("invalid")

    with pytest.raises(ValueError, match="Unsupported time unit"):
        parse_rate_limit("10/fortnight")


# ==============================================================================
# Criterion 1: Burst test returns 429 in the standard envelope
# ==============================================================================


def test_burst_test_returns_429_in_standard_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance Criterion 1: Bursting past the limit returns HTTP 429 with standard error envelope."""
    # Configure low limits for testing burst
    monkeypatch.setenv("RATE_LIMIT_DEFAULT", "3/minute")
    from backend.app.core.config import get_settings

    get_settings.cache_clear()

    limiter = RateLimiter()
    app = _build_test_app(custom_limiter=limiter)
    client = TestClient(app)

    # First 3 requests to general endpoint succeed
    for i in range(3):
        res = client.get("/api/v1/items", headers={"X-Forwarded-For": "192.168.1.100"})
        assert res.status_code == 200, f"Request {i + 1} failed unexpectedly"
        assert res.headers.get("X-RateLimit-Limit") == "3"

    # 4th burst request must be rejected with 429
    blocked_res = client.get(
        "/api/v1/items", headers={"X-Forwarded-For": "192.168.1.100"}
    )
    assert blocked_res.status_code == 429
    assert "Retry-After" in blocked_res.headers
    assert blocked_res.headers.get("X-RateLimit-Remaining") == "0"

    # Verify standard error envelope structure
    payload = blocked_res.json()
    assert "error" in payload
    error = payload["error"]
    assert error["code"] == "RATE_LIMIT_EXCEEDED"
    assert "Rate limit exceeded" in error["message"]
    assert error["retryable"] is False
    assert error["details"]["limit"] == 3
    assert error["details"]["window_seconds"] == 60
    assert error["details"]["retry_after_seconds"] >= 1
    assert error["details"]["client_ip"] == "192.168.1.100"

    get_settings.cache_clear()


def test_strict_endpoints_have_lower_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that /predict/image and /analyze enforce stricter limits than default."""
    monkeypatch.setenv("RATE_LIMIT_DEFAULT", "10/minute")
    monkeypatch.setenv("RATE_LIMIT_STRICT", "2/minute")
    from backend.app.core.config import get_settings

    get_settings.cache_clear()

    limiter = RateLimiter()
    app = _build_test_app(custom_limiter=limiter)
    client = TestClient(app)

    # Strict endpoint /predict/image should allow 2, then fail on 3rd
    client.post("/api/v1/predict/image", headers={"X-Forwarded-For": "10.0.0.1"})
    client.post("/api/v1/predict/image", headers={"X-Forwarded-For": "10.0.0.1"})
    res_strict = client.post(
        "/api/v1/predict/image", headers={"X-Forwarded-For": "10.0.0.1"}
    )
    assert res_strict.status_code == 429
    assert res_strict.json()["error"]["details"]["limit"] == 2

    # Meanwhile, /api/v1/items for the same IP still has capacity because it uses default tier
    res_default = client.get("/api/v1/items", headers={"X-Forwarded-For": "10.0.0.1"})
    assert res_default.status_code == 200

    # Strict endpoint /analyze also uses strict limit
    client.post("/api/v1/analyze", headers={"X-Forwarded-For": "10.0.0.2"})
    client.post("/api/v1/analyze", headers={"X-Forwarded-For": "10.0.0.2"})
    res_analyze = client.post(
        "/api/v1/analyze", headers={"X-Forwarded-For": "10.0.0.2"}
    )
    assert res_analyze.status_code == 429
    assert res_analyze.json()["error"]["details"]["limit"] == 2

    get_settings.cache_clear()


def test_per_ip_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that hitting the rate limit on one IP does not block another IP."""
    monkeypatch.setenv("RATE_LIMIT_DEFAULT", "2/minute")
    from backend.app.core.config import get_settings

    get_settings.cache_clear()

    limiter = RateLimiter()
    app = _build_test_app(custom_limiter=limiter)
    client = TestClient(app)

    # Exhaust IP 1
    client.get("/api/v1/items", headers={"X-Forwarded-For": "1.1.1.1"})
    client.get("/api/v1/items", headers={"X-Forwarded-For": "1.1.1.1"})
    assert (
        client.get("/api/v1/items", headers={"X-Forwarded-For": "1.1.1.1"}).status_code
        == 429
    )

    # IP 2 should still be allowed
    res_ip2 = client.get("/api/v1/items", headers={"X-Forwarded-For": "2.2.2.2"})
    assert res_ip2.status_code == 200

    get_settings.cache_clear()


# ==============================================================================
# Criterion 2: Limits are configurable via env
# ==============================================================================


def test_limits_configurable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Acceptance Criterion 2: Rate limit values and enable switch are configurable via env."""
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_DEFAULT", "1/second")
    monkeypatch.setenv("RATE_LIMIT_STRICT", "1/second")
    from backend.app.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.rate_limit_enabled is True
    assert settings.rate_limit_default == "1/second"
    assert settings.rate_limit_strict == "1/second"

    limiter = RateLimiter()
    app = _build_test_app(custom_limiter=limiter)
    client = TestClient(app)

    # First request passes
    res1 = client.get("/api/v1/items", headers={"X-Forwarded-For": "3.3.3.3"})
    assert res1.status_code == 200

    # Immediate second request fails
    res2 = client.get("/api/v1/items", headers={"X-Forwarded-For": "3.3.3.3"})
    assert res2.status_code == 429
    assert res2.json()["error"]["details"]["limit"] == 1

    get_settings.cache_clear()


def test_rate_limiting_disabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that setting RATE_LIMIT_ENABLED=false bypasses all rate limiting."""
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("RATE_LIMIT_DEFAULT", "1/minute")
    from backend.app.core.config import get_settings

    get_settings.cache_clear()

    limiter = RateLimiter()
    app = _build_test_app(custom_limiter=limiter)
    client = TestClient(app)

    # 10 burst requests should all pass with 200 when disabled
    for _ in range(10):
        res = client.get("/api/v1/items", headers={"X-Forwarded-For": "4.4.4.4"})
        assert res.status_code == 200

    get_settings.cache_clear()


# ==============================================================================
# Criterion 3: Health endpoint is exempt
# ==============================================================================


def test_health_endpoints_are_exempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Acceptance Criterion 3: Health endpoints are exempt from rate limiting even during heavy bursts."""
    # Set an extremely strict 1 request per hour limit
    monkeypatch.setenv("RATE_LIMIT_DEFAULT", "1/hour")
    monkeypatch.setenv("RATE_LIMIT_STRICT", "1/hour")
    from backend.app.core.config import get_settings

    get_settings.cache_clear()

    limiter = RateLimiter()
    app = _build_test_app(custom_limiter=limiter)
    client = TestClient(app)

    # Heavy burst of 50 requests to /health must all return 200
    for _ in range(50):
        res_root = client.get("/health", headers={"X-Forwarded-For": "5.5.5.5"})
        assert res_root.status_code == 200
        assert res_root.json() == {"status": "ok"}

    # Heavy burst of 50 requests to /api/v1/health must all return 200
    for _ in range(50):
        res_v1 = client.get("/api/v1/health", headers={"X-Forwarded-For": "5.5.5.5"})
        assert res_v1.status_code == 200
        assert res_v1.json() == {"status": "ok"}

    # Normal endpoint for the same IP still triggers 429 after 1 request
    res_item1 = client.get("/api/v1/items", headers={"X-Forwarded-For": "5.5.5.5"})
    assert res_item1.status_code == 200
    res_item2 = client.get("/api/v1/items", headers={"X-Forwarded-For": "5.5.5.5"})
    assert res_item2.status_code == 429

    get_settings.cache_clear()
