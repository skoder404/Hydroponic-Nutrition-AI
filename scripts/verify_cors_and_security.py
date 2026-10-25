#!/usr/bin/env python3
"""Verification script for CORS allow-list and security headers (T3-005).

Acceptance Criteria:
1. Disallowed origin is blocked.
2. Headers present on responses (X-Content-Type-Options, Referrer-Policy, Content-Security-Policy).
3. Allowed origin works from the dev frontend.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def verify_disallowed_origin_blocked() -> tuple[bool, str]:
    """Criterion 1: Disallowed origin is blocked."""
    app = create_app(configure_logging=False)
    client = TestClient(app)

    disallowed_origin = "http://malicious-attacker.com"

    # 1. Simple GET request with disallowed Origin
    res = client.get("/api/v1/health", headers={"Origin": disallowed_origin})
    allowed_hdr = res.headers.get("access-control-allow-origin")
    if allowed_hdr == disallowed_origin or allowed_hdr == "*":
        return False, f"Disallowed origin was allowed in simple request header: '{allowed_hdr}'"

    # 2. Preflight OPTIONS request with disallowed Origin
    preflight = client.options(
        "/api/v1/health",
        headers={
            "Origin": disallowed_origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    preflight_allow = preflight.headers.get("access-control-allow-origin")
    if preflight_allow == disallowed_origin or preflight_allow == "*":
        return False, f"Disallowed origin was allowed in preflight OPTIONS: '{preflight_allow}'"

    # 3. Forbid wildcard in Settings
    try:
        Settings(allowed_origins="http://localhost:3000, *")
        return False, "Settings failed to forbid wildcard '*' in allowed_origins."
    except ValueError as exc:
        if "forbidden" not in str(exc).lower():
            return False, f"Unexpected error message: {exc}"

    return True, f"Disallowed origin '{disallowed_origin}' is strictly blocked from CORS responses and wildcards are forbidden."


def verify_security_headers_present() -> tuple[bool, str]:
    """Criterion 2: Headers present on responses."""
    app = create_app(configure_logging=False)
    client = TestClient(app)

    res = client.get("/api/v1/health")
    if res.status_code != 200:
        return False, f"Health endpoint returned non-200: {res.status_code}"

    headers = res.headers

    # Check X-Content-Type-Options
    if headers.get("X-Content-Type-Options") != "nosniff":
        return False, f"Missing or incorrect X-Content-Type-Options: {headers.get('X-Content-Type-Options')}"

    # Check Referrer-Policy
    if headers.get("Referrer-Policy") != "strict-origin-when-cross-origin":
        return False, f"Missing or incorrect Referrer-Policy: {headers.get('Referrer-Policy')}"

    # Check Content-Security-Policy
    csp = headers.get("Content-Security-Policy")
    if not csp:
        return False, "Missing Content-Security-Policy header."
    for directive in ("default-src 'self'", "frame-ancestors 'none'"):
        if directive not in csp:
            return False, f"Content-Security-Policy missing required directive '{directive}'"

    # Check X-Frame-Options
    if headers.get("X-Frame-Options") != "DENY":
        return False, f"Missing or incorrect X-Frame-Options: {headers.get('X-Frame-Options')}"

    return True, "Security headers verified present: X-Content-Type-Options, Referrer-Policy, Content-Security-Policy, X-Frame-Options."


def verify_allowed_origin_works() -> tuple[bool, str]:
    """Criterion 3: Allowed origin works from the dev frontend."""
    app = create_app(configure_logging=False)
    client = TestClient(app)

    allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

    for origin in allowed_origins:
        # Simple request
        res = client.get("/api/v1/health", headers={"Origin": origin})
        if res.status_code != 200:
            return False, f"Request from allowed origin '{origin}' failed with status {res.status_code}"
        if res.headers.get("access-control-allow-origin") != origin:
            return False, f"Expected access-control-allow-origin='{origin}', got '{res.headers.get('access-control-allow-origin')}'"
        if res.headers.get("access-control-allow-credentials") != "true":
            return False, f"Expected credentials=true for origin '{origin}'"

        # Preflight OPTIONS request
        preflight = client.options(
            "/api/v1/health",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        if preflight.status_code != 200:
            return False, f"Preflight OPTIONS for origin '{origin}' returned status {preflight.status_code}"
        if preflight.headers.get("access-control-allow-origin") != origin:
            return False, f"Preflight access-control-allow-origin expected '{origin}', got '{preflight.headers.get('access-control-allow-origin')}'"

    return True, f"Allowed origins {allowed_origins} verified fully functional with credentials and preflight."


def main() -> int:
    checks = [
        ("Disallowed origin is blocked", verify_disallowed_origin_blocked),
        ("Headers present on responses", verify_security_headers_present),
        ("Allowed origin works from the dev frontend", verify_allowed_origin_works),
    ]

    print("================================================================================")
    print("VERIFY CORS ALLOW-LIST & SECURITY HEADERS (T3-005)")
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
