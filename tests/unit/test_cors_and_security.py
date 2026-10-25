"""Unit tests for CORS allow-list and security headers (T3-005)."""
from __future__ import annotations

from scripts.verify_cors_and_security import (
    verify_allowed_origin_works,
    verify_disallowed_origin_blocked,
    verify_security_headers_present,
)


def test_disallowed_origin_blocked():
    ok, msg = verify_disallowed_origin_blocked()
    assert ok, msg


def test_security_headers_present():
    ok, msg = verify_security_headers_present()
    assert ok, msg


def test_allowed_origin_works():
    ok, msg = verify_allowed_origin_works()
    assert ok, msg
