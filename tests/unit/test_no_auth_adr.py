"""Unit tests for ADR 0005 No Authentication in v1 (T3-001)."""
from scripts.verify_no_auth_adr import check_adr, check_readme_link


def test_adr_contents():
    ok, msg = check_adr()
    assert ok, msg


def test_readme_link():
    ok, msg = check_readme_link()
    assert ok, msg
