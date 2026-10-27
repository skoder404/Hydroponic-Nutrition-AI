"""Unit tests for HTTPS/TLS deployment documentation and README links (T3-008).

Acceptance Criteria:
1. Doc includes a working sample config.
2. States localhost-only exception.
3. Linked from README.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def test_https_doc_exists_and_contains_configs() -> None:
    """Acceptance Criterion 1: docs/deployment/https.md exists and contains sample Caddy & NGINX configs."""
    doc_path = ROOT / "docs" / "deployment" / "https.md"
    assert doc_path.is_file(), f"Expected {doc_path} to exist as a file."

    content = doc_path.read_text(encoding="utf-8")

    # Caddy checks
    assert "```caddy" in content
    assert "Strict-Transport-Security" in content
    assert "reverse_proxy" in content
    assert "127.0.0.1:8000" in content

    # NGINX checks
    assert "```nginx" in content
    assert "ssl_certificate" in content
    assert "proxy_pass http://127.0.0.1:8000;" in content
    assert "proxy_pass http://127.0.0.1:3000;" in content
    assert "max-age=31536000" in content


def test_https_doc_states_localhost_only_exception() -> None:
    """Acceptance Criterion 2: Doc explicitly states plain HTTP is for localhost only."""
    doc_path = ROOT / "docs" / "deployment" / "https.md"
    content = doc_path.read_text(encoding="utf-8")

    assert "localhost" in content.lower()
    assert "127.0.0.1" in content
    assert "plain http" in content.lower()
    assert "only" in content.lower()


def test_https_doc_linked_from_readme() -> None:
    """Acceptance Criterion 3: README.md links to docs/deployment/https.md."""
    readme_path = ROOT / "README.md"
    content = readme_path.read_text(encoding="utf-8")

    assert "docs/deployment/https.md" in content
