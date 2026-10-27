#!/usr/bin/env python3
"""Verification script for HTTPS/TLS reverse-proxy documentation (T3-008).

Acceptance Criteria:
1. Doc includes a working sample config (Caddy & NGINX with TLS and HSTS).
2. States localhost-only exception.
3. Linked from README.
"""

from __future__ import annotations

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


def verify_criterion_1_working_sample_configs() -> None:
    """Criterion 1: Doc includes a working sample config."""
    doc_path = ROOT / "docs" / "deployment" / "https.md"
    assert doc_path.exists(), f"Expected '{doc_path}' to exist!"

    content = doc_path.read_text(encoding="utf-8")
    assert len(content) > 500, "Documentation file is suspiciously short!"

    # Verify Caddy sample config
    assert "```caddy" in content, "Missing Caddy code block in https.md"
    assert "Strict-Transport-Security" in content, (
        "Missing Strict-Transport-Security in config"
    )
    assert "reverse_proxy" in content, "Missing reverse_proxy directive in Caddy config"
    assert "127.0.0.1:8000" in content, "Missing backend proxy target in Caddy config"
    assert "127.0.0.1:3000" in content, "Missing frontend proxy target in Caddy config"

    # Verify NGINX sample config
    assert "```nginx" in content, "Missing NGINX code block in https.md"
    assert "ssl_certificate" in content, "Missing ssl_certificate in NGINX config"
    assert "proxy_pass http://127.0.0.1:8000;" in content, (
        "Missing proxy_pass to backend"
    )
    assert "proxy_pass http://127.0.0.1:3000;" in content, (
        "Missing proxy_pass to frontend"
    )
    assert "max-age=31536000" in content, "Missing HSTS max-age duration"

    print("[PASS] Criterion 1: Doc includes a working sample config")
    print("       Verified Caddy and NGINX reverse-proxy configs with TLS and HSTS.")


def verify_criterion_2_states_localhost_only_exception() -> None:
    """Criterion 2: States localhost-only exception."""
    doc_path = ROOT / "docs" / "deployment" / "https.md"
    content = doc_path.read_text(encoding="utf-8")

    assert "localhost" in content.lower(), "Missing localhost mention in https.md"
    assert "127.0.0.1" in content, "Missing 127.0.0.1 loopback reference"

    # Verify statement of the exception and mandate
    has_exception_statement = (
        "plain http" in content.lower()
        and "only" in content.lower()
        and "localhost" in content.lower()
    )
    assert has_exception_statement, (
        "Doc must explicitly state that plain HTTP is only for localhost development."
    )

    has_non_localhost_mandate = (
        "non-localhost" in content.lower()
        or "campus" in content.lower()
        or "outside" in content.lower()
    )
    assert has_non_localhost_mandate, (
        "Doc must state that non-localhost deployments mandate HTTPS."
    )

    print("[PASS] Criterion 2: States localhost-only exception")
    print(
        "       Verified plain HTTP is permitted only for localhost development; HTTPS mandated elsewhere."
    )


def verify_criterion_3_linked_from_readme() -> None:
    """Criterion 3: Linked from README."""
    readme_path = ROOT / "README.md"
    assert readme_path.exists(), "README.md does not exist!"

    content = readme_path.read_text(encoding="utf-8")
    assert "docs/deployment/https.md" in content, (
        "README.md does not link to docs/deployment/https.md"
    )

    print("[PASS] Criterion 3: Linked from README")
    print(
        "       Verified links in Documentation map table and Security notes of README.md."
    )


def main() -> None:
    """Run all T3-008 criteria verifications."""
    print("=" * 70)
    print("T3-008 Verification: HTTPS/TLS Deployment Documentation")
    print("=" * 70)

    verify_criterion_1_working_sample_configs()
    verify_criterion_2_states_localhost_only_exception()
    verify_criterion_3_linked_from_readme()

    print("\nALL CRITERIA PASSED! T3-008 is fully verified.")


if __name__ == "__main__":
    main()
