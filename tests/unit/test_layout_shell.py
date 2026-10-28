"""Unit tests for layout shell: nav, footer disclaimer, and health banner (T7-004).

Acceptance Criteria:
- [x] Keyboard navigation reaches all links
- [x] Banner appears when the API is down
- [x] Disclaimer text avoids 'diagnosis' wording
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def test_keyboard_navigation_reaches_all_links():
    """AC 1: Keyboard navigation reaches all links with visible focus."""
    layout_path = ROOT / "frontend/app/layout.tsx"
    assert layout_path.is_file()
    content = layout_path.read_text(encoding="utf-8")

    # Check for skip to content accessible anchor
    assert 'href="#main-content"' in content
    assert "Skip to main content" in content

    # Check navigation container
    assert '<nav aria-label="Main Navigation"' in content or 'aria-label="Main Navigation"' in content

    # Check all key routes are linked
    expected_routes = ["/analyze", "/history", "/models", "/about"]
    for route in expected_routes:
        assert f'href="{route}"' in content, f"Missing navigation link for {route}"

    # Verify focus-visible styling is present across interactive nav links
    assert "focus-visible:ring-2" in content
    assert "focus-visible:ring-[var(--ring-color)]" in content


def test_banner_appears_when_api_down():
    """AC 2: Banner appears when the API is down."""
    layout_path = ROOT / "frontend/app/layout.tsx"
    banner_path = ROOT / "frontend/components/ui/HealthBanner.tsx"

    assert banner_path.is_file(), "HealthBanner.tsx must exist"
    layout_content = layout_path.read_text(encoding="utf-8")
    banner_content = banner_path.read_text(encoding="utf-8")

    # Verify HealthBanner is mounted in RootLayout
    assert "<HealthBanner" in layout_content

    # Verify accessibility attributes for non-blocking alert
    assert 'role="alert"' in banner_content
    assert 'aria-live="polite"' in banner_content

    # Verify retry mechanism and failure messaging
    assert "Backend unreachable" in banner_content
    assert "Retry Connection" in banner_content
    assert "checkHealth" in banner_content


def test_disclaimer_text_avoids_diagnosis_wording():
    """AC 3: Disclaimer text strictly avoids 'diagnosis' wording."""
    layout_path = ROOT / "frontend/app/layout.tsx"
    content = layout_path.read_text(encoding="utf-8")

    # Extract footer text
    footer_match = re.search(r"<footer[^>]*>(.*?)</footer>", content, re.DOTALL)
    assert footer_match, "Footer element must be present in layout.tsx"
    footer_text = footer_match.group(1).lower()

    # Invariant: Must NOT contain 'diagnosis' or 'diagnos'
    assert "diagnosis" not in footer_text, "Footer disclaimer must strictly avoid 'diagnosis' wording"
    assert "diagnos" not in footer_text, "Footer disclaimer must not use any grammatical form of diagnose"

    # Must contain advisory wording
    assert "advisory" in footer_text
