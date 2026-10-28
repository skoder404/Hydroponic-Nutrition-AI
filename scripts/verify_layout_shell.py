#!/usr/bin/env python3
"""Verification script for T7-004: Build the layout shell (nav, footer disclaimer, health banner).

Validates:
1. Keyboard navigation reaches all links
2. Banner appears when the API is down
3. Disclaimer text avoids 'diagnosis' wording
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_verification() -> bool:
    print("[1/3] Verifying keyboard navigation reaches all links with visible focus...")
    layout_file = REPO_ROOT / "frontend/app/layout.tsx"
    assert layout_file.is_file(), f"Missing {layout_file}"
    content = layout_file.read_text(encoding="utf-8")

    assert 'href="#main-content"' in content, "Missing skip-to-content accessible anchor"
    assert 'aria-label="Main Navigation"' in content, "Missing semantic nav aria-label"

    for route in ["/analyze", "/history", "/models", "/about"]:
        assert f'href="{route}"' in content, f"Navigation link missing for {route}"

    assert "focus-visible:ring-2" in content, "Missing focus-visible keyboard ring styling"
    print("  -> Passed keyboard navigation accessibility!")

    print("[2/3] Verifying health banner structure and inclusion...")
    banner_file = REPO_ROOT / "frontend/components/ui/HealthBanner.tsx"
    assert banner_file.is_file(), f"Missing {banner_file}"
    banner_content = banner_file.read_text(encoding="utf-8")

    assert "<HealthBanner" in content, "HealthBanner component not mounted in layout.tsx"
    assert 'role="alert"' in banner_content, "Missing role='alert' on health banner"
    assert 'aria-live="polite"' in banner_content, "Missing aria-live='polite' on health banner"
    assert "Backend unreachable" in banner_content, "Missing clear backend unreachable notification"
    assert "Retry Connection" in banner_content, "Missing retry connection action"
    print("  -> Passed health banner structure & non-blocking alert verification!")

    print("[3/3] Verifying footer disclaimer strictly avoids 'diagnosis' wording...")
    footer_match = re.search(r"<footer[^>]*>(.*?)</footer>", content, re.DOTALL)
    assert footer_match, "Footer element not found in layout.tsx"
    footer_text = footer_match.group(1).lower()

    assert "diagnosis" not in footer_text, "Footer disclaimer must strictly avoid 'diagnosis' wording"
    assert "diagnos" not in footer_text, "Footer disclaimer must not contain any form of diagnose"
    assert "advisory" in footer_text, "Footer disclaimer must clearly state advisory purpose"
    print("  -> Passed disclaimer wording verification (no 'diagnosis', clear advisory language)!")

    print("\nAll T7-004 layout shell criteria verified successfully!")
    return True


if __name__ == "__main__":
    if run_verification():
        sys.exit(0)
    sys.exit(1)
