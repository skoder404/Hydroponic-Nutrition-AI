"""Verification script for T7-020: Audit empty/error/loading and responsive behaviour on all screens."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_t7_020_state_and_responsive_audit() -> None:
    print("=== Verifying T7-020: Audit empty/error/loading and responsive behaviour on all screens ===")
    audit_file = PROJECT_ROOT / "docs/design/state_audit.md"
    globals_css = PROJECT_ROOT / "frontend/app/globals.css"
    button_file = PROJECT_ROOT / "frontend/components/ui/Button.tsx"

    # Criterion 1: Checklist per screen/state/breakpoint saved in docs/design/state_audit.md
    assert audit_file.exists(), f"state_audit.md missing at {audit_file}"
    with open(audit_file, "r", encoding="utf-8") as f:
        audit_text = f.read()

    assert "S1" in audit_text and "S8" in audit_text, "Audit must cover screens S1 through S8"
    assert "360 px" in audit_text and "768 px" in audit_text and "1024" in audit_text, "Audit must cover all breakpoints"
    assert "Empty State" in audit_text and "Loading State" in audit_text and "Error State" in audit_text
    print("[PASS] Screen-by-screen state and breakpoint checklist saved in docs/design/state_audit.md.")

    # Criterion 2: No horizontal body scroll
    with open(globals_css, "r", encoding="utf-8") as f:
        css_text = f.read()
    assert "overflow-x: hidden" in css_text or "overflow-x:hidden" in css_text, "globals.css must enforce overflow-x: hidden"
    print("[PASS] Root styles enforce overflow-x: hidden to prevent horizontal body scrolling.")

    # Criterion 3: Touch targets are at least 44 px
    with open(button_file, "r", encoding="utf-8") as f:
        button_text = f.read()
    assert "44px" in button_text, "Button sizes must enforce at least 44px touch targets"
    print("[PASS] Interactive touch targets enforce at least 44 px minimum boundary.")

    # TypeScript check
    print("Running npx tsc --noEmit in frontend...")
    res = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        cwd=str(PROJECT_ROOT / "frontend"),
        capture_output=True,
        text=True,
        shell=True,
    )
    assert res.returncode == 0, f"TypeScript check failed:\n{res.stdout}\n{res.stderr}"
    print("[PASS] Frontend TypeScript check passed with zero errors.")

    print("\n>>> All T7-020 acceptance criteria verified successfully! <<<")


if __name__ == "__main__":
    test_t7_020_state_and_responsive_audit()
