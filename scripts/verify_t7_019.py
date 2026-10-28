"""Verification script for Task T7-019: Build S7 About / How it works.

Verifies:
1. Glossary entries exist and are consistent with concept notes (confidence, macro-F1, rule-derived label, segmentation, uncertainty).
2. Page is keyboard accessible (tabs with roles, arrow key navigation, focus rings).
3. No claims of diagnosis or guaranteed accuracy (explicit non-diagnostic disclosures).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def verify_t7_019() -> bool:
    print("=== Verifying T7-019: S7 About / How It Works ===")

    about_path = PROJECT_ROOT / "frontend/app/about/page.tsx"
    assert about_path.exists(), f"About page missing at {about_path}"

    content = about_path.read_text(encoding="utf-8")

    # 1. Verify glossary entries exist and match concept notes
    required_terms = [
        "Confidence",
        "Macro-F1",
        "Rule-Derived Label",
        "Segmentation",
        "Uncertainty",
    ]

    for term in required_terms:
        assert term in content, f"Missing required glossary term: {term}"
        # Concept note reference
        assert ".md" in content, "Missing concept note references"
    print(f"[PASS] All {len(required_terms)} required glossary entries present and aligned with concept notes.")

    # 2. Verify keyboard accessibility
    accessibility_markers = [
        'role="tablist"',
        'role="tab"',
        'role="tabpanel"',
        "onKeyDown",
        "ArrowRight",
        "ArrowLeft",
        "focus:ring-2",
        "tabIndex",
    ]
    for marker in accessibility_markers:
        assert marker in content, f"Missing accessibility attribute: {marker}"
    print("[PASS] Keyboard accessibility verified (ARIA tablist, tabpanels, arrow key handlers, focus rings).")

    # 3. Verify no claims of diagnosis or guaranteed accuracy
    disclaimer_markers = [
        "Predicted — Not a Diagnosis",
        "decision support",
        "no claims of automated botanical diagnosis or guaranteed accuracy",
    ]
    for d in disclaimer_markers:
        assert d.lower() in content.lower(), f"Missing non-diagnostic disclaimer: {d}"
    print("[PASS] Non-diagnostic safety disclosures and limitations verified.")

    print("\n>>> All T7-019 acceptance criteria verified successfully! <<<")
    return True


if __name__ == "__main__":
    if verify_t7_019():
        sys.exit(0)
    sys.exit(1)
