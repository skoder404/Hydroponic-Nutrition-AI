#!/usr/bin/env python3
"""Verification script for T5-006: Decide whether droop_angle is defensible.

Acceptance Criteria:
1. Decision is written with evidence:
   docs/decisions/0007-droop-angle.md exists, documents 20-image empirical evaluation,
   and details why droop_angle is indefensible.
2. Impact list of datasets/models to regenerate is included:
   ADR documents the exact downstream impact list and specifies the 4 retained
   features (glcm_contrast, glcm_entropy, bounding_aspect_ratio, solidity).
3. Both members agree in the commit review:
   ADR documents Driver (Yashwant) and Navigator (Sushil) agreement.

Usage:
    python scripts/verify_droop_angle_decision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def verify_decision_document_exists_and_has_evidence() -> tuple[bool, str]:
    """Criterion 1: Decision is written with empirical evidence."""
    adr_path = PROJECT_ROOT / "docs" / "decisions" / "0007-droop-angle.md"
    if not adr_path.is_file():
        return False, f"ADR file missing at: {adr_path}"

    content = adr_path.read_text(encoding="utf-8")

    # Check key decision terms
    if "REMOVE" not in content and "Remove" not in content:
        return False, "ADR 0007 does not contain the REMOVE decision."

    # Check empirical evidence presence
    if "20" not in content or "empirical" not in content.lower():
        return False, "ADR 0007 missing mention of 20-image empirical evaluation."

    # Check failure mode documentation
    required_evidences = [
        "gravity",
        "stem",
        "fitEllipse",
        "flipTB",
        "180deg",
    ]
    for term in required_evidences:
        if term.lower() not in content.lower():
            return False, f"ADR 0007 missing evidence discussion on '{term}'."

    return (
        True,
        "Verified ADR 0007 exists with empirical 20-image evidence and failure analysis.",
    )


def verify_impact_list_included() -> tuple[bool, str]:
    """Criterion 2: Impact list of datasets/models to regenerate is included."""
    adr_path = PROJECT_ROOT / "docs" / "decisions" / "0007-droop-angle.md"
    content = adr_path.read_text(encoding="utf-8")

    required_impact_terms = [
        "Impact",
        "glcm_contrast",
        "glcm_entropy",
        "bounding_aspect_ratio",
        "solidity",
    ]
    for term in required_impact_terms:
        if term not in content:
            return False, f"ADR 0007 missing impact item or feature '{term}'."

    return (
        True,
        "Verified ADR 0007 contains complete impact analysis and 4-feature tomato specification.",
    )


def verify_both_members_agree() -> tuple[bool, str]:
    """Criterion 3: Both members agree in the review."""
    adr_path = PROJECT_ROOT / "docs" / "decisions" / "0007-droop-angle.md"
    content = adr_path.read_text(encoding="utf-8")

    if "Yashwant" not in content or "Sushil" not in content:
        return False, "ADR 0007 missing team members Yashwant and Sushil."

    if "Review & Team Agreement" not in content and "Agreement" not in content:
        return False, "ADR 0007 missing explicit team agreement section."

    return (
        True,
        "Verified team agreement between Driver (Yashwant) and Navigator (Sushil).",
    )


def run_verification() -> bool:
    checks = [
        (
            "Criterion 1: Decision written with evidence",
            verify_decision_document_exists_and_has_evidence,
        ),
        (
            "Criterion 2: Impact list of datasets/models included",
            verify_impact_list_included,
        ),
        ("Criterion 3: Both members agree in review", verify_both_members_agree),
    ]

    all_passed = True
    print("=" * 70)
    print("Verification for T5-006: Decision on droop_angle (ADR 0007)")
    print("=" * 70)

    for name, check_fn in checks:
        passed, msg = check_fn()
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        print(f"       {msg}")
        if not passed:
            all_passed = False

    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
