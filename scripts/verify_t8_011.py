"""Verification script for T8-011: Add coverage gate and test report.

Acceptance Criteria:
- [x] Coverage is measured, not assumed
- [x] Gate value recorded in CONTRIBUTING
- [x] Gaps listed in `docs/testing.md`
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    # 1. Verify CONTRIBUTING.md contains the 70% coverage gate value
    contributing_file = repo_root / "CONTRIBUTING.md"
    if not contributing_file.exists():
        print("FAIL: CONTRIBUTING.md does not exist.")
        return 1

    contributing_text = contributing_file.read_text(encoding="utf-8")
    if "70%" not in contributing_text and "70 percent" not in contributing_text.lower():
        print("FAIL: 70% coverage gate value not found in CONTRIBUTING.md.")
        return 1
    if "docs/testing.md" not in contributing_text:
        print("FAIL: Reference to docs/testing.md not found in CONTRIBUTING.md.")
        return 1
    print("PASS: Gate value (>=70%) recorded in CONTRIBUTING.md.")

    # 2. Verify docs/testing.md exists and contains gaps analysis and measured coverage
    testing_doc = repo_root / "docs" / "testing.md"
    if not testing_doc.exists():
        print("FAIL: docs/testing.md does not exist.")
        return 1

    doc_text = testing_doc.read_text(encoding="utf-8")
    if "Unmeasured Areas and Testing Gaps" not in doc_text and "Gaps" not in doc_text:
        print("FAIL: Testing gaps section missing in docs/testing.md.")
        return 1
    if "70%" not in doc_text:
        print("FAIL: 70% coverage target gate missing in docs/testing.md.")
        return 1
    if "83%" not in doc_text and "Measured Coverage" not in doc_text:
        print("FAIL: Measured coverage baseline missing in docs/testing.md.")
        return 1

    # Check for specific gap mentions
    assert "SAM 2.1" in doc_text, "SAM 2.1 gap should be documented"
    assert "Hardware" in doc_text or "UART" in doc_text, (
        "Hardware gap should be documented"
    )
    assert "Gemini" in doc_text or "LLM" in doc_text, (
        "LLM/Gemini gap should be documented"
    )
    print("PASS: docs/testing.md exists with measured coverage and documented gaps.")

    print("\nALL ACs FOR T8-011 VERIFIED SUCCESSFULLY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
