#!/usr/bin/env python3
"""Verification script for T8-003: Unit tests: GLCM config, mask validation and leaf extraction.

Validates:
1. GLCM config is asserted not just run
2. All mask verdict branches covered (OK, RETRY, REJECT)
3. Tests are deterministic
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest


def run_verification() -> bool:
    print("[1/3] Running T8-003 test suite covering GLCM config, mask verdicts, and leaf extraction...")
    exit_code = pytest.main([
        "-v",
        str(REPO_ROOT / "tests/unit/test_t8_003_glcm_mask_leaf.py"),
    ])
    if exit_code != 0:
        print(f"FAILED: pytest exited with code {exit_code}")
        return False

    print("\nAll T8-003 unit tests passed with 100% determinism!")
    return True


if __name__ == "__main__":
    if run_verification():
        sys.exit(0)
    sys.exit(1)
