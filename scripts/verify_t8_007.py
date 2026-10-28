#!/usr/bin/env python3
"""Verification script for T8-007: Integration tests: history round trip and pagination.

Validates:
1. Timezone survives round trip (preserves observed time, timezone string, and source provenance).
2. Limit above 100 is capped (pagination limits, offset navigation, and X-Total-Count header).
3. Old record is not marked live ('historical' flag is True, and stale telemetry cannot be treated as live).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest


def run_verification() -> bool:
    print("[1/3] Running T8-007 integration test suite...")
    exit_code = pytest.main([
        "-v",
        str(REPO_ROOT / "tests/integration/test_history_round_trip_pagination.py"),
    ])
    if exit_code != 0:
        print(f"FAILED: pytest exited with code {exit_code}")
        return False

    print("\nAll T8-007 integration tests passed successfully!")
    return True


if __name__ == "__main__":
    if run_verification():
        sys.exit(0)
    sys.exit(1)
