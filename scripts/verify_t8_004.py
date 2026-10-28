"""Verification script for T8-004: Thresholds, rules, registry, metrics, CV.

Acceptance Criteria:
- [x] Boundary values for each documented threshold are tested
- [x] Promote/rollback tested
- [x] Group overlap in CV raises
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEST_FILE = (
    REPO_ROOT / "tests" / "unit" / "test_t8_004_thresholds_rules_registry_metrics_cv.py"
)


def verify_unit_tests() -> bool:
    """Run pytest directly on the T8-004 unit test suite."""
    print(f"[1/3] Running pytest on {TEST_FILE.name}...")
    exit_code = pytest.main(
        [
            "-v",
            str(TEST_FILE),
        ]
    )
    if exit_code != 0:
        print(f"FAILED: Pytest exited with code {exit_code}")
        return False
    print("PASS: All unit tests in T8-004 passed successfully.\n")
    return True


def verify_acceptance_criteria_presence() -> bool:
    """Verify that all three acceptance criteria are covered in the test file."""
    print("[2/3] Checking acceptance criteria coverage in test suite...")
    content = TEST_FILE.read_text(encoding="utf-8")

    criteria = [
        (
            "Boundary values for each documented threshold",
            "test_lettuce_action_band_boundaries",
        ),
        (
            "Boundary values for each documented threshold",
            "test_image_uncertainty_threshold_boundaries_from_calibration",
        ),
        ("Promote/rollback tested", "test_promote_and_rollback_lifecycle"),
        (
            "Group overlap in CV raises",
            "test_assert_no_group_overlap_raises_on_overlap",
        ),
    ]

    for label, func_name in criteria:
        if func_name not in content:
            print(f"FAILED: Missing test function '{func_name}' for '{label}'")
            return False
        print(f"  - Verified coverage: {func_name} ({label})")

    print("PASS: All T8-004 acceptance criteria are explicitly tested.\n")
    return True


def verify_ruff_cleanliness() -> bool:
    """Verify ruff linting and formatting on the test file and this verification script."""
    print("[3/3] Checking ruff lint and formatting...")
    files = [str(TEST_FILE), str(Path(__file__).resolve())]
    cmd_check = [sys.executable, "-m", "ruff", "check", *files]
    res_check = subprocess.run(
        cmd_check, capture_output=True, text=True, cwd=str(REPO_ROOT), check=False
    )
    if res_check.returncode != 0:
        print(f"FAILED: ruff check failed:\n{res_check.stdout or res_check.stderr}")
        return False

    cmd_format = [sys.executable, "-m", "ruff", "format", "--check", *files]
    res_format = subprocess.run(
        cmd_format, capture_output=True, text=True, cwd=str(REPO_ROOT), check=False
    )
    if res_format.returncode != 0:
        print(
            f"FAILED: ruff format --check failed:\n{res_format.stdout or res_format.stderr}"
        )
        return False

    print("PASS: Ruff check and format passed cleanly.\n")
    return True


def main() -> int:
    """Execute all verification stages."""
    print(
        "=== Verifying T8-004: Unit tests (thresholds, rules, registry, metrics, CV) ===\n"
    )
    if not verify_unit_tests():
        return 1
    if not verify_acceptance_criteria_presence():
        return 1
    if not verify_ruff_cleanliness():
        return 1

    print("All T8-004 verification steps passed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
