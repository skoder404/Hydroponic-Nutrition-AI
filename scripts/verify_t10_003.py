"""Verification script for T10-003: Probability calibration experiment.

Acceptance Criteria:
- [x] Calibration is fit on validation only
- [x] Before/after reliability plots saved
- [x] Decision recorded in an ADR
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEST_FILE = REPO_ROOT / "tests" / "unit" / "test_t10_003_probability_calibration.py"
REPORT_JSON = REPO_ROOT / "reports" / "calibration" / "calibration_experiment.json"
PLOT_PNG = REPO_ROOT / "reports" / "calibration" / "t10_003_reliability_plots.png"
ADR_MD = REPO_ROOT / "docs" / "decisions" / "0009-probability-calibration.md"


def verify_unit_tests() -> bool:
    """Run pytest directly on T10-003 unit test file."""
    print(f"[1/4] Running pytest on {TEST_FILE.name}...")
    exit_code = pytest.main(
        [
            "-v",
            str(TEST_FILE),
        ]
    )
    if exit_code != 0:
        print(f"FAILED: Pytest exited with code {exit_code}")
        return False
    print("PASS: All unit tests in T10-003 passed.\n")
    return True


def verify_validation_fit_only() -> bool:
    """Verify that calibration was strictly fit on validation data."""
    print("[2/4] Verifying calibration split is validation-only...")
    if not REPORT_JSON.is_file():
        print(f"FAILED: Report JSON does not exist at {REPORT_JSON}")
        return False

    with open(REPORT_JSON, encoding="utf-8") as f:
        data = json.load(f)

    if data.get("split_evaluated") != "validation":
        print(
            f"FAILED: Overall split_evaluated is not validation: {data.get('split_evaluated')}"
        )
        return False

    if data.get("image_model", {}).get("split_evaluated") != "validation":
        print("FAILED: Image model split_evaluated is not validation")
        return False

    if data.get("sensor_model", {}).get("split_evaluated") != "validation":
        print("FAILED: Sensor model split_evaluated is not validation")
        return False

    print("PASS: Calibration strictly fit on validation split only.\n")
    return True


def verify_artifacts_and_adr() -> bool:
    """Verify that before/after reliability plots exist and ADR is recorded."""
    print("[3/4] Verifying reliability plot and ADR...")
    if not PLOT_PNG.is_file() or PLOT_PNG.stat().st_size < 50_000:
        print(f"FAILED: Reliability plot missing or too small at {PLOT_PNG}")
        return False

    if not ADR_MD.is_file():
        print(f"FAILED: ADR file missing at {ADR_MD}")
        return False

    adr_text = ADR_MD.read_text(encoding="utf-8")
    if "ADR 0009" not in adr_text or "RETAIN 'model score'" not in adr_text:
        print("FAILED: ADR 0009 missing required decision statement")
        return False

    print("PASS: Reliability plots and ADR 0009 verified.\n")
    return True


def verify_ruff() -> bool:
    """Verify ruff linting and formatting."""
    print("[4/4] Checking ruff lint and formatting...")
    files = [
        str(REPO_ROOT / "scripts" / "experiment_probability_calibration.py"),
        str(TEST_FILE),
        str(Path(__file__).resolve()),
    ]
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
        print(f"FAILED: ruff format failed:\n{res_format.stdout or res_format.stderr}")
        return False

    print("PASS: Ruff lint and format passed cleanly.\n")
    return True


def main() -> int:
    print("=== Verifying T10-003: Probability calibration experiment ===\n")
    if not verify_unit_tests():
        return 1
    if not verify_validation_fit_only():
        return 1
    if not verify_artifacts_and_adr():
        return 1
    if not verify_ruff():
        return 1

    print("All T10-003 verification steps passed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
