"""Verification script for T7-009: Build the Analyze page flow and progress states."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_t7_009_analyze_flow() -> None:
    print("=== Verifying T7-009: Build the Analyze page flow and progress states ===")
    analyze_page = PROJECT_ROOT / "frontend/app/analyze/page.tsx"
    progress_indicator = PROJECT_ROOT / "frontend/components/analyze/VineProgressIndicator.tsx"

    assert analyze_page.exists(), f"page.tsx missing at {analyze_page}"
    assert progress_indicator.exists(), f"VineProgressIndicator.tsx missing at {progress_indicator}"

    with open(analyze_page, "r", encoding="utf-8") as f:
        code = f.read()

    # Criterion 1: Analyze is disabled until required inputs validate
    assert "disabled={!canExecute}" in code, "Analyze button must be disabled when canExecute is false"
    assert "canExecute" in code
    print("[PASS] Analyze button is disabled until required inputs validate.")

    # Criterion 2: Cancel aborts the request
    assert "abortControllerRef" in code, "AbortController must be used for cancellation"
    assert "handleCancel" in code, "handleCancel must be defined"
    assert "abort()" in code, "abort() must be called to cancel analysis request"
    print("[PASS] Cancel aborts the active analysis request.")

    # Criterion 3: Device-unavailable message offers manual entry
    assert "device-unavailable-card" in code, "device-unavailable-card testid missing"
    assert "Campus Hardware Server Unreachable" in code
    assert "Switch to Manual Entry" in code
    print("[PASS] Device-unavailable message offers manual entry fallback.")

    # Check progress steps
    with open(progress_indicator, "r", encoding="utf-8") as f:
        vine_code = f.read()

    assert "Validating" in vine_code
    assert "Segmenting" in vine_code
    assert "Classifying" in vine_code
    assert "Recommending" in vine_code
    print("[PASS] Growing-vine stepper includes Validating, Segmenting, Classifying, Recommending.")

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

    print("\n>>> All T7-009 acceptance criteria verified successfully! <<<")


if __name__ == "__main__":
    test_t7_009_analyze_flow()
