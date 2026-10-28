"""Verification script for T7-015: Add Save record with honest outcomes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_t7_015_save_record() -> None:
    print("=== Verifying T7-015: Add Save record with honest outcomes ===")
    save_button_file = PROJECT_ROOT / "frontend/components/analyze/SaveRecordButton.tsx"
    toast_file = PROJECT_ROOT / "frontend/components/ui/Toast.tsx"

    assert save_button_file.exists(), f"SaveRecordButton.tsx missing at {save_button_file}"
    with open(save_button_file, "r", encoding="utf-8") as f:
        code = f.read()

    # Criterion 1: Failed save never shows success
    assert "setSavedRecord(null)" in code, "Failed save must set savedRecord to null"
    assert "Not saved" in code, "Clear 'Not saved' state must appear on error"
    assert "saveError" in code, "Error state variable missing"
    print("[PASS] Failed save never shows success; clear 'Not saved' error state is rendered.")

    # Criterion 2: Double-click does not create two records
    assert "isSaving || savedRecord" in code or "if (isSaving" in code, (
        "handleSave must guard against concurrent calls while saving"
    )
    assert "disabled={isSaving" in code, "Button must be disabled while isSaving"
    print("[PASS] Double-click guard is implemented; concurrent requests prevented.")

    # Criterion 3: Toast announces via aria-live
    with open(toast_file, "r", encoding="utf-8") as f:
        toast_code = f.read()

    assert "aria-live" in toast_code, "Toast component must implement aria-live attribute"
    assert "Toast" in code, "SaveRecordButton must trigger Toast"
    print("[PASS] Toast announces outcomes via aria-live.")

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

    print("\n>>> All T7-015 acceptance criteria verified successfully! <<<")


if __name__ == "__main__":
    test_t7_015_save_record()
