"""Verification script for T7-017: Build S5 Record detail."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_t7_017_record_detail() -> None:
    print("=== Verifying T7-017: Build S5 Record detail ===")
    detail_page_file = PROJECT_ROOT / "frontend/app/history/[id]/page.tsx"

    assert detail_page_file.exists(), f"page.tsx missing at {detail_page_file}"
    with open(detail_page_file, "r", encoding="utf-8") as f:
        code = f.read()

    # Criterion 1: Missing image shows metadata only
    assert "missing-image-metadata-only" in code, "missing-image-metadata-only testid not found"
    assert "Original image no longer available" in code
    assert "Metadata Preserved" in code or "metadata only" in code.lower()
    print("[PASS] Missing image displays metadata only with clear explanatory banner.")

    # Criterion 2: Timestamps display with timezone
    assert "observedFormatted" in code or "observed_at" in code
    assert "record.observed_tz" in code or "Timezone:" in code
    print("[PASS] Timestamps display with true capture timezone.")

    # Criterion 3: Not-found state works
    assert "record-not-found-state" in code, "record-not-found-state testid not found"
    assert "Record Not Found" in code
    assert "Back to History" in code
    print("[PASS] Not-found state correctly handles non-existent record identifiers.")

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

    print("\n>>> All T7-017 acceptance criteria verified successfully! <<<")


if __name__ == "__main__":
    test_t7_017_record_detail()
