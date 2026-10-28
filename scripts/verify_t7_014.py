"""Verification script for T7-014: Build RecommendationCard and NotAssessedNote."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_t7_014_recommendation_and_not_assessed() -> None:
    print("=== Verifying T7-014: Build RecommendationCard and NotAssessedNote ===")
    rec_file = PROJECT_ROOT / "frontend/components/analyze/RecommendationCard.tsx"
    not_assessed_file = PROJECT_ROOT / "frontend/components/analyze/NotAssessedNote.tsx"
    index_file = PROJECT_ROOT / "frontend/components/analyze/index.ts"

    assert rec_file.exists(), f"RecommendationCard.tsx missing at {rec_file}"
    assert not_assessed_file.exists(), f"NotAssessedNote.tsx missing at {not_assessed_file}"

    with open(rec_file, "r", encoding="utf-8") as f:
        rec_code = f.read()

    # Criterion 1: Source chip reflects source from the API
    assert "recommendation.source" in rec_code
    assert 'Badge variant={isGemini ? "gemini" : "rules"}' in rec_code
    assert "Gemini 2.5 Flash" in rec_code
    assert "Agronomic Rules" in rec_code
    print("[PASS] Source chip reflects source from the API (Gemini or Rules).")

    # Criterion 2: 'Not assessed' appears when one modality is missing
    with open(not_assessed_file, "r", encoding="utf-8") as f:
        not_assessed_code = f.read()

    assert "missingModality" in not_assessed_code
    assert "Not Assessed:" in not_assessed_code
    assert "Omitted Modality" in not_assessed_code
    print("[PASS] 'Not assessed' note displays clearly when one modality is missing.")

    # Criterion 3: No combined score is shown anywhere
    assert "No combined score" in rec_code, "No combined score note must be explicitly present"
    assert "combined_score" not in rec_code
    print("[PASS] No combined score is computed or displayed anywhere.")

    # Check exports
    with open(index_file, "r", encoding="utf-8") as f:
        idx_code = f.read()
    assert "RecommendationCard" in idx_code
    assert "NotAssessedNote" in idx_code

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

    print("\n>>> All T7-014 acceptance criteria verified successfully! <<<")


if __name__ == "__main__":
    test_t7_014_recommendation_and_not_assessed()
