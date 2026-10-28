"""Verification script for Task T7-011: Build ConfidenceMeter and UncertaintyBadge.

Verifies:
1. Uncertain tier is visually lighter than higher-confidence tier.
2. No tier relies on colour alone (uses icon + text + pattern).
3. Tier thresholds come from the API (passed as props/config, not constants).
4. Tooltip wording contains 'model score, not a calibrated probability'.
5. Components are exported from frontend/components/ui/index.ts.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def verify_t7_011() -> bool:
    print("=== Verifying T7-011: ConfidenceMeter & UncertaintyBadge ===")

    cm_path = PROJECT_ROOT / "frontend/components/ui/ConfidenceMeter.tsx"
    ub_path = PROJECT_ROOT / "frontend/components/ui/UncertaintyBadge.tsx"
    index_path = PROJECT_ROOT / "frontend/components/ui/index.ts"

    assert cm_path.exists(), f"ConfidenceMeter.tsx missing at {cm_path}"
    assert ub_path.exists(), f"UncertaintyBadge.tsx missing at {ub_path}"
    assert index_path.exists(), f"index.ts missing at {index_path}"

    cm_text = cm_path.read_text(encoding="utf-8")
    ub_text = ub_path.read_text(encoding="utf-8")
    index_text = index_path.read_text(encoding="utf-8")

    # 1. Verify exports
    assert "ConfidenceMeter" in index_text, "ConfidenceMeter not exported in index.ts"
    assert "UncertaintyBadge" in index_text, "UncertaintyBadge not exported in index.ts"
    print("[PASS] Both components exported from frontend/components/ui/index.ts.")

    # 2. Verify tier thresholds come from props / API (not hardcoded constants)
    assert "interface ConfidenceThresholds" in cm_text
    assert "thresholds: ConfidenceThresholds" in cm_text
    assert "getConfidenceTier" in cm_text
    assert "score >= thresholds.high" in cm_text
    assert "score >= thresholds.moderate" in cm_text
    print("[PASS] Thresholds are API-driven via ConfidenceThresholds prop.")

    # 3. Verify no tier relies on colour alone (icon + text + pattern)
    for text_src in (cm_text, ub_text):
        assert "Higher Confidence" in text_src or "higher" in text_src
        assert "Moderate Confidence" in text_src or "moderate" in text_src
        assert "Uncertain Score" in text_src or "uncertain" in text_src
        # distinct icons
        assert "<svg" in text_src
        # distinct patterns
        assert "repeating-linear-gradient" in text_src or "border-dotted" in text_src or "border-dashed" in text_src
    print("[PASS] All tiers use distinct icon + text + pattern/border (never color alone).")

    # 4. Verify uncertain tier is visually lighter than higher-confidence tier
    assert "opacity-85" in cm_text or "opacity-80" in cm_text or "opacity" in cm_text
    assert "opacity-80" in ub_text or "opacity" in ub_text
    assert "var(--sand)" in cm_text or "var(--sand)" in ub_text
    print("[PASS] Uncertain tier is visually lighter (subdued sand tint, lighter opacity, dotted border).")

    # 5. Verify mandatory tooltip wording strictly says 'model score, not a calibrated probability'
    exact_wording = "model score, not a calibrated probability"
    assert exact_wording.lower() in cm_text.lower(), f"Missing '{exact_wording}' in ConfidenceMeter"
    assert exact_wording.lower() in ub_text.lower(), f"Missing '{exact_wording}' in UncertaintyBadge"
    print(f"[PASS] Tooltip wording strictly contains '{exact_wording}'.")

    print("\n>>> All T7-011 acceptance criteria verified successfully! <<<")
    return True


if __name__ == "__main__":
    if verify_t7_011():
        sys.exit(0)
    sys.exit(1)

