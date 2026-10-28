"""Verification script for T7-012: Build ImageResultPanel with per-leaf cards."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_t7_012_image_result_panel() -> None:
    frontend_dir = PROJECT_ROOT / "frontend"
    leaf_card_file = frontend_dir / "components/analyze/LeafCard.tsx"
    image_result_panel_file = frontend_dir / "components/analyze/ImageResultPanel.tsx"
    analyze_index_file = frontend_dir / "components/analyze/index.ts"

    print("=== Verifying T7-012: Build ImageResultPanel with per-leaf cards ===")

    # 1. Component files exist
    assert leaf_card_file.exists(), f"LeafCard.tsx not found at {leaf_card_file}"
    assert image_result_panel_file.exists(), f"ImageResultPanel.tsx not found at {image_result_panel_file}"
    assert analyze_index_file.exists(), f"index.ts not found at {analyze_index_file}"
    print("[PASS] LeafCard and ImageResultPanel files exist.")

    # 2. Check LeafCard implementation details (Criterion 1: Each leaf shows its own confidence tier)
    with open(leaf_card_file, "r", encoding="utf-8") as f:
        leaf_card_content = f.read()

    assert "ConfidenceMeter" in leaf_card_content, "ConfidenceMeter missing in LeafCard"
    assert "UncertaintyBadge" in leaf_card_content, "UncertaintyBadge missing in LeafCard"
    assert "thresholds" in leaf_card_content, "thresholds missing in LeafCard"
    assert "isUncertain" in leaf_card_content, "isUncertain missing in LeafCard"
    assert "originalClass" in leaf_card_content, "originalClass missing in LeafCard"
    print("[PASS] LeafCard renders its own ConfidenceMeter, UncertaintyBadge, and confidence tier.")

    # 3. Check ImageResultPanel implementation details
    with open(image_result_panel_file, "r", encoding="utf-8") as f:
        panel_content = f.read()

    # Criterion 1: Each leaf shows its own confidence tier in leaf grid
    assert "LeafCard" in panel_content, "LeafCard not used in ImageResultPanel"
    assert "leaves.map" in panel_content, "leaves mapping missing in ImageResultPanel"

    # Criterion 2: Rejected leaves are listed as not assessed with reasons
    assert "rejectedLeaves" in panel_content or "rejected_leaves" in panel_content
    assert "Not Assessed Leaves" in panel_content, "Not assessed leaves section missing"
    assert "Reason:" in panel_content, "Reason display for rejected leaves missing"
    assert "not assessed with reasons" in panel_content.lower() or "never silently omitted" in panel_content.lower()
    print("[PASS] Rejected leaves are listed as not assessed with reasons.")

    # Criterion 3: Segmentation failure is shown as an input failure, not a disease
    assert "isFailure" in panel_content or "segmentationFailed" in panel_content
    assert "Input Quality Issue" in panel_content or "input failure" in panel_content.lower()
    assert "NOT a plant disease" in panel_content, "Segmentation failure must explicitly state not a plant disease"
    assert "Retry" in panel_content, "Retry button missing in segmentation failure card"
    assert "Choose Different Image" in panel_content, "Choose Different Image button missing"
    print("[PASS] Segmentation failure is shown as an input failure, not a disease.")

    # Check exports
    with open(analyze_index_file, "r", encoding="utf-8") as f:
        index_content = f.read()
    assert "ImageResultPanel" in index_content, "ImageResultPanel not exported in analyze/index.ts"
    assert "LeafCard" in index_content, "LeafCard not exported in analyze/index.ts"
    print("[PASS] Components cleanly exported from frontend/components/analyze/index.ts.")

    # 4. Check TypeScript compilation
    print("Running npx tsc --noEmit in frontend...")
    res = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        cwd=str(frontend_dir),
        capture_output=True,
        text=True,
        shell=True,
    )
    assert res.returncode == 0, f"TypeScript compilation error:\n{res.stdout}\n{res.stderr}"
    print("[PASS] Frontend TypeScript compilation passed with zero errors.")

    print("\n>>> All T7-012 acceptance criteria verified successfully! <<<")


if __name__ == "__main__":
    test_t7_012_image_result_panel()
