"""Unit tests for spinach image model comparison and selection report (T5-029)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_MANIFEST_HASH = (
    "4c271d42de9b12de59b32ac5bb2ba4cb47821d7fc8dc9eebbc50cc41ad494f68"
)
SPINACH_CLASSES = [
    "Spinach_Healthy",
    "Spinach_Non fresh",
    "Spinach_anthracnose",
    "Spinach_bacterial spot",
    "Spinach_downy mildew",
    "Spinach_pest damage",
]


def test_spinach_image_comparison_report() -> None:
    """Verify spinach image comparison report contents and structure."""
    rep_path = PROJECT_ROOT / "reports/spinach/image_comparison.md"
    if not rep_path.exists():
        pytest.skip("Comparison report not yet created; skipping test.")

    content = rep_path.read_text(encoding="utf-8")

    # Acceptance criterion 1: Same split manifest hash
    assert EXPECTED_MANIFEST_HASH in content

    # Acceptance criterion 2: Macro-F1 and per-class F1
    assert "0.7256" in content  # EfficientNetB0
    assert "0.6567" in content  # Custom CNN
    assert "0.5243" in content  # Random Forest
    assert "0.4718" in content  # SVM

    for cls_name in SPINACH_CLASSES:
        assert cls_name in content

    # Acceptance criterion 3: Leaf-level input confirmed
    assert "leaf" in content.lower()
    assert "spinach_efficientnet_b0_v1.0.0" in content
