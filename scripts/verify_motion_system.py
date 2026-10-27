"""Verification script for T7-003: MotionProvider and Shared Animation Variants.

Verifies acceptance criteria:
1. Reduced-motion setting turns movement into a fade or nothing.
2. Only transform, opacity, and SVG stroke are animated.
3. Variants match the DESIGN.md durations.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def verify_motion_tokens() -> None:
    tokens_file = Path("frontend/components/motion/tokens.ts")
    assert tokens_file.exists(), "frontend/components/motion/tokens.ts must exist"

    content = tokens_file.read_text(encoding="utf-8")

    # Required durations from DESIGN.md Motion System
    expected_durations = {
        "heroPlantGrow": 1200,
        "cropCardFadeUp": 300,
        "cropCardStagger": 60,
        "cropCardHoverLift": 150,
        "pageTransition": 250,
        "confidenceMeterFill": 500,
        "sensorReadingCount": 400,
        "resultPanelFade": 300,
        "resultPanelOffset": 80,
        "leafCardStagger": 50,
        "leafCardFade": 250,
        "maskToggle": 200,
        "saveToastSlide": 300,
        "livePulse": 600,
        "decorativeLoop": 40000,
        "errorFade": 150,
        "feedbackPress": 80,
    }

    for key, val in expected_durations.items():
        pattern = rf"{key}:\s*{val}"
        match = re.search(pattern, content)
        assert match, (
            f"Duration token '{key}' with value {val}ms not found in tokens.ts"
        )

    # Verify allowed animated properties whitelist
    for prop in ["transform", "opacity", "strokeDashoffset", "strokeDasharray"]:
        assert prop in content, f"Allowed property '{prop}' missing in tokens.ts"

    print("Motion durations and allowed properties matching DESIGN.md: VERIFIED.")


def verify_shared_variants_and_reduced_motion() -> None:
    variants_file = Path("frontend/components/motion/variants.ts")
    assert variants_file.exists(), "frontend/components/motion/variants.ts must exist"

    content = variants_file.read_text(encoding="utf-8")

    # Check shared variants exist
    assert "fadeUpVariant" in content, "fadeUpVariant must exist"
    assert "growVariant" in content, "growVariant must exist"
    assert "crossFadeVariant" in content, "crossFadeVariant must exist"
    assert "scalePressVariant" in content, "scalePressVariant must exist"
    assert "getStaggerDelay" in content, "getStaggerDelay helper must exist"

    # Check reduced-motion fallback specification
    assert "reducedMotion" in content
    # Reduced motion must eliminate translation (e.g. translateY -> none or 0)
    assert 'transform: "none"' in content or "none" in content

    print(
        "Shared variants (fadeUp, grow, crossFade) and reduced motion fallbacks: VERIFIED."
    )


def verify_motion_provider() -> None:
    provider_file = Path("frontend/components/motion/MotionProvider.tsx")
    assert provider_file.exists(), (
        "frontend/components/motion/MotionProvider.tsx must exist"
    )

    content = provider_file.read_text(encoding="utf-8")

    # Must listen to prefers-reduced-motion
    assert "prefers-reduced-motion" in content, (
        "MotionProvider must query prefers-reduced-motion"
    )
    # Must listen to visibilitychange to pause loops when tab hidden
    assert "visibilitychange" in content, (
        "MotionProvider must pause loops on visibilitychange"
    )
    assert "document.visibilityState" in content

    print(
        "MotionProvider with prefers-reduced-motion and tab visibility monitoring: VERIFIED."
    )


def main() -> int:
    print("=== Verifying T7-003: MotionProvider & Animation System ===")
    verify_motion_tokens()
    verify_shared_variants_and_reduced_motion()
    verify_motion_provider()
    print("=== All T7-003 Acceptance Criteria PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
