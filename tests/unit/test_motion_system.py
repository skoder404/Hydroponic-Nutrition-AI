"""Unit tests for frontend MotionProvider and shared animation variants (T7-003)."""

from pathlib import Path


def test_motion_tokens_match_design_md():
    tokens_path = Path("frontend/components/motion/tokens.ts")
    assert tokens_path.exists()
    content = tokens_path.read_text(encoding="utf-8")

    # Key durations from DESIGN.md Motion System
    assert "heroPlantGrow: 1200" in content
    assert "cropCardFadeUp: 300" in content
    assert "cropCardStagger: 60" in content
    assert "pageTransition: 250" in content
    assert "confidenceMeterFill: 500" in content
    assert "maskToggle: 200" in content
    assert "errorFade: 150" in content
    assert "decorativeLoop: 40000" in content


def test_allowed_properties_whitelist():
    tokens_path = Path("frontend/components/motion/tokens.ts")
    content = tokens_path.read_text(encoding="utf-8")

    # In accordance with DESIGN.md rule 4: animate ONLY transform, opacity & SVG stroke
    for prop in ["transform", "opacity", "strokeDashoffset"]:
        assert prop in content


def test_shared_variants_reduced_motion_fallback():
    variants_path = Path("frontend/components/motion/variants.ts")
    assert variants_path.exists()
    content = variants_path.read_text(encoding="utf-8")

    assert "fadeUpVariant" in content
    assert "growVariant" in content
    assert "crossFadeVariant" in content
    assert "reducedMotion" in content


def test_motion_provider_monitors_reduced_motion_and_visibility():
    provider_path = Path("frontend/components/motion/MotionProvider.tsx")
    assert provider_path.exists()
    content = provider_path.read_text(encoding="utf-8")

    assert "prefers-reduced-motion" in content
    assert "visibilitychange" in content
    assert "document.visibilityState" in content
