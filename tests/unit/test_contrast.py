"""Unit tests for WCAG 2.1 AA token pair contrast verification (T7-002).

Acceptance Criteria:
- [x] Text pairs are at least 4.5:1 and UI components at least 3:1
- [x] Updated hex values are reflected in DESIGN.md
- [x] Results table is saved in docs/design/contrast.md
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def srgb_to_lin(val: float) -> float:
    c = val / 255.0
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_str: str) -> float:
    h = hex_str.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return 0.2126 * srgb_to_lin(r) + 0.7152 * srgb_to_lin(g) + 0.0722 * srgb_to_lin(b)


def contrast_ratio(hex1: str, hex2: str) -> float:
    l1 = relative_luminance(hex1)
    l2 = relative_luminance(hex2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def extract_css_tokens(css_path: Path) -> dict[str, str]:
    text = css_path.read_text(encoding="utf-8")
    matches = re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6});", text)
    return dict(matches)


def test_text_pairs_are_at_least_4_5_to_1():
    """Verify all text token pairs are at least 4.5:1 against bg, surface, and sand."""
    tokens = extract_css_tokens(ROOT / "frontend/app/globals.css")

    # Tokens used for text / typography
    text_tokens = [
        "--forest",
        "--moss",
        "--clay",
        "--bark",
        "--stone",
        "--water",
        "--success",
        "--warning",
        "--error",
        "--info",
    ]

    backgrounds = ["--bg", "--surface", "--sand"]

    for bg_name in backgrounds:
        bg_hex = tokens[bg_name]
        for fg_name in text_tokens:
            fg_hex = tokens[fg_name]
            ratio = contrast_ratio(fg_hex, bg_hex)
            assert ratio >= 4.5, (
                f"Contrast failure: {fg_name} ({fg_hex}) on {bg_name} ({bg_hex}) "
                f"has ratio {ratio:.2f}:1 (< 4.5:1 requirement)"
            )


def test_white_text_on_fills_is_at_least_4_5_to_1():
    """Verify white text on solid buttons/chips is at least 4.5:1."""
    tokens = extract_css_tokens(ROOT / "frontend/app/globals.css")
    button_fills = [
        "--forest",
        "--moss",
        "--clay",
        "--bark",
        "--water",
        "--success",
        "--warning",
        "--error",
        "--info",
    ]
    for fill_name in button_fills:
        fill_hex = tokens[fill_name]
        ratio = contrast_ratio("#FFFFFF", fill_hex)
        assert ratio >= 4.5, (
            f"Contrast failure: white text on {fill_name} ({fill_hex}) "
            f"has ratio {ratio:.2f}:1 (< 4.5:1)"
        )


def test_ui_components_are_at_least_3_to_1():
    """Verify UI components (borders and focus rings) are at least 3.0:1."""
    tokens = extract_css_tokens(ROOT / "frontend/app/globals.css")
    ui_tokens = ["--border-strong", "--ring-color"]
    backgrounds = ["--bg", "--surface", "--sand"]

    for bg_name in backgrounds:
        bg_hex = tokens[bg_name]
        for ui_name in ui_tokens:
            ui_hex = tokens[ui_name]
            ratio = contrast_ratio(ui_hex, bg_hex)
            assert ratio >= 3.0, (
                f"UI component contrast failure: {ui_name} ({ui_hex}) on {bg_name} ({bg_hex}) "
                f"has ratio {ratio:.2f}:1 (< 3.0:1 requirement)"
            )


def test_updated_hex_values_reflected_in_design_md():
    """Verify updated --clay hex (#A24E2D) is documented in DESIGN.md."""
    design_md = (ROOT / "docs/DESIGN.md").read_text(encoding="utf-8")
    assert "#A24E2D" in design_md
    assert "| `--clay` | `#A24E2D` |" in design_md


def test_contrast_audit_report_exists():
    """Verify docs/design/contrast.md exists and contains the results table."""
    report_path = ROOT / "docs/design/contrast.md"
    assert report_path.is_file(), "docs/design/contrast.md must exist"
    content = report_path.read_text(encoding="utf-8")
    assert "WCAG 2.1 AA Colour Contrast Audit" in content
    assert "--clay" in content
    assert "#A24E2D" in content
    assert "9.06:1" in content
