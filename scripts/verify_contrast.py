#!/usr/bin/env python3
"""Verification script for T7-002: Verify contrast for every token pair (WCAG AA).

Validates:
1. Text pairs are at least 4.5:1 and UI components at least 3:1
2. Updated hex values are reflected in DESIGN.md
3. Results table is saved in docs/design/contrast.md
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def run_verification() -> bool:
    print("[1/3] Verifying all text pairs >= 4.5:1 and UI components >= 3.0:1...")
    css_tokens = extract_css_tokens(REPO_ROOT / "frontend/app/globals.css")

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

    for bg in backgrounds:
        for fg in text_tokens:
            cr = contrast_ratio(css_tokens[fg], css_tokens[bg])
            assert cr >= 4.5, f"Text contrast failure: {fg} on {bg} ({cr:.2f}:1 < 4.5:1)"

    ui_tokens = ["--border-strong", "--ring-color"]
    for bg in backgrounds:
        for ui in ui_tokens:
            cr = contrast_ratio(css_tokens[ui], css_tokens[bg])
            assert cr >= 3.0, f"UI contrast failure: {ui} on {bg} ({cr:.2f}:1 < 3.0:1)"

    print("  -> All 30 text pairs >= 4.5:1 and 6 UI component pairs >= 3.0:1 verified!")

    print("[2/3] Verifying updated hex values in DESIGN.md...")
    design_text = (REPO_ROOT / "docs/DESIGN.md").read_text(encoding="utf-8")
    assert "| `--clay` | `#A24E2D` |" in design_text, "Updated --clay hex (#A24E2D) must be in DESIGN.md"
    print("  -> Updated --clay hex #A24E2D reflected in DESIGN.md!")

    print("[3/3] Verifying results table saved in docs/design/contrast.md...")
    contrast_doc = REPO_ROOT / "docs/design/contrast.md"
    assert contrast_doc.is_file(), "docs/design/contrast.md must exist"
    doc_text = contrast_doc.read_text(encoding="utf-8")
    assert "WCAG 2.1 AA Colour Contrast Audit" in doc_text
    assert "#A24E2D" in doc_text
    print("  -> Contrast audit results table confirmed in docs/design/contrast.md!")

    print("\nAll T7-002 contrast verification criteria successfully passed!")
    return True


if __name__ == "__main__":
    if run_verification():
        sys.exit(0)
    sys.exit(1)
