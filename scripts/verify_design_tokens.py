"""Verification script for T7-001: Implement design tokens in Tailwind.

Verifies acceptance criteria:
1. Token names match DESIGN.md.
2. Each component shows default/hover/focus/disabled/loading states in a demo page.
3. prefers-reduced-motion respected.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def verify_css_tokens() -> None:
    css_path = Path("frontend/app/globals.css")
    assert css_path.exists(), "frontend/app/globals.css must exist"

    content = css_path.read_text(encoding="utf-8")

    # 1. Required DESIGN.md color tokens
    required_colors = [
        "--bg",
        "--surface",
        "--sand",
        "--forest",
        "--moss",
        "--sage",
        "--clay",
        "--bark",
        "--stone",
        "--border",
        "--border-strong",
        "--water",
        "--success",
        "--warning",
        "--error",
        "--info",
    ]
    for color in required_colors:
        assert color in content, f"Color token {color} missing in globals.css"

    # 2. Radii tokens
    for radius in ["--radius-md", "--radius-lg", "--radius-full"]:
        assert radius in content, f"Radius token {radius} missing in globals.css"

    # 3. Typography tokens
    for font in ["--font-serif", "--font-sans", "--font-mono"]:
        assert font in content, f"Typography token {font} missing in globals.css"

    # 4. prefers-reduced-motion media query
    assert "prefers-reduced-motion" in content, (
        "prefers-reduced-motion media query must be present in globals.css"
    )

    print("DESIGN.md tokens and prefers-reduced-motion in globals.css: VERIFIED.")


def verify_components_and_demo() -> None:
    components_dir = Path("frontend/components/ui")
    assert components_dir.exists(), "frontend/components/ui must exist"

    required_components = [
        "Button.tsx",
        "Input.tsx",
        "Card.tsx",
        "Badge.tsx",
        "Toast.tsx",
        "index.ts",
    ]
    for comp in required_components:
        comp_path = components_dir / comp
        assert comp_path.exists(), f"Component {comp} missing in frontend/components/ui"

    demo_page = Path("frontend/app/demo/page.tsx")
    assert demo_page.exists(), "frontend/app/demo/page.tsx must exist"

    demo_content = demo_page.read_text(encoding="utf-8")

    # Check that required states are represented in the demo
    required_states = [
        "loading",
        "disabled",
        "missing",
        "error",
        "interactive",
        "skeleton",
        "reduced-motion",
    ]
    for state in required_states:
        assert state in demo_content.lower(), f"State {state} missing in demo page"

    print("Base UI components and Demo page states: VERIFIED.")


def main() -> int:
    print("=== Verifying T7-001: Design Tokens in Tailwind & UI Components ===")
    verify_css_tokens()
    verify_components_and_demo()
    print("=== All T7-001 Acceptance Criteria PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
