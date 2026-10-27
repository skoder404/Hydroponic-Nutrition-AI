"""Unit tests for frontend design tokens and base components (T7-001)."""

from pathlib import Path


def test_globals_css_contains_design_md_tokens():
    css_file = Path("frontend/app/globals.css")
    assert css_file.exists()
    content = css_file.read_text(encoding="utf-8")

    # Tokens from DESIGN.md
    colors = [
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
    for c in colors:
        assert c in content, f"Missing color token {c}"

    assert "--radius-md" in content
    assert "--radius-lg" in content
    assert "--font-serif" in content
    assert "--font-sans" in content
    assert "prefers-reduced-motion" in content


def test_components_exist_and_exported():
    base_dir = Path("frontend/components/ui")
    assert base_dir.is_dir()

    expected = [
        "Button.tsx",
        "Input.tsx",
        "Card.tsx",
        "Badge.tsx",
        "Toast.tsx",
        "index.ts",
    ]
    for f in expected:
        assert (base_dir / f).is_file()

    index_text = (base_dir / "index.ts").read_text(encoding="utf-8")
    assert "Button" in index_text
    assert "Input" in index_text
    assert "Card" in index_text
    assert "Badge" in index_text
    assert "Toast" in index_text


def test_demo_page_exists_with_states():
    demo_file = Path("frontend/app/demo/page.tsx")
    assert demo_file.is_file()
    text = demo_file.read_text(encoding="utf-8")

    assert "reduced-motion" in text
    assert "loading" in text.lower()
    assert "disabled" in text.lower()
    assert "error" in text.lower()
    assert "missing" in text.lower()
    assert "skeleton" in text.lower()
