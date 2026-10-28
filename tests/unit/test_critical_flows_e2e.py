"""Unit tests verifying Playwright E2E configuration and critical flows coverage (T8-010).

Acceptance Criteria:
- Four flows pass headless
- Tests use the stub model and mock adapter
- Failures capture screenshots
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
PLAYWRIGHT_CONFIG = FRONTEND_DIR / "playwright.config.ts"
CRITICAL_FLOWS_SPEC = FRONTEND_DIR / "e2e" / "critical-flows.spec.ts"
FIXTURES_DIR = FRONTEND_DIR / "e2e" / "fixtures"


def test_playwright_config_exists_and_headless_screenshot():
    """Verify Playwright configuration enforces headless execution and screenshot capture."""
    assert PLAYWRIGHT_CONFIG.exists(), f"Missing config: {PLAYWRIGHT_CONFIG}"
    content = PLAYWRIGHT_CONFIG.read_text(encoding="utf-8")

    assert "headless: true" in content or "headless: true" in content.lower(), (
        "Playwright must run headless"
    )
    assert (
        'screenshot: "only-on-failure"' in content
        or "screenshot: 'only-on-failure'" in content
    ), "Playwright must configure screenshot capture on failure"
    assert 'testDir: "./e2e"' in content or "testDir: './e2e'" in content, (
        "testDir must point to e2e"
    )


def test_critical_flows_spec_covers_four_flows():
    """Verify critical-flows.spec.ts implements the 4 required critical flows."""
    assert CRITICAL_FLOWS_SPEC.exists(), f"Missing spec: {CRITICAL_FLOWS_SPEC}"
    spec = CRITICAL_FLOWS_SPEC.read_text(encoding="utf-8")

    # Flow 1: Crop to sensor entry to result to save to history
    assert "Crop to sensor entry to result to save to history" in spec
    assert "/analyze" in spec
    assert "/history" in spec
    assert "data-testid='save-record-button'" in spec

    # Flow 2: Campus live reading (mock adapter) to analyze
    assert "Campus live reading (mock adapter) to analyze" in spec
    assert 'adapter: "mock"' in spec or "adapter: 'mock'" in spec

    # Flow 3: Plant photo upload showing per-leaf cards
    assert "Plant photo upload showing per-leaf cards" in spec
    assert "data-testid='leaf-grid'" in spec or "leaf-grid" in spec
    assert "Leaf 1" in spec
    assert "Leaf 2" in spec
    assert "Leaf 3" in spec

    # Flow 4: Fallbacks & error handling (device unavailable, segmentation failure, uncertain tier)
    assert "Device unavailable offers manual entry fallback" in spec
    assert "device-unavailable-card" in spec
    assert "Switch to Manual Entry" in spec

    assert "segmentation failure displays retry prompt" in spec
    assert "Leaf Segmentation Unsuccessful" in spec
    assert "Retry Segmentation" in spec

    assert "Low-confidence result shows the uncertain tier" in spec
    assert "Uncertain Score" in spec


def test_e2e_fixtures_exist():
    """Verify small image fixtures exist for stub/fixture testing without real datasets."""
    assert FIXTURES_DIR.exists()
    assert (FIXTURES_DIR / "test_leaf.png").exists()
    assert (FIXTURES_DIR / "seg_fail_blur.png").exists()
    assert (FIXTURES_DIR / "uncertain_foliage.png").exists()
