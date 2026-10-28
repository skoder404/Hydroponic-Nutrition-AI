#!/usr/bin/env python3
"""Verification script for T8-010: E2E tests for critical flows.

Validates:
1. Four flows pass headless (crop->sensor->history, mock live reading, photo per-leaf cards, fallbacks).
2. Tests use the stub model and mock adapter.
3. Failures capture screenshots.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
sys.path.insert(0, str(REPO_ROOT))


def run_verification() -> bool:
    print("[1/3] Running Python unit verification for T8-010...")
    exit_code = pytest.main(
        [
            "-v",
            str(REPO_ROOT / "tests/unit/test_critical_flows_e2e.py"),
        ]
    )
    if exit_code != 0:
        print(f"FAILED: pytest exited with code {exit_code}")
        return False

    print("\n[2/3] Verifying Playwright config and test fixtures...")
    config_file = FRONTEND_DIR / "playwright.config.ts"
    if not config_file.exists():
        print(f"FAILED: Missing config file {config_file}")
        return False

    config_text = config_file.read_text(encoding="utf-8")
    if (
        'screenshot: "only-on-failure"' not in config_text
        and "screenshot: 'only-on-failure'" not in config_text
    ):
        print("FAILED: Playwright config does not set screenshot: 'only-on-failure'")
        return False

    print("\n[3/3] Running headless Playwright critical flow tests...")
    # On Windows, use npx.cmd
    npx_cmd = "npx.cmd" if sys.platform == "win32" else "npx"
    try:
        proc = subprocess.run(
            [npx_cmd, "playwright", "test", "e2e/critical-flows.spec.ts"],
            cwd=str(FRONTEND_DIR),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
            print(f"FAILED: Playwright tests exited with code {proc.returncode}")
            return False
    except (subprocess.SubprocessError, OSError) as e:
        print(f"FAILED: Error running Playwright tests: {e}")
        return False

    print("\nAll T8-010 critical E2E flows passed successfully!")
    return True


if __name__ == "__main__":
    if run_verification():
        sys.exit(0)
    sys.exit(1)
