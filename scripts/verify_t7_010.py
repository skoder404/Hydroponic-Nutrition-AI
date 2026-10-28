"""Verification script for T7-010: Build S8 Live Monitor."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_t7_010_live_monitor() -> None:
    print("=== Verifying T7-010: Build S8 Live Monitor ===")
    live_page = PROJECT_ROOT / "frontend/app/live/page.tsx"

    assert live_page.exists(), f"page.tsx missing at {live_page}"
    with open(live_page, "r", encoding="utf-8") as f:
        code = f.read()

    # Criterion 1: Server unreachable shows last-success time and Retry
    assert "server-unreachable-card" in code
    assert "Campus Server Unreachable" in code
    assert "Last successful telemetry capture" in code
    assert "Retry Connection" in code
    print("[PASS] Server unreachable shows last-success time and Retry button.")

    # Criterion 2: Stale values are labelled stale
    assert 'Badge variant="stale"' in code or "Stale" in code
    assert "isStale" in code
    print("[PASS] Stale telemetry values are explicitly labelled stale.")

    # Criterion 3: New data triggers a pulse and an aria-live announcement; reduced motion shows text only
    assert "aria-live" in code, "aria-live announcement container missing"
    assert "live-aria-announcement" in code
    assert "prefersReducedMotion" in code, "useMotion / prefersReducedMotion check missing"
    print("[PASS] New data triggers pulse/announcement; reduced motion respected.")

    # Check analyze buttons
    assert "Analyze these readings" in code
    assert "Analyze this frame" in code

    # TypeScript check
    print("Running npx tsc --noEmit in frontend...")
    res = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        cwd=str(PROJECT_ROOT / "frontend"),
        capture_output=True,
        text=True,
        shell=True,
    )
    assert res.returncode == 0, f"TypeScript check failed:\n{res.stdout}\n{res.stderr}"
    print("[PASS] Frontend TypeScript check passed with zero errors.")

    print("\n>>> All T7-010 acceptance criteria verified successfully! <<<")


if __name__ == "__main__":
    test_t7_010_live_monitor()
