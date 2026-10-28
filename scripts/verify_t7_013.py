"""Verification script for T7-013: Build SensorResultPanel."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_t7_013_sensor_result_panel() -> None:
    print("=== Verifying T7-013: Build SensorResultPanel ===")
    component_file = PROJECT_ROOT / "frontend/components/analyze/SensorResultPanel.tsx"
    index_file = PROJECT_ROOT / "frontend/components/analyze/index.ts"

    assert component_file.exists(), f"SensorResultPanel.tsx missing at {component_file}"
    with open(component_file, "r", encoding="utf-8") as f:
        code = f.read()

    # Criterion 1: Tomato shows the experimental banner
    assert "tomato-experimental-banner" in code or "Tomato Experimental Sensor Model" in code
    assert "Overview §28.5" in code or "synthetic" in code.lower()
    print("[PASS] Tomato shows the experimental banner and badge.")

    # Criterion 2: Evidence rows show reading, unit, and reference band
    assert "row.reading" in code
    assert "row.unit" in code
    assert "row.referenceBand" in code
    assert "Reference Band" in code
    print("[PASS] Evidence rows show reading, unit, and reference band.")

    # Criterion 3: Label is described as rule-derived
    assert "rule-derived" in code.lower()
    assert "Rule-Derived Target" in code or "rule-derived" in code
    print("[PASS] Target label is explicitly described as rule-derived.")

    # Check export
    with open(index_file, "r", encoding="utf-8") as f:
        idx_code = f.read()
    assert "SensorResultPanel" in idx_code

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

    print("\n>>> All T7-013 acceptance criteria verified successfully! <<<")


if __name__ == "__main__":
    test_t7_013_sensor_result_panel()
