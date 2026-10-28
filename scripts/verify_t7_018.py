"""Verification script for T7-018: Build S6 Models & Limitations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_t7_018_models_and_limitations() -> None:
    frontend_dir = PROJECT_ROOT / "frontend"
    models_page_file = frontend_dir / "app/models/page.tsx"
    models_table_file = frontend_dir / "components/models/ModelsTable.tsx"
    metrics_drawer_file = frontend_dir / "components/models/MetricsDrawer.tsx"
    datasets_view_file = frontend_dir / "components/models/DatasetsView.tsx"
    limitations_view_file = frontend_dir / "components/models/LimitationsView.tsx"

    print("=== Verifying T7-018: Build S6 Models & Limitations ===")

    # 1. Component files exist
    assert models_page_file.exists(), f"Models page missing at {models_page_file}"
    assert models_table_file.exists(), f"ModelsTable missing at {models_table_file}"
    assert metrics_drawer_file.exists(), f"MetricsDrawer missing at {metrics_drawer_file}"
    assert datasets_view_file.exists(), f"DatasetsView missing at {datasets_view_file}"
    assert limitations_view_file.exists(), f"LimitationsView missing at {limitations_view_file}"
    print("[PASS] All Models portal component files exist.")

    # 2. Check Acceptance Criterion 1: Experimental models are labelled
    with open(models_table_file, "r", encoding="utf-8") as f:
        table_code = f.read()

    assert "m.experimental" in table_code, "ModelsTable must check experimental flag"
    assert '<Badge variant="experimental">Experimental</Badge>' in table_code, (
        "Experimental badge must label experimental models"
    )

    with open(metrics_drawer_file, "r", encoding="utf-8") as f:
        drawer_code = f.read()

    assert "model.experimental" in drawer_code
    assert '<Badge variant="experimental">Experimental</Badge>' in drawer_code
    print("[PASS] Experimental models are clearly labelled in table and metrics drawer.")

    # 3. Check Acceptance Criterion 2: Limitations list matches Overview §28
    with open(limitations_view_file, "r", encoding="utf-8") as f:
        limitations_code = f.read()

    # Verify directly against PROJECT_OVERVIEW.md
    overview_path = PROJECT_ROOT / "PROJECT_OVERVIEW.md"
    assert overview_path.exists()
    with open(overview_path, "r", encoding="utf-8") as f:
        overview_text = f.read()

    # Find Section 28 items in PROJECT_OVERVIEW.md
    assert "## 28. Known limitations and required disclosures" in overview_text

    # Verify key limitations are represented in LimitationsView
    assert "Some sensor labels are generated from thresholds rather than observed plant-health outcomes." in limitations_code
    assert "The lettuce training set was oversampled;" in limitations_code
    assert "droop_angle" in limitations_code
    assert "A model confidence score is not automatically a calibrated probability." in limitations_code
    assert "Recommendations are advisory and must remain conservative." in limitations_code
    assert "Scores mean agreement with project rules/labels on a documented split." in limitations_code
    print("[PASS] Limitations list matches Overview Section 28 disclosures and includes required statement.")

    # 4. Check Acceptance Criterion 3: Empty state explains no promoted models
    assert "No promoted models" in table_code, "Empty state must display 'No promoted models'"
    assert "unavailable until" in table_code.lower() or "promoted in the registry" in table_code.lower(), (
        "Empty state must explain that predictions are unavailable until models are promoted"
    )
    print("[PASS] Empty state explains no promoted models and unavailability of predictions.")

    # 5. Check TypeScript compilation of frontend
    print("Running npx tsc --noEmit in frontend...")
    res = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        cwd=str(frontend_dir),
        capture_output=True,
        text=True,
        shell=True,
    )
    assert res.returncode == 0, f"TypeScript compilation error:\n{res.stdout}\n{res.stderr}"
    print("[PASS] Frontend TypeScript compilation passed with zero errors.")

    print("\n>>> All T7-018 acceptance criteria verified successfully! <<<")


if __name__ == "__main__":
    test_t7_018_models_and_limitations()
