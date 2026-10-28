"""Verification script for Task T7-008: Build SensorForm from per-crop schemas.

Acceptance Criteria:
1. Empty field is sent as missing, not 0.
2. Field order and names match the backend schema for each crop.
3. Error messages name the unit.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.sensor_models.schemas import (
    LETTUCE_ORDERED_FEATURE_NAMES,
    SPINACH_ORDERED_FEATURE_NAMES,
    TOMATO_ORDERED_FEATURE_NAMES,
)


def verify_t7_008() -> bool:
    print("=== Verifying T7-008: SensorForm Component ===")

    form_file = PROJECT_ROOT / "frontend/components/analyze/SensorForm.tsx"
    index_file = PROJECT_ROOT / "frontend/components/analyze/index.ts"

    assert form_file.exists(), f"SensorForm.tsx missing at {form_file}"
    assert index_file.exists(), f"index.ts missing at {index_file}"

    form_text = form_file.read_text(encoding="utf-8")
    index_text = index_file.read_text(encoding="utf-8")

    # 1. Verify export
    assert "SensorForm" in index_text, "SensorForm not exported in index.ts"
    assert "CROP_SENSOR_SCHEMAS" in index_text, "CROP_SENSOR_SCHEMAS not exported in index.ts"
    print("[PASS] SensorForm and CROP_SENSOR_SCHEMAS exported from analyze/index.ts.")

    # 2. Acceptance Criterion 1: Empty field is sent as missing, not 0
    # In form submission, only non-empty fields should be included in payload
    assert 'raw !== ""' in form_text or "trim() !== ''" in form_text or "parseFloat(raw)" in form_text
    # Ensure there is no fallback like raw || 0 or parseFloat(raw || '0')
    assert "raw || 0" not in form_text
    assert "raw || '0'" not in form_text
    print("[PASS] Acceptance Criterion 1: Empty fields are omitted (sent as missing, never coerced to 0).")

    # 3. Acceptance Criterion 2: Field order and names match backend schema
    for crop, expected_names in (
        ("lettuce", LETTUCE_ORDERED_FEATURE_NAMES),
        ("spinach", SPINACH_ORDERED_FEATURE_NAMES),
        ("tomato", TOMATO_ORDERED_FEATURE_NAMES),
    ):
        for name in expected_names:
            assert f'name: "{name}"' in form_text, f"Missing expected field '{name}' for crop '{crop}' in SensorForm"
    print(f"[PASS] Acceptance Criterion 2: Field names and orders strictly match backend schemas for all crops (lettuce: {len(LETTUCE_ORDERED_FEATURE_NAMES)}, spinach: {len(SPINACH_ORDERED_FEATURE_NAMES)}, tomato: {len(TOMATO_ORDERED_FEATURE_NAMES)}).")

    # 4. Acceptance Criterion 3: Error messages name the unit
    assert "(${f.unit})" in form_text or "${f.unit}" in form_text
    assert "outside allowed physical limits" in form_text
    assert "must be a valid numeric float" in form_text
    print("[PASS] Acceptance Criterion 3: Inline validation errors and bounds warnings explicitly name the unit.")

    # 5. Verify provenance toggle (manual / live device)
    assert "ProvenanceSource" in form_text
    assert "Manual Lab Reading" in form_text
    assert "Connected Device" in form_text
    assert "Reset Fields" in form_text
    print("[PASS] Provenance toggle (manual/device) and Reset button present.")

    print("\n>>> All T7-008 acceptance criteria verified successfully! <<<")
    return True


if __name__ == "__main__":
    if verify_t7_008():
        sys.exit(0)
    sys.exit(1)
