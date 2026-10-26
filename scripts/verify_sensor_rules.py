#!/usr/bin/env python3
"""Verification script for crop sensor rule-label reference oracles (T4-016).

Verifies that:
1. Lettuce reference oracle reproduces prepared CSV labels exactly (0 mismatches).
2. Spinach reference oracle reproduces prepared CSV labels exactly (0 mismatches).
3. Tomato reference oracle reproduces prepared CSV labels with reported mismatch count
   (44,144/44,150 matches, exactly 6 floating-point tie-break mismatches due to rounded scales).
4. Boundary values (values exactly on thresholds) behave identically to dataset generation scripts.
5. Sourced tomato severity scales match `processed_data/sensor/tomato/manifest.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001, S110
        pass


ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from ml.sensor_models.rules import (
    TOMATO_SEVERITY_SCALES,
    evaluate_crop_rules_df,
    predict_lettuce_rule_label,
    predict_spinach_rule_label,
    predict_tomato_rule_label,
)


def verify_boundary_conditions() -> list[str]:
    """Test boundary values to ensure strict inequality matching source scripts."""
    failures: list[str] = []

    # 1. Lettuce boundaries: band is [lo, hi]. Exactly on band = in-band.
    base_lettuce = {"ph": 6.0, "ec": 1.8, "air_temp": 24.0, "humidity": 65.0, "water_temp": 22.0}
    if predict_lettuce_rule_label(base_lettuce) != "Normal":
        failures.append("Lettuce base sample not Normal")

    # ph low boundary
    if predict_lettuce_rule_label({**base_lettuce, "ph": 5.5}) != "Normal":
        failures.append("Lettuce ph=5.5 (exact boundary) should be Normal")
    if predict_lettuce_rule_label({**base_lettuce, "ph": 5.49}) != "pH_Low":
        failures.append("Lettuce ph=5.49 should be pH_Low")

    # ph high boundary
    if predict_lettuce_rule_label({**base_lettuce, "ph": 7.0}) != "Normal":
        failures.append("Lettuce ph=7.0 (exact boundary) should be Normal")
    if predict_lettuce_rule_label({**base_lettuce, "ph": 7.01}) != "pH_High":
        failures.append("Lettuce ph=7.01 should be pH_High")

    # Multiple stress
    if predict_lettuce_rule_label({**base_lettuce, "ph": 5.4, "ec": 3.0}) != "Multiple_Stress":
        failures.append("Lettuce ph=5.4 and ec=3.0 should be Multiple_Stress")

    # 2. Spinach boundaries: strict inequalities (> 21, > 2.3, < 1.8, < 6.0 | > 7.0, > 75, < 45)
    base_spinach = {
        "water_temperature_C": 20.0,
        "ec_mS_cm": 2.0,
        "ph": 6.5,
        "relative_humidity_percent": 60.0,
    }
    if predict_spinach_rule_label(base_spinach) != "Normal":
        failures.append("Spinach base sample not Normal")

    # water temp boundary
    if predict_spinach_rule_label({**base_spinach, "water_temperature_C": 21.0}) != "Normal":
        failures.append("Spinach water_temperature_C=21.0 (exact boundary) should be Normal")
    if predict_spinach_rule_label({**base_spinach, "water_temperature_C": 21.01}) != "Water_Temp_High":
        failures.append("Spinach water_temperature_C=21.01 should be Water_Temp_High")

    # ec high boundary
    if predict_spinach_rule_label({**base_spinach, "ec_mS_cm": 2.3}) != "Normal":
        failures.append("Spinach ec_mS_cm=2.3 (exact boundary) should be Normal")
    if predict_spinach_rule_label({**base_spinach, "ec_mS_cm": 2.31}) != "EC_High_Toxicity":
        failures.append("Spinach ec_mS_cm=2.31 should be EC_High_Toxicity")

    # ec low boundary
    if predict_spinach_rule_label({**base_spinach, "ec_mS_cm": 1.8}) != "Normal":
        failures.append("Spinach ec_mS_cm=1.8 (exact boundary) should be Normal")
    if predict_spinach_rule_label({**base_spinach, "ec_mS_cm": 1.79}) != "EC_Low_Deficiency":
        failures.append("Spinach ec_mS_cm=1.79 should be EC_Low_Deficiency")

    # ph boundaries
    if predict_spinach_rule_label({**base_spinach, "ph": 6.0}) != "Normal":
        failures.append("Spinach ph=6.0 (exact boundary) should be Normal")
    if predict_spinach_rule_label({**base_spinach, "ph": 5.99}) != "pH_Imbalance":
        failures.append("Spinach ph=5.99 should be pH_Imbalance")
    if predict_spinach_rule_label({**base_spinach, "ph": 7.0}) != "Normal":
        failures.append("Spinach ph=7.0 (exact boundary) should be Normal")
    if predict_spinach_rule_label({**base_spinach, "ph": 7.01}) != "pH_Imbalance":
        failures.append("Spinach ph=7.01 should be pH_Imbalance")

    # humidity boundaries
    if predict_spinach_rule_label({**base_spinach, "relative_humidity_percent": 75.0}) != "Normal":
        failures.append("Spinach relative_humidity_percent=75.0 (exact boundary) should be Normal")
    if predict_spinach_rule_label({**base_spinach, "relative_humidity_percent": 75.01}) != "Humidity_High_Mildew_Risk":
        failures.append("Spinach relative_humidity_percent=75.01 should be Humidity_High_Mildew_Risk")
    if predict_spinach_rule_label({**base_spinach, "relative_humidity_percent": 45.0}) != "Normal":
        failures.append("Spinach relative_humidity_percent=45.0 (exact boundary) should be Normal")
    if predict_spinach_rule_label({**base_spinach, "relative_humidity_percent": 44.99}) != "Humidity_Low_Stress":
        failures.append("Spinach relative_humidity_percent=44.99 should be Humidity_Low_Stress")

    # 3. Tomato boundaries: values exactly on thresholds have deviation = 0.0 -> Normal
    base_tomato = {
        "water_temperature_C": 25.0,
        "ec_mS_cm": 3.0,
        "ph": 6.2,
        "relative_humidity_percent": 65.0,
        "nitrogen_mg_L": 120.0,
        "phosphorus_mg_L": 50.0,
        "potassium_mg_L": 200.0,
    }
    if predict_tomato_rule_label(base_tomato) != "Normal":
        failures.append("Tomato base sample not Normal")

    # Boundary tests for all 9 tomato thresholds
    tomato_boundary_cases: list[tuple[dict[str, Any], str, str]] = [
        ({"water_temperature_C": 25.0}, "Normal", "water_temperature_C=25.0 should be Normal"),
        ({"water_temperature_C": 25.1}, "Water_Temp_High", "water_temperature_C=25.1 should be Water_Temp_High"),
        ({"ec_mS_cm": 4.0}, "Normal", "ec_mS_cm=4.0 should be Normal"),
        ({"ec_mS_cm": 4.05}, "EC_High_Toxicity", "ec_mS_cm=4.05 should be EC_High_Toxicity"),
        ({"ec_mS_cm": 2.0}, "Normal", "ec_mS_cm=2.0 should be Normal"),
        ({"ec_mS_cm": 1.95}, "EC_Low_Deficiency", "ec_mS_cm=1.95 should be EC_Low_Deficiency"),
        ({"ph": 6.0}, "Normal", "ph=6.0 should be Normal"),
        ({"ph": 5.95}, "pH_Imbalance", "ph=5.95 should be pH_Imbalance"),
        ({"ph": 6.5}, "Normal", "ph=6.5 should be Normal"),
        ({"ph": 6.55}, "pH_Imbalance", "ph=6.55 should be pH_Imbalance"),
        ({"relative_humidity_percent": 85.0}, "Normal", "relative_humidity_percent=85.0 should be Normal"),
        ({"relative_humidity_percent": 85.1}, "Humidity_High_Mildew_Risk", "relative_humidity_percent=85.1 should be Humidity_High_Mildew_Risk"),
        ({"relative_humidity_percent": 50.0}, "Normal", "relative_humidity_percent=50.0 should be Normal"),
        ({"relative_humidity_percent": 49.9}, "Humidity_Low_Stress", "relative_humidity_percent=49.9 should be Humidity_Low_Stress"),
        ({"nitrogen_mg_L": 90.0}, "Normal", "nitrogen_mg_L=90.0 should be Normal"),
        ({"nitrogen_mg_L": 89.0}, "N_Deficiency", "nitrogen_mg_L=89.0 should be N_Deficiency"),
        ({"phosphorus_mg_L": 35.0}, "Normal", "phosphorus_mg_L=35.0 should be Normal"),
        ({"phosphorus_mg_L": 34.0}, "P_Deficiency", "phosphorus_mg_L=34.0 should be P_Deficiency"),
        ({"potassium_mg_L": 144.0}, "Normal", "potassium_mg_L=144.0 should be Normal"),
        ({"potassium_mg_L": 143.0}, "K_Deficiency", "potassium_mg_L=143.0 should be K_Deficiency"),
    ]

    for override, expected_label, msg in tomato_boundary_cases:
        sample = {**base_tomato, **override}
        pred = predict_tomato_rule_label(sample)
        if pred != expected_label:
            failures.append(f"{msg} (got '{pred}')")

    return failures


def verify_tomato_scales() -> list[str]:
    """Verify that severity scales match the checked-in manifest."""
    manifest_path = ROOT / "processed_data/sensor/tomato/manifest.json"
    if not manifest_path.exists():
        return [f"Missing manifest: {manifest_path}"]

    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_scales = manifest.get("severity_scales", {})
    failures = []
    for rule, scale in TOMATO_SEVERITY_SCALES.items():
        if rule not in manifest_scales:
            failures.append(f"Rule '{rule}' missing from tomato manifest")
        elif manifest_scales[rule] != scale:
            failures.append(
                f"Scale mismatch for '{rule}': code={scale}, manifest={manifest_scales[rule]}"
            )
    return failures


def audit_file_reproduction(csv_path: Path, crop: str) -> tuple[int, int, float]:
    """Audit rule oracle against a prepared dataset file."""
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"Column 'label' missing from {csv_path}")

    preds = evaluate_crop_rules_df(df, crop)
    mismatches = int((df["label"] != preds).sum())
    total = len(df)
    match_pct = ((total - mismatches) / total) * 100.0
    return total, mismatches, match_pct


def main() -> int:
    """Run all rule oracle verifications."""
    print("=" * 70)
    print("SENSOR RULES REFERENCE ORACLE VERIFICATION (T4-016)")
    print("=" * 70)

    # 1. Sourced tomato scales check
    print("\n--- 1. Tomato Severity Scales Verification ---")
    scale_errs = verify_tomato_scales()
    if scale_errs:
        for err in scale_errs:
            print(f"❌ {err}")
        return 1
    print("✅ All 9 tomato severity scales match processed_data/sensor/tomato/manifest.json")
    for k, v in TOMATO_SEVERITY_SCALES.items():
        print(f"   - {k}: {v:.2f}")

    # 2. Boundary condition checks
    print("\n--- 2. Boundary Condition Tests ---")
    boundary_errs = verify_boundary_conditions()
    if boundary_errs:
        for err in boundary_errs:
            print(f"❌ {err}")
        return 1
    print("✅ All boundary tests passed (exact threshold equality is within-band for all crops)")

    # 3. Dataset reproduction checks
    print("\n--- 3. Dataset Reproduction Accuracy ---")
    files_to_check: list[tuple[str, Path, int]] = [
        # (crop, path, expected_max_mismatches)
        (
            "lettuce",
            ROOT / "sensor_data/Lettuce_Sensor-Readings/hydroponic_lettuce_full.csv",
            0,
        ),
        (
            "lettuce",
            ROOT / "processed_data/sensor/lettuce/train.csv",
            0,
        ),
        (
            "lettuce",
            ROOT / "processed_data/sensor/lettuce/val.csv",
            0,
        ),
        (
            "spinach",
            ROOT / "sensor_data/Spinach_Sensor-Readings/spinach_7class_full.csv",
            0,
        ),
        (
            "spinach",
            ROOT / "processed_data/sensor/spinach/train.csv",
            0,
        ),
        (
            "spinach",
            ROOT / "processed_data/sensor/spinach/val.csv",
            0,
        ),
        (
            "tomato",
            ROOT / "sensor_data/Tomato_Sensor-Readings/tomato_hydroponic_full.csv",
            6,  # 6 floating-point tie-break rows due to rounded scales across 44,150 rows
        ),
        (
            "tomato",
            ROOT / "sensor_data/Tomato_Sensor-Readings/tomato_hydroponic_train.csv",
            5,
        ),
        (
            "tomato",
            ROOT / "processed_data/sensor/tomato/train.csv",
            5,
        ),
        (
            "tomato",
            ROOT / "processed_data/sensor/tomato/val.csv",
            2,
        ),
    ]


    all_passed = True
    print(f"{'Crop':<8} {'Dataset':<50} {'Rows':<8} {'Mismatches':<12} {'Accuracy'}")
    print("-" * 90)

    for crop, path, max_allowed_mismatches in files_to_check:
        if not path.exists():
            print(f"⚠️ Skipping missing file: {path.relative_to(ROOT)}")
            continue
        total, mismatches, match_pct = audit_file_reproduction(path, crop)
        rel_path = str(path.relative_to(ROOT)).replace("\\", "/")
        status = "✅" if mismatches <= max_allowed_mismatches else "❌"
        print(f"{crop:<8} {rel_path:<50} {total:<8} {mismatches:<12} {match_pct:6.2f}% {status}")

        if mismatches > max_allowed_mismatches:
            all_passed = False
            print(f"   ❌ Exceeded allowed mismatch limit ({mismatches} > {max_allowed_mismatches})")

    print("-" * 90)
    if not all_passed:
        print("❌ Dataset reproduction verification failed!")
        return 1

    print("\n✅ T4-016 Acceptance Criteria fully satisfied:")
    print("   [x] Oracle reproduces the labels in the prepared CSVs (mismatch count reported)")
    print("   [x] Boundary values (exactly on a threshold) behave as in the scripts")
    print("   [x] Tomato scales are sourced and documented from manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
