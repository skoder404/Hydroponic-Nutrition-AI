#!/usr/bin/env python3
"""Verification script for Sensor Feature Schemas and Leakage Blacklists (T2-018).

Acceptance Criteria:
1. Feature order matches the audited order.
2. Blacklists are enforced by a function that raises on any blacklisted column in a feature matrix.
3. Hard bounds are separate from the normal-band hints.

Re-run command:
    python scripts/verify_sensor_schemas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.sensor_models.schemas import (
    CROP_SENSOR_SCHEMAS,
    DataLeakageError,
    SchemaValidationError,
    assert_no_leakage,
    get_sensor_schema,
    validate_feature_matrix,
)

AUDITED_FEATURE_ORDERS = {
    "lettuce": ("ph", "ec", "water_temp", "humidity", "air_temp"),
    "spinach": (
        "air_temperature_C",
        "relative_humidity_percent",
        "water_temperature_C",
        "ph",
        "ec_mS_cm",
    ),
    "tomato": (
        "air_temperature_C",
        "relative_humidity_percent",
        "water_temperature_C",
        "ph",
        "ec_mS_cm",
        "nitrogen_mg_L",
        "phosphorus_mg_L",
        "potassium_mg_L",
    ),
}


def verify_feature_orders() -> tuple[bool, str]:
    """Criterion 1: Feature order matches the audited order."""
    for crop, expected_order in AUDITED_FEATURE_ORDERS.items():
        schema = get_sensor_schema(crop)
        actual_order = schema.feature_names
        if actual_order != expected_order:
            return (
                False,
                f"Feature order mismatch for '{crop}': expected {expected_order}, got {actual_order}",
            )

    return (
        True,
        (
            f"All 3 crops match audited feature orders: "
            f"lettuce ({len(AUDITED_FEATURE_ORDERS['lettuce'])}), "
            f"spinach ({len(AUDITED_FEATURE_ORDERS['spinach'])}), "
            f"tomato ({len(AUDITED_FEATURE_ORDERS['tomato'])})"
        ),
    )


def verify_blacklist_enforcement() -> tuple[bool, str]:
    """Criterion 2: Blacklists are enforced by a function that raises on any blacklisted column."""
    test_cases = [
        # (crop, violating_column)
        ("lettuce", "label_next"),
        ("lettuce", "ph_prev"),
        ("lettuce", "ec_delta"),
        ("lettuce", "water_temp_roll3"),
        ("lettuce", "humidity_day_mean"),
        ("lettuce", "tds"),
        ("lettuce", "co2"),
        ("lettuce", "hours_logged"),
        ("lettuce", "experiment"),
        ("spinach", "tds_ppm"),
        ("spinach", "label_id"),
        ("spinach", "humidity_source"),
        ("spinach", "sample_id"),
        ("spinach", "source_timestamp_EDT"),
        ("spinach", "bench"),
        ("tomato", "n_rules_broken"),
        ("tomato", "source_row"),
        ("tomato", "source_timestamp"),
        ("tomato", "label_id"),
        ("tomato", "sample_id"),
    ]

    for crop, bad_col in test_cases:
        # Test 1: assert_no_leakage on list of columns
        try:
            assert_no_leakage(["valid_col", bad_col], crop=crop)
            return (
                False,
                f"assert_no_leakage failed to raise DataLeakageError on column '{bad_col}' for '{crop}'",
            )
        except DataLeakageError:
            pass  # Expected

        # Test 2: validate_feature_matrix on DataFrame
        dummy_df = pd.DataFrame({col: [1.0] for col in get_sensor_schema(crop).feature_names})
        dummy_df[bad_col] = 1.0
        try:
            validate_feature_matrix(dummy_df, crop=crop)
            return (
                False,
                f"validate_feature_matrix failed to raise DataLeakageError on column '{bad_col}' for '{crop}'",
            )
        except DataLeakageError:
            pass  # Expected

    # Verify clean feature matrix passes without error
    for crop in ("lettuce", "spinach", "tomato"):
        schema = get_sensor_schema(crop)
        clean_df = pd.DataFrame(
            {
                feat: [(schema.normal_bands[feat][0] + schema.normal_bands[feat][1]) / 2.0]
                for feat in schema.feature_names
            }
        )
        try:
            assert_no_leakage(clean_df, crop=crop)
            validated = validate_feature_matrix(clean_df, crop=crop, check_bounds=True)
            if list(validated.columns) != list(schema.feature_names):
                return False, f"Validated matrix columns mismatch for '{crop}'"
        except (DataLeakageError, SchemaValidationError, ValueError) as e:
            return False, f"Unexpected error on clean matrix for '{crop}': {e}"

    return True, f"Blacklist enforcement validated across {len(test_cases)} leakage vectors"


def verify_hard_bounds_separate_from_normal_bands() -> tuple[bool, str]:
    """Criterion 3: Hard bounds are separate from normal-band hints."""
    for crop, schema in CROP_SENSOR_SCHEMAS.items():
        hard = schema.hard_bounds
        normal = schema.normal_bands
        units = schema.units

        if set(hard.keys()) != set(normal.keys()):
            return False, f"Key mismatch between hard_bounds and normal_bands for '{crop}'"

        for feat in schema.feature_names:
            h_lo, h_hi = hard[feat]
            n_lo, n_hi = normal[feat]
            unit = units[feat]

            # Hard bounds must strictly contain or equal normal bounds (hard min <= normal min < normal max <= hard max)
            if h_lo > n_lo or h_hi < n_hi:
                return (
                    False,
                    f"Physical bounds conflict for '{crop}' '{feat}': hard=[{h_lo}, {h_hi}] vs normal=[{n_lo}, {n_hi}]",
                )
            if not unit:
                return False, f"Empty unit string for '{crop}' '{feat}'"

    return (
        True,
        "Hard physical bounds and normal agronomic bands are strictly separated for all features in all crops",
    )


def verify_real_data_matrices() -> tuple[bool, str]:
    """Test schemas against real prepared datasets."""
    from ml.sensor_models.schemas import extract_feature_matrix

    # Lettuce v2
    lettuce_train = pd.read_csv(ROOT / "processed_data" / "sensor" / "lettuce" / "train.csv")
    lettuce_features = extract_feature_matrix(lettuce_train, crop="lettuce", check_bounds=True)
    if len(lettuce_features) != len(lettuce_train):
        return False, "Failed to validate lettuce train features"

    # Spinach v2
    spinach_train = pd.read_csv(ROOT / "processed_data" / "sensor" / "spinach" / "train.csv")
    spinach_features = extract_feature_matrix(spinach_train, crop="spinach", check_bounds=True)
    if len(spinach_features) != len(spinach_train):
        return False, "Failed to validate spinach train features"

    # Tomato v1 train
    tomato_train = pd.read_csv(
        ROOT / "sensor_data" / "Tomato_Sensor-Readings" / "tomato_hydroponic_train.csv"
    )
    tomato_features = extract_feature_matrix(tomato_train, crop="tomato", check_bounds=True)
    if len(tomato_features) != len(tomato_train):
        return False, "Failed to validate tomato train features"

    return (
        True,
        (
            f"Real datasets passed schema extraction and bounds validation: "
            f"lettuce={len(lettuce_features)} rows, spinach={len(spinach_features)} rows, tomato={len(tomato_features)} rows"
        ),
    )


def run_all_checks() -> int:
    """Execute all verification checks."""
    print("=" * 70)
    print("Sensor Feature Schemas and Leakage Blacklist Verification (T2-018)")
    print("=" * 70)

    checks = [
        ("Criterion 1: Audited feature order match", verify_feature_orders),
        ("Criterion 2: Blacklist enforcement & leakage error", verify_blacklist_enforcement),
        ("Criterion 3: Hard bounds separated from normal bands", verify_hard_bounds_separate_from_normal_bands),
        ("Criterion 4: Validation on real prepared datasets", verify_real_data_matrices),
    ]

    all_passed = True
    for title, check_fn in checks:
        passed, msg = check_fn()
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] {title}")
        print(f"         {msg}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL VERIFICATION CHECKS PASSED (T2-018)")
        return 0
    else:
        print("SOME VERIFICATION CHECKS FAILED (T2-018)")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_checks())
