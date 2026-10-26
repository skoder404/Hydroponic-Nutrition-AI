"""Unit tests for sensor feature schemas and leakage blacklists (T2-018)."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.sensor_models.schemas import (
    CROP_SENSOR_SCHEMAS,
    DataLeakageError,
    SchemaValidationError,
    assert_no_leakage,
    extract_feature_matrix,
    get_leakage_blacklist,
    get_sensor_schema,
    is_column_blacklisted,
    validate_feature_matrix,
)


def test_registered_crops() -> None:
    """Verify all 3 crops are registered in CROP_SENSOR_SCHEMAS."""
    assert set(CROP_SENSOR_SCHEMAS.keys()) == {"lettuce", "spinach", "tomato"}
    for crop in ("lettuce", "spinach", "tomato"):
        schema = get_sensor_schema(crop)
        assert schema.crop == crop
        assert len(schema.features) > 0


def test_unknown_crop_raises() -> None:
    """Verify querying an unknown crop raises ValueError."""
    with pytest.raises(ValueError, match="Unknown crop 'wheat'"):
        get_sensor_schema("wheat")


def test_feature_order_matches_audit() -> None:
    """Verify feature names and ordering match audited dataset schemas."""
    assert get_sensor_schema("lettuce").feature_names == (
        "ph",
        "ec",
        "water_temp",
        "humidity",
        "air_temp",
    )
    assert get_sensor_schema("spinach").feature_names == (
        "air_temperature_C",
        "relative_humidity_percent",
        "water_temperature_C",
        "ph",
        "ec_mS_cm",
    )
    assert get_sensor_schema("tomato").feature_names == (
        "air_temperature_C",
        "relative_humidity_percent",
        "water_temperature_C",
        "ph",
        "ec_mS_cm",
        "nitrogen_mg_L",
        "phosphorus_mg_L",
        "potassium_mg_L",
    )


def test_hard_bounds_separate_from_normal_bands() -> None:
    """Verify hard physical bounds strictly contain or match normal bands."""
    for crop in ("lettuce", "spinach", "tomato"):
        schema = get_sensor_schema(crop)
        for feat in schema.features:
            h_min, h_max = feat.hard_bounds
            n_min, n_max = feat.normal_band
            assert h_min <= n_min, f"{crop}.{feat.name}: hard min {h_min} > normal min {n_min}"
            assert h_max >= n_max, f"{crop}.{feat.name}: hard max {h_max} < normal max {n_max}"
            assert feat.unit != "", f"{crop}.{feat.name}: empty unit"


def test_lettuce_leakage_blacklist_detection() -> None:
    """Verify lettuce blacklists target trends, daily aggregates, and metadata."""
    blacklist = get_leakage_blacklist("lettuce")
    assert "label_next" in blacklist
    assert "experiment" in blacklist
    assert "replicate" in blacklist
    assert "ph_prev" in blacklist
    assert "ec_roll3" in blacklist
    assert "air_temp_day_mean" in blacklist

    # Check pattern matching
    assert is_column_blacklisted("ph_prev", "lettuce")
    assert is_column_blacklisted("ec_delta", "lettuce")
    assert is_column_blacklisted("water_temp_roll3", "lettuce")
    assert is_column_blacklisted("custom_day_mean", "lettuce")


def test_spinach_leakage_blacklist_detection() -> None:
    """Verify spinach blacklists tds_ppm, metadata, and timestamps."""
    blacklist = get_leakage_blacklist("spinach")
    assert "tds_ppm" in blacklist
    assert "label_id" in blacklist
    assert "sample_id" in blacklist
    assert "source_timestamp_EDT" in blacklist
    assert "bench" in blacklist

    assert is_column_blacklisted("tds_ppm", "spinach")
    assert is_column_blacklisted("bench", "spinach")


def test_tomato_leakage_blacklist_detection() -> None:
    """Verify tomato blacklists n_rules_broken, source rows, and timestamps."""
    blacklist = get_leakage_blacklist("tomato")
    assert "n_rules_broken" in blacklist
    assert "source_row" in blacklist
    assert "source_timestamp" in blacklist
    assert "sample_id" in blacklist
    assert "label_id" in blacklist

    assert is_column_blacklisted("n_rules_broken", "tomato")
    assert is_column_blacklisted("source_row", "tomato")


def test_assert_no_leakage_raises_on_blacklisted_columns() -> None:
    """Verify assert_no_leakage raises DataLeakageError."""
    # List input
    with pytest.raises(DataLeakageError, match="Data leakage detected.*label_next"):
        assert_no_leakage(["ph", "ec", "label_next"], crop="lettuce")

    # DataFrame input
    df_spinach_leaked = pd.DataFrame(
        {
            "air_temperature_C": [22.0],
            "relative_humidity_percent": [65.0],
            "water_temperature_C": [19.0],
            "ph": [6.5],
            "ec_mS_cm": [2.0],
            "tds_ppm": [1400.0],
        }
    )
    with pytest.raises(DataLeakageError, match="Data leakage detected.*tds_ppm"):
        assert_no_leakage(df_spinach_leaked, crop="spinach")


def test_validate_feature_matrix_clean() -> None:
    """Verify validate_feature_matrix accepts a clean matrix matching schema."""
    df_clean = pd.DataFrame(
        {
            "ph": [6.0, 6.2],
            "ec": [1.8, 1.9],
            "water_temp": [20.0, 21.0],
            "humidity": [60.0, 65.0],
            "air_temp": [22.0, 23.0],
        }
    )
    validated = validate_feature_matrix(df_clean, crop="lettuce", strict_order=True, check_bounds=True)
    assert list(validated.columns) == ["ph", "ec", "water_temp", "humidity", "air_temp"]
    assert len(validated) == 2


def test_validate_feature_matrix_rejects_leakage() -> None:
    """Verify validate_feature_matrix rejects blacklisted columns."""
    df_leaked = pd.DataFrame(
        {
            "ph": [6.0],
            "ec": [1.8],
            "water_temp": [20.0],
            "humidity": [60.0],
            "air_temp": [22.0],
            "experiment": ["EXP_1"],
        }
    )
    with pytest.raises(DataLeakageError, match="Data leakage detected"):
        validate_feature_matrix(df_leaked, crop="lettuce")


def test_validate_feature_matrix_rejects_missing_features() -> None:
    """Verify validate_feature_matrix raises SchemaValidationError when features are missing."""
    df_incomplete = pd.DataFrame(
        {
            "ph": [6.0],
            "ec": [1.8],
            # missing water_temp, humidity, air_temp
        }
    )
    with pytest.raises(SchemaValidationError, match="Missing required sensor features"):
        validate_feature_matrix(df_incomplete, crop="lettuce")


def test_validate_feature_matrix_strict_order() -> None:
    """Verify validate_feature_matrix raises SchemaValidationError when columns are out of order."""
    df_unordered = pd.DataFrame(
        {
            "air_temp": [22.0],
            "humidity": [60.0],
            "water_temp": [20.0],
            "ec": [1.8],
            "ph": [6.0],
        }
    )
    with pytest.raises(SchemaValidationError, match="Feature columns do not match audited order"):
        validate_feature_matrix(df_unordered, crop="lettuce", strict_order=True)

    # When strict_order is False, it should reorder and succeed
    reordered = validate_feature_matrix(df_unordered, crop="lettuce", strict_order=False)
    assert list(reordered.columns) == ["ph", "ec", "water_temp", "humidity", "air_temp"]


def test_validate_feature_matrix_hard_bounds_violation() -> None:
    """Verify validate_feature_matrix raises on physically impossible values."""
    df_impossible = pd.DataFrame(
        {
            "ph": [14.5],  # Hard bound is [3.0, 10.0]
            "ec": [1.8],
            "water_temp": [20.0],
            "humidity": [60.0],
            "air_temp": [22.0],
        }
    )
    with pytest.raises(SchemaValidationError, match="Hard bound violations detected"):
        validate_feature_matrix(df_impossible, crop="lettuce", check_bounds=True)


def test_extract_feature_matrix() -> None:
    """Verify extract_feature_matrix extracts clean features from raw tabular data."""
    raw_df = pd.DataFrame(
        {
            "sample_id": ["S001", "S002"],
            "air_temperature_C": [21.5, 23.0],
            "relative_humidity_percent": [60.0, 68.0],
            "water_temperature_C": [19.5, 20.2],
            "ph": [6.4, 6.6],
            "ec_mS_cm": [2.1, 2.05],
            "tds_ppm": [1470, 1435],
            "label": ["Normal", "Normal"],
        }
    )
    features = extract_feature_matrix(raw_df, crop="spinach", check_bounds=True)
    assert list(features.columns) == [
        "air_temperature_C",
        "relative_humidity_percent",
        "water_temperature_C",
        "ph",
        "ec_mS_cm",
    ]
    assert "tds_ppm" not in features.columns
    assert "sample_id" not in features.columns
    assert "label" not in features.columns
