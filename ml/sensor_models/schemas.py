"""Single source of truth for sensor feature schemas, physical bounds, and leakage blacklists (T2-018).

Defines:
- Ordered sensor feature lists matching audited dataset order per crop.
- Feature units, agronomic normal bands, and physical hard bounds (physically impossible limits).
- Explicit data leakage blacklists prohibiting derived metrics, future targets, timestamps, and metadata.
- Enforcement functions that raise DataLeakageError when blacklisted columns are present in feature matrices.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


class DataLeakageError(ValueError):
    """Raised when one or more data leakage or blacklisted columns are detected in feature inputs."""



class SchemaValidationError(ValueError):
    """Raised when feature inputs fail schema order, missing features, or physical bound validation."""



@dataclass(frozen=True)
class SensorFeature:
    """Specification of an individual physical sensor feature.

    Attributes:
        name: Column name in audited dataset.
        unit: Physical engineering unit of measurement.
        hard_bounds: (min, max) physically plausible limit. Values outside are physically
            impossible or indicate sensor hardware failure/disconnection.
        normal_band: (min, max) agronomically healthy/optimal target operating range for the crop.
        description: Brief description of sensor measurement.
    """

    name: str
    unit: str
    hard_bounds: tuple[float, float]
    normal_band: tuple[float, float]
    description: str = ""


# ==============================================================================
# Lettuce Sensor Features & Blacklist
# ==============================================================================

LETTUCE_FEATURES: tuple[SensorFeature, ...] = (
    SensorFeature(
        name="ph",
        unit="pH",
        hard_bounds=(3.0, 10.0),
        normal_band=(5.5, 6.5),
        description="Nutrient solution pH acidity/alkalinity level",
    ),
    SensorFeature(
        name="ec",
        unit="mS/cm",
        hard_bounds=(0.0, 6.0),
        normal_band=(1.2, 2.2),
        description="Electrical conductivity of the nutrient solution",
    ),
    SensorFeature(
        name="water_temp",
        unit="degC",
        hard_bounds=(0.0, 45.0),
        normal_band=(18.0, 24.0),
        description="Hydroponic reservoir water temperature",
    ),
    SensorFeature(
        name="humidity",
        unit="%",
        hard_bounds=(0.0, 100.0),
        normal_band=(50.0, 70.0),
        description="Greenhouse ambient relative humidity",
    ),
    SensorFeature(
        name="air_temp",
        unit="degC",
        hard_bounds=(0.0, 55.0),
        normal_band=(18.0, 26.0),
        description="Greenhouse ambient air temperature",
    ),
)

LETTUCE_ORDERED_FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in LETTUCE_FEATURES)

# Explicit leakage blacklist for lettuce
LETTUCE_BLACKLIST_EXACT: frozenset[str] = frozenset(
    {
        "label",
        "label_next",
        "tds",
        "co2",
        "lux_meter",
        "hours_logged",
        "experiment",
        "replicate",
        "treatment",
        "date",
        "day_index",
        "slot",
        "timestamp",
        # Explicitly named trend / rolling / aggregate columns in lettuce full dataset
        "ph_prev",
        "ph_delta",
        "ph_roll3",
        "ec_prev",
        "ec_delta",
        "ec_roll3",
        "water_temp_prev",
        "water_temp_delta",
        "water_temp_roll3",
        "air_temp_day_mean",
        "air_temp_day_min",
        "air_temp_day_max",
        "humidity_day_mean",
        "co2_day_mean",
    }
)

LETTUCE_BLACKLIST_PATTERNS: tuple[str, ...] = (
    "*_prev",
    "*_delta",
    "*_roll3",
    "*_day_mean",
    "*_day_min",
    "*_day_max",
)


# ==============================================================================
# Spinach Sensor Features & Blacklist
# ==============================================================================

# Physical hard bounds for spinach sensors:
# - air_temperature_C: 0 to 55 °C (greenhouse logger plausibility)
# - relative_humidity_percent: 0 to 100 % (physical RH bounds)
# - water_temperature_C: 0 to 45 °C (solution liquid bounds)
# - ph: 3.0 to 10.0 (nutrient solution chemical limits)
# - ec_mS_cm: 0.0 to 6.0 mS/cm (hydroponic solution solubility limits)
# Normal bands are defined from domain agronomy and dataset generation rules:
# - normal pH 6.0-7.0 (deviation triggers pH_Imbalance)
# - normal EC 1.8-2.3 (deviation triggers EC_Low / EC_High)
# - normal RH 45-75 (deviation triggers Low_Stress / Mildew_Risk)
# - normal water temp <= 21 °C (deviation triggers Water_Temp_High)

SPINACH_FEATURES: tuple[SensorFeature, ...] = (
    SensorFeature(
        name="air_temperature_C",
        unit="degC",
        hard_bounds=(0.0, 55.0),
        normal_band=(15.0, 25.0),
        description="Greenhouse logger air temperature",
    ),
    SensorFeature(
        name="relative_humidity_percent",
        unit="%",
        hard_bounds=(0.0, 100.0),
        normal_band=(45.0, 75.0),
        description="Greenhouse relative humidity",
    ),
    SensorFeature(
        name="water_temperature_C",
        unit="degC",
        hard_bounds=(0.0, 45.0),
        normal_band=(15.0, 21.0),
        description="Nutrient solution water temperature",
    ),
    SensorFeature(
        name="ph",
        unit="pH",
        hard_bounds=(3.0, 10.0),
        normal_band=(6.0, 7.0),
        description="Nutrient solution pH",
    ),
    SensorFeature(
        name="ec_mS_cm",
        unit="mS/cm",
        hard_bounds=(0.0, 6.0),
        normal_band=(1.8, 2.3),
        description="Nutrient solution electrical conductivity",
    ),
)

SPINACH_ORDERED_FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in SPINACH_FEATURES)

SPINACH_BLACKLIST_EXACT: frozenset[str] = frozenset(
    {
        "label",
        "label_id",
        "tds_ppm",
        "humidity_source",
        "sample_id",
        "source_timestamp_EDT",
        "bench",
    }
)

SPINACH_BLACKLIST_PATTERNS: tuple[str, ...] = ()


# ==============================================================================
# Tomato Sensor Features & Blacklist
# ==============================================================================

# Physical hard bounds for tomato sensors:
# - air_temperature_C: 0 to 55 °C
# - relative_humidity_percent: 0 to 100 %
# - water_temperature_C: 0 to 45 °C
# - ph: 3.0 to 10.0
# - ec_mS_cm: 0.0 to 8.0 mS/cm
# - nitrogen_mg_L: 0 to 500 mg/L (solution concentration physical limits)
# - phosphorus_mg_L: 0 to 200 mg/L (solution concentration physical limits)
# - potassium_mg_L: 0 to 800 mg/L (solution concentration physical limits)
# Normal bands are defined from hydroponic tomato formulation (CFAES / UA-CEA):
# - N: 90-250 mg/L (below 90 triggers N_Deficiency)
# - P: 35-100 mg/L (below 35 triggers P_Deficiency)
# - K: 144-400 mg/L (below 144 triggers K_Deficiency)
# - pH: 6.0-6.5 (outside triggers pH_Imbalance)
# - EC: 2.0-4.0 mS/cm (outside triggers EC_Low / EC_High)
# - Water temp: <= 25 °C (above triggers Water_Temp_High)
# - Humidity: 50-85 % (outside triggers Low_Stress / Mildew_Risk)

TOMATO_FEATURES: tuple[SensorFeature, ...] = (
    SensorFeature(
        name="air_temperature_C",
        unit="degC",
        hard_bounds=(0.0, 55.0),
        normal_band=(18.0, 28.0),
        description="Greenhouse ambient air temperature",
    ),
    SensorFeature(
        name="relative_humidity_percent",
        unit="%",
        hard_bounds=(0.0, 100.0),
        normal_band=(50.0, 85.0),
        description="Greenhouse ambient relative humidity",
    ),
    SensorFeature(
        name="water_temperature_C",
        unit="degC",
        hard_bounds=(0.0, 45.0),
        normal_band=(18.0, 25.0),
        description="Hydroponic root-zone water temperature",
    ),
    SensorFeature(
        name="ph",
        unit="pH",
        hard_bounds=(3.0, 10.0),
        normal_band=(6.0, 6.5),
        description="Hydroponic solution pH",
    ),
    SensorFeature(
        name="ec_mS_cm",
        unit="mS/cm",
        hard_bounds=(0.0, 8.0),
        normal_band=(2.0, 4.0),
        description="Electrical conductivity in mS/cm",
    ),
    SensorFeature(
        name="nitrogen_mg_L",
        unit="mg/L",
        hard_bounds=(0.0, 500.0),
        normal_band=(90.0, 250.0),
        description="Dissolved nitrogen concentration in solution",
    ),
    SensorFeature(
        name="phosphorus_mg_L",
        unit="mg/L",
        hard_bounds=(0.0, 200.0),
        normal_band=(35.0, 100.0),
        description="Dissolved phosphorus concentration in solution",
    ),
    SensorFeature(
        name="potassium_mg_L",
        unit="mg/L",
        hard_bounds=(0.0, 800.0),
        normal_band=(144.0, 400.0),
        description="Dissolved potassium concentration in solution",
    ),
)

TOMATO_ORDERED_FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in TOMATO_FEATURES)

TOMATO_BLACKLIST_EXACT: frozenset[str] = frozenset(
    {
        "label",
        "label_id",
        "n_rules_broken",
        "source_row",
        "source_timestamp",
        "sample_id",
    }
)

TOMATO_BLACKLIST_PATTERNS: tuple[str, ...] = ()


# ==============================================================================
# Crop Schema Container & Registries
# ==============================================================================


@dataclass(frozen=True)
class CropSensorSchema:
    """Sensor schema configuration for a crop."""

    crop: str
    features: tuple[SensorFeature, ...]
    blacklist_exact: frozenset[str]
    blacklist_patterns: tuple[str, ...] = ()

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Audited ordered list of feature column names."""
        return tuple(f.name for f in self.features)

    @property
    def hard_bounds(self) -> dict[str, tuple[float, float]]:
        """Mapping from feature name to physical hard bounds (min, max)."""
        return {f.name: f.hard_bounds for f in self.features}

    @property
    def normal_bands(self) -> dict[str, tuple[float, float]]:
        """Mapping from feature name to agronomic optimal range (min, max)."""
        return {f.name: f.normal_band for f in self.features}

    @property
    def units(self) -> dict[str, str]:
        """Mapping from feature name to engineering unit."""
        return {f.name: f.unit for f in self.features}


CROP_SENSOR_SCHEMAS: dict[str, CropSensorSchema] = {
    "lettuce": CropSensorSchema(
        crop="lettuce",
        features=LETTUCE_FEATURES,
        blacklist_exact=LETTUCE_BLACKLIST_EXACT,
        blacklist_patterns=LETTUCE_BLACKLIST_PATTERNS,
    ),
    "spinach": CropSensorSchema(
        crop="spinach",
        features=SPINACH_FEATURES,
        blacklist_exact=SPINACH_BLACKLIST_EXACT,
        blacklist_patterns=SPINACH_BLACKLIST_PATTERNS,
    ),
    "tomato": CropSensorSchema(
        crop="tomato",
        features=TOMATO_FEATURES,
        blacklist_exact=TOMATO_BLACKLIST_EXACT,
        blacklist_patterns=TOMATO_BLACKLIST_PATTERNS,
    ),
}


# ==============================================================================
# Leakage and Schema Enforcement Functions
# ==============================================================================


def get_sensor_schema(crop: str) -> CropSensorSchema:
    """Retrieve the sensor schema for the specified crop."""
    normalized_crop = crop.strip().lower()
    if normalized_crop not in CROP_SENSOR_SCHEMAS:
        raise ValueError(
            f"Unknown crop '{crop}'. Valid crops are: {sorted(CROP_SENSOR_SCHEMAS.keys())}"
        )
    return CROP_SENSOR_SCHEMAS[normalized_crop]


def get_leakage_blacklist(crop: str) -> frozenset[str]:
    """Return the set of explicitly blacklisted exact column names for a crop."""
    schema = get_sensor_schema(crop)
    return schema.blacklist_exact


def is_column_blacklisted(column_name: str, crop: str) -> bool:
    """Check whether a single column name is blacklisted for the given crop."""
    schema = get_sensor_schema(crop)
    if column_name in schema.blacklist_exact:
        return True
    return any(fnmatch.fnmatch(column_name, pat) for pat in schema.blacklist_patterns)


def find_blacklisted_columns(columns: Iterable[str], crop: str) -> list[str]:
    """Return all columns from the iterable that violate the crop's leakage blacklist."""
    schema = get_sensor_schema(crop)
    violating: list[str] = []
    for col in columns:
        if col in schema.blacklist_exact or any(fnmatch.fnmatch(col, pat) for pat in schema.blacklist_patterns):
            violating.append(col)
    return sorted(violating)


def assert_no_leakage(
    columns_or_df: Iterable[str] | pd.DataFrame,
    crop: str,
) -> None:
    """Enforce leakage blacklists, raising DataLeakageError if any blacklisted column is present.

    Args:
        columns_or_df: Iterable of column names or a pandas DataFrame.
        crop: Target crop name ('lettuce', 'spinach', 'tomato').

    Raises:
        DataLeakageError: If one or more blacklisted/leakage columns are detected.
    """
    if isinstance(columns_or_df, pd.DataFrame):
        cols = list(columns_or_df.columns)
    else:
        cols = list(columns_or_df)

    violating = find_blacklisted_columns(cols, crop)
    if violating:
        raise DataLeakageError(
            f"Data leakage detected! Columns {violating} are blacklisted for crop '{crop}'. "
            f"Blacklisted features (future targets, trends, IDs, metadata) must never enter "
            f"training or inference matrices."
        )


def validate_feature_matrix(
    df: pd.DataFrame,
    crop: str,
    strict_order: bool = True,
    check_bounds: bool = True,
) -> pd.DataFrame:
    """Validate a DataFrame against the crop's sensor feature schema and leakage rules.

    Args:
        df: Input feature matrix to validate.
        crop: Crop name ('lettuce', 'spinach', 'tomato').
        strict_order: When True, requires df columns to match the exact audited feature order.
        check_bounds: When True, asserts that all values respect physical hard bounds.

    Returns:
        pd.DataFrame containing only the schema features in audited order.

    Raises:
        DataLeakageError: If blacklisted columns are present.
        SchemaValidationError: If required features are missing, out of order, or violate hard bounds.
    """
    assert_no_leakage(df, crop)

    schema = get_sensor_schema(crop)
    expected_features = list(schema.feature_names)

    missing = set(expected_features) - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"Missing required sensor features for '{crop}': {sorted(missing)}. "
            f"Expected features: {expected_features}"
        )

    if strict_order:
        current_cols = list(df.columns)[: len(expected_features)]
        if current_cols != expected_features:
            raise SchemaValidationError(
                f"Feature columns do not match audited order for '{crop}'.\n"
                f"Expected: {expected_features}\n"
                f"Actual:   {current_cols}"
            )

    extracted_df = df[expected_features].copy()

    if check_bounds:
        hard_bounds = schema.hard_bounds
        bound_violations: list[str] = []
        for feat, (lo, hi) in hard_bounds.items():
            col_series = extracted_df[feat]
            below = col_series < lo
            above = col_series > hi
            if below.any() or above.any():
                n_bad = int((below | above).sum())
                min_val = float(col_series.min())
                max_val = float(col_series.max())
                bound_violations.append(
                    f"{feat}: {n_bad} values outside [{lo}, {hi}] (observed [{min_val}, {max_val}])"
                )

        if bound_violations:
            raise SchemaValidationError(
                f"Hard bound violations detected in '{crop}' sensor inputs (physically impossible):\n"
                + "\n".join(f" - {v}" for v in bound_violations)
            )

    return extracted_df


def extract_feature_matrix(
    df: pd.DataFrame,
    crop: str,
    check_bounds: bool = True,
) -> pd.DataFrame:
    """Safely extract only the audited sensor feature columns from a raw dataset DataFrame.

    Ensures none of the extracted columns violate the leakage blacklist, re-orders them to match
    the audited feature schema, and optionally verifies physical hard bounds.

    Args:
        df: Raw dataset DataFrame containing feature columns alongside metadata/targets.
        crop: Crop name ('lettuce', 'spinach', 'tomato').
        check_bounds: If True, validates that all values lie within physical hard bounds.

    Returns:
        pd.DataFrame containing strictly the schema features in audited order.
    """
    schema = get_sensor_schema(crop)
    expected_features = list(schema.feature_names)

    missing = set(expected_features) - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"Missing required sensor features for '{crop}': {sorted(missing)}. "
            f"Expected features: {expected_features}"
        )

    feature_df = df[expected_features].copy()
    return validate_feature_matrix(
        feature_df,
        crop=crop,
        strict_order=True,
        check_bounds=check_bounds,
    )

