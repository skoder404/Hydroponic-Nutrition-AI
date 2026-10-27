"""Unit tests for live-compatible tomato sensor variant and live adapter (T4-020).

Validates:
- Feature availability table matches ADR 0008 campus probe findings.
- Variant class list excludes N/P/K deficiencies (7 classes).
- Live adapter maps only the 5 variant features and rejects/drops N/P/K.
- TomatoLiveSensorRequest enforces extra='forbid' and validates ranges.
- ModelMetadata serializes feature_set and preserves experimental=True.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from backend.app.schemas.sensors import (
    TomatoLiveSensorRequest,
    validate_sensor_request,
)
from ml.adapters.live_tomato_adapter import (
    FEATURE_AVAILABILITY_TABLE,
    LiveAdapterMappingError,
    format_feature_availability_markdown,
    map_campus_to_live_tomato_features,
)
from ml.labels import (
    SENSOR_CLASSES,
    TOMATO_LIVE_SENSOR_CLASSES,
    TOMATO_SENSOR_CLASSES,
)
from ml.registry.metadata_schema import ModelMetadata
from ml.sensor_models.schemas import (
    TOMATO_LIVE_ORDERED_FEATURE_NAMES,
    TOMATO_ORDERED_FEATURE_NAMES,
)


class TestFeatureAvailabilityTable:
    """Acceptance Criterion 1: Feature availability table in report and adapter."""

    def test_feature_availability_table_covers_all_eight_tomato_features(self) -> None:
        table_features = [row["model_feature"] for row in FEATURE_AVAILABILITY_TABLE]
        assert len(table_features) == 8
        assert table_features == list(TOMATO_ORDERED_FEATURE_NAMES)

    def test_npk_features_marked_missing_and_disabled(self) -> None:
        npk_rows = [
            row
            for row in FEATURE_AVAILABILITY_TABLE
            if row["model_feature"]
            in ("nitrogen_mg_L", "phosphorus_mg_L", "potassium_mg_L")
        ]
        assert len(npk_rows) == 3
        for row in npk_rows:
            assert "Missing" in row["status"] or "Disabled" in row["status"]
            assert "_deleted_keys" in row["notes"] or "deleted" in row["notes"].lower()

    def test_live_features_have_valid_units_and_server_fields(self) -> None:
        row_map = {row["model_feature"]: row for row in FEATURE_AVAILABILITY_TABLE}
        assert row_map["air_temperature_C"]["server_field"] == "temp_avg"
        assert row_map["relative_humidity_percent"]["server_field"] == "humidity_avg"
        assert row_map["ph"]["server_field"] == "ph"
        assert row_map["ec_mS_cm"]["server_field"] == "ec"
        assert "Scaled" in row_map["ec_mS_cm"]["conversion"]

    def test_format_feature_availability_markdown_renders_gfm_table(self) -> None:
        md = format_feature_availability_markdown()
        assert "| Model Feature | Server Field |" in md
        assert "`air_temperature_C`" in md
        assert "`nitrogen_mg_L`" in md
        assert "**Available**" in md
        assert "**Missing / Disabled**" in md


class TestClassListAndTaxonomy:
    """Acceptance Criterion 2: Variant has its own class list excluding N/P/K."""

    def test_variant_class_count_is_seven(self) -> None:
        assert len(TOMATO_LIVE_SENSOR_CLASSES) == 7
        assert len(TOMATO_SENSOR_CLASSES) == 10

    def test_npk_deficiencies_strictly_excluded(self) -> None:
        excluded = {"N_Deficiency", "P_Deficiency", "K_Deficiency"}
        for cls_name in excluded:
            assert cls_name not in TOMATO_LIVE_SENSOR_CLASSES
            assert cls_name in TOMATO_SENSOR_CLASSES

    def test_retained_classes(self) -> None:
        expected = {
            "EC_High_Toxicity",
            "EC_Low_Deficiency",
            "Humidity_High_Mildew_Risk",
            "Humidity_Low_Stress",
            "Normal",
            "Water_Temp_High",
            "pH_Imbalance",
        }
        assert set(TOMATO_LIVE_SENSOR_CLASSES) == expected

    def test_registered_in_sensor_classes_mapping(self) -> None:
        assert "tomato_live" in SENSOR_CLASSES
        assert SENSOR_CLASSES["tomato_live"] == TOMATO_LIVE_SENSOR_CLASSES


class TestLiveAdapterMapping:
    """Acceptance Criterion 3: Live adapter maps only to features the variant uses."""

    @pytest.fixture
    def live_gateway_payload(self) -> dict[str, Any]:
        """Real sample telemetry modeled after live_device_payload.json."""
        return {
            "temp_avg": 40.0,
            "temperature1": 41.2,
            "temperature2": 38.9,
            "humidity_avg": 36.8,
            "humidity1": 35.1,
            "humidity2": 38.5,
            "ph": 7.67,
            "ec": 36.8,  # in µS/cm
            "tds": 18.3,
            "salinity": 0.02,
            "nitrogen_mg_L": 120.0,  # simulate rogue key in raw payload
            "phosphorus": 45.0,
            "potassium": 200.0,
            "device_online": True,
        }

    def test_maps_strictly_to_five_live_features(
        self, live_gateway_payload: dict[str, Any]
    ) -> None:
        mapped = map_campus_to_live_tomato_features(live_gateway_payload)
        assert list(mapped.keys()) == list(TOMATO_LIVE_ORDERED_FEATURE_NAMES)
        assert len(mapped) == 5

    def test_ec_scaling_from_microsiemens_to_millisiemens(
        self, live_gateway_payload: dict[str, Any]
    ) -> None:
        mapped = map_campus_to_live_tomato_features(live_gateway_payload)
        # 36.8 µS/cm -> 0.0368 mS/cm
        assert abs(mapped["ec_mS_cm"] - 0.0368) < 1e-4

    def test_thermal_proxy_for_water_temperature(
        self, live_gateway_payload: dict[str, Any]
    ) -> None:
        mapped = map_campus_to_live_tomato_features(live_gateway_payload)
        # Proxy formula: 8.0 + 0.55 * air_temp = 8.0 + 0.55 * 40.0 = 30.0
        assert abs(mapped["water_temperature_C"] - 30.0) < 1e-2

    def test_direct_immersion_water_temp_used_if_supplied(
        self, live_gateway_payload: dict[str, Any]
    ) -> None:
        live_gateway_payload["water_temp"] = 22.5
        mapped = map_campus_to_live_tomato_features(live_gateway_payload)
        assert mapped["water_temperature_C"] == 22.5

    def test_strictly_drops_npk_features(
        self, live_gateway_payload: dict[str, Any]
    ) -> None:
        mapped = map_campus_to_live_tomato_features(live_gateway_payload)
        for prohibited in (
            "nitrogen_mg_L",
            "phosphorus_mg_L",
            "potassium_mg_L",
            "nitrogen",
            "phosphorus",
            "potassium",
        ):
            assert prohibited not in mapped

    def test_raises_when_required_telemetry_missing(self) -> None:
        incomplete = {"temp_avg": 25.0}  # missing humidity, pH, EC
        with pytest.raises(LiveAdapterMappingError, match="Missing humidity"):
            map_campus_to_live_tomato_features(incomplete)


class TestRequestValidators:
    """Test TomatoLiveSensorRequest Pydantic validation."""

    def test_valid_request(self) -> None:
        payload = {
            "air_temperature_C": 25.0,
            "relative_humidity_percent": 65.0,
            "water_temperature_C": 21.0,
            "ph": 6.2,
            "ec_mS_cm": 2.5,
        }
        req = validate_sensor_request("tomato_live", payload)
        assert isinstance(req, TomatoLiveSensorRequest)
        d = req.to_feature_dict()
        assert len(d) == 5
        assert d["ph"] == 6.2

    def test_extra_fields_forbidden(self) -> None:
        payload = {
            "air_temperature_C": 25.0,
            "relative_humidity_percent": 65.0,
            "water_temperature_C": 21.0,
            "ph": 6.2,
            "ec_mS_cm": 2.5,
            "nitrogen_mg_L": 120.0,  # extra field not in live variant
        }
        with pytest.raises(ValidationError):
            validate_sensor_request("tomato_live", payload)


class TestModelMetadataFeatureSet:
    """Test ModelMetadata feature_set field and serialization."""

    def test_metadata_serializes_feature_set(self) -> None:
        meta_dict = {
            "model_id": "tomato_live_sensor_random_forest_v1.0.0",
            "crop": "tomato_live",
            "task": "classification",
            "modality": "sensor",
            "family": "RandomForestClassifier",
            "artifact_version": "1.0.0",
            "artifact_file": "sensor/tomato_live/v1.0.0/model.joblib",
            "commit": "b6f1df1f87c2b3e8",
            "training_date": "2026-10-27T12:00:00+00:00",
            "dataset_manifest_hash": "c62bf6a127fce1c4efdfa88d752ad755a7201b2fba9eb73b306b9b1e92d6e3c5",
            "rule_version": "1.0.0",
            "preprocessing_version": "1.0.0",
            "experimental": True,
            "status": "candidate",
            "classes": list(TOMATO_LIVE_SENSOR_CLASSES),
            "display_mapping": {
                c: ("healthy" if c == "Normal" else f"unhealthy/{c}")
                for c in TOMATO_LIVE_SENSOR_CLASSES
            },
            "split_protocol": "Calendar month holdout",
            "split_counts": {"train": 20000, "validation": 3500, "test": 0},
            "ordered_features": list(TOMATO_LIVE_ORDERED_FEATURE_NAMES),
            "feature_units": {"ph": "pH", "ec_mS_cm": "mS/cm"},
            "feature_set": "live_compatible",
            "input_unit": "sensor_reading",
            "hyperparameters": {"n_estimators": 100},
            "confidence_settings": {"threshold": 0.4},
            "expected_input_schema": {"type": "object"},
            "dependency_versions": {"python": "3.14"},
            "validation_metrics": {"macro_f1": 0.95},
        }
        meta = ModelMetadata.model_validate(meta_dict)
        assert meta.feature_set == "live_compatible"
        assert meta.experimental is True
        serialized = json.loads(meta.to_json())
        assert serialized["feature_set"] == "live_compatible"
