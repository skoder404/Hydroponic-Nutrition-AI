"""Unit tests for labels, schemas, and validators (T8-001).

Acceptance Criteria:
1. Every Overview Section 6 class is covered (all 21 image classes + all sensor classes).
2. Missing vs 0 behaviour is strictly asserted (None/null is never coerced to 0.0).
3. Sensor feature ordering, units, and hard physical ranges are validated per crop.
4. Unknown fields and data leakage columns are rejected (extra='forbid').
5. Error-envelope structure, status codes, and retryable flags are verified.
6. All tests pass cleanly in CI.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.core.errors import (
    AppException,
    DeviceUnavailableError,
    InvalidInputError,
    MissingFieldError,
    ModelUnavailableError,
    PayloadTooLargeError,
    PersistenceFailedError,
    RateLimitExceededError,
    SchemaMismatchError,
    SegmentationFailedError,
    UnsupportedMediaError,
    register_error_handlers,
)
from backend.app.schemas.sensors import (
    LettuceSensorRequest,
    SpinachSensorRequest,
    TomatoSensorRequest,
    get_sensor_request_model,
    validate_sensor_request,
)
from ml.labels import (
    ALL_IMAGE_CLASSES,
    HEALTHY_IMAGE_LABELS,
    HEALTHY_SENSOR_LABELS,
    LETTUCE_IMAGE_CLASSES,
    LETTUCE_SENSOR_CLASSES,
    SPINACH_IMAGE_CLASSES,
    SPINACH_SENSOR_CLASSES,
    TOMATO_IMAGE_CLASSES,
    TOMATO_SENSOR_CLASSES,
    UnknownCropError,
    UnknownLabelError,
    UnknownModalityError,
    get_class_from_index,
    get_class_index,
    get_classes,
    get_display_label,
    num_classes,
    validate_label,
)
from ml.sensor_models.schemas import (
    LETTUCE_ORDERED_FEATURE_NAMES,
    SPINACH_ORDERED_FEATURE_NAMES,
    TOMATO_ORDERED_FEATURE_NAMES,
    get_sensor_schema,
)

# ==============================================================================
# Overview Section 6: Image Class Taxonomy & Display Mapping (All 21 Classes)
# ==============================================================================


class TestImageClassTaxonomy:
    """Validate all 21 image classes across lettuce, spinach, and tomato."""

    def test_total_image_class_count(self) -> None:
        """Overview Section 6: There are exactly 21 image classes total."""
        assert len(ALL_IMAGE_CLASSES) == 21
        assert len(LETTUCE_IMAGE_CLASSES) == 4
        assert len(SPINACH_IMAGE_CLASSES) == 6
        assert len(TOMATO_IMAGE_CLASSES) == 11

    @pytest.mark.parametrize(
        "crop,expected_classes",
        [
            (
                "lettuce",
                (
                    "Healthier",
                    "NitrogenDeficiency",
                    "PhosphorusDeficiency",
                    "PotassiumDeficiency",
                ),
            ),
            (
                "spinach",
                (
                    "Spinach_Healthy",
                    "Spinach_Non fresh",
                    "Spinach_anthracnose",
                    "Spinach_bacterial spot",
                    "Spinach_downy mildew",
                    "Spinach_pest damage",
                ),
            ),
            (
                "tomato",
                (
                    "Tomato_Bacterial_spot",
                    "Tomato_Early_blight",
                    "Tomato_Healthy",
                    "Tomato_Late_blight",
                    "Tomato_Leaf_Mold",
                    "Tomato_Powdery_Mildew",
                    "Tomato_Septoria_leaf_spot",
                    "Tomato_Spider_mites Two-spotted_spider_mite",
                    "Tomato_Target_Spot",
                    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
                    "Tomato_Tomato_mosaic_virus",
                ),
            ),
        ],
    )
    def test_all_classes_exact_names(
        self, crop: str, expected_classes: tuple[str, ...]
    ) -> None:
        """Verify registered classes exactly match Overview Section 6 on-disk folder names."""
        registered = get_classes(crop, "image")
        assert registered == expected_classes
        assert num_classes(crop, "image") == len(expected_classes)

    @pytest.mark.parametrize(
        "crop,label,expected_display",
        [
            # Lettuce image classes (4)
            ("lettuce", "Healthier", "healthy"),
            ("lettuce", "NitrogenDeficiency", "unhealthy/NitrogenDeficiency"),
            ("lettuce", "PhosphorusDeficiency", "unhealthy/PhosphorusDeficiency"),
            ("lettuce", "PotassiumDeficiency", "unhealthy/PotassiumDeficiency"),
            # Spinach image classes (6)
            ("spinach", "Spinach_Healthy", "healthy"),
            ("spinach", "Spinach_Non fresh", "unhealthy/Spinach_Non fresh"),
            ("spinach", "Spinach_anthracnose", "unhealthy/Spinach_anthracnose"),
            ("spinach", "Spinach_bacterial spot", "unhealthy/Spinach_bacterial spot"),
            ("spinach", "Spinach_downy mildew", "unhealthy/Spinach_downy mildew"),
            ("spinach", "Spinach_pest damage", "unhealthy/Spinach_pest damage"),
            # Tomato image classes (11)
            ("tomato", "Tomato_Healthy", "healthy"),
            ("tomato", "Tomato_Bacterial_spot", "unhealthy/Tomato_Bacterial_spot"),
            ("tomato", "Tomato_Early_blight", "unhealthy/Tomato_Early_blight"),
            ("tomato", "Tomato_Late_blight", "unhealthy/Tomato_Late_blight"),
            ("tomato", "Tomato_Leaf_Mold", "unhealthy/Tomato_Leaf_Mold"),
            ("tomato", "Tomato_Powdery_Mildew", "unhealthy/Tomato_Powdery_Mildew"),
            (
                "tomato",
                "Tomato_Septoria_leaf_spot",
                "unhealthy/Tomato_Septoria_leaf_spot",
            ),
            (
                "tomato",
                "Tomato_Spider_mites Two-spotted_spider_mite",
                "unhealthy/Tomato_Spider_mites Two-spotted_spider_mite",
            ),
            ("tomato", "Tomato_Target_Spot", "unhealthy/Tomato_Target_Spot"),
            (
                "tomato",
                "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
                "unhealthy/Tomato_Tomato_Yellow_Leaf_Curl_Virus",
            ),
            (
                "tomato",
                "Tomato_Tomato_mosaic_virus",
                "unhealthy/Tomato_Tomato_mosaic_virus",
            ),
        ],
    )
    def test_image_display_mapping_for_all_21_classes(
        self, crop: str, label: str, expected_display: str
    ) -> None:
        """Every one of the 21 image classes maps to healthy or unhealthy/<original class>."""
        assert get_display_label(crop, label, "image") == expected_display
        assert validate_label(crop, label, "image") == label

    def test_healthy_image_labels_set(self) -> None:
        """Verify HEALTHY_IMAGE_LABELS contains all healthy variants."""
        assert "Healthier" in HEALTHY_IMAGE_LABELS
        assert "Spinach_Healthy" in HEALTHY_IMAGE_LABELS
        assert "Tomato_Healthy" in HEALTHY_IMAGE_LABELS

    def test_exact_spelling_and_spaces_preserved(self) -> None:
        """Spelling, spaces, and casing must be preserved exactly."""
        spider_label = "Tomato_Spider_mites Two-spotted_spider_mite"
        assert spider_label in TOMATO_IMAGE_CLASSES
        assert (
            get_display_label("tomato", spider_label, "image")
            == f"unhealthy/{spider_label}"
        )

        powdery_label = "Tomato_Powdery_Mildew"
        assert powdery_label in TOMATO_IMAGE_CLASSES

    def test_image_class_bidirectional_indexing(self) -> None:
        """Verify 0-indexed integer mapping and inverse lookup for all image classes."""
        for crop in ("lettuce", "spinach", "tomato"):
            classes = get_classes(crop, "image")
            for idx, name in enumerate(classes):
                assert get_class_index(crop, name, "image") == idx
                assert get_class_from_index(crop, idx, "image") == name

            with pytest.raises(IndexError):
                get_class_from_index(crop, -1, "image")
            with pytest.raises(IndexError):
                get_class_from_index(crop, len(classes), "image")


# ==============================================================================
# Overview Section 6: Sensor Class Taxonomy & Display Mapping
# ==============================================================================


class TestSensorClassTaxonomy:
    """Validate all sensor classes across lettuce, spinach, and tomato."""

    @pytest.mark.parametrize(
        "crop,expected_classes",
        [
            (
                "lettuce",
                (
                    "EC_High",
                    "EC_Low",
                    "Humidity_Low",
                    "Multiple_Stress",
                    "Normal",
                    "WaterTemp_High",
                    "WaterTemp_Low",
                    "pH_High",
                    "pH_Low",
                ),
            ),
            (
                "spinach",
                (
                    "EC_High_Toxicity",
                    "EC_Low_Deficiency",
                    "Humidity_High_Mildew_Risk",
                    "Humidity_Low_Stress",
                    "Normal",
                    "Water_Temp_High",
                    "pH_Imbalance",
                ),
            ),
            (
                "tomato",
                (
                    "EC_High_Toxicity",
                    "EC_Low_Deficiency",
                    "Humidity_High_Mildew_Risk",
                    "Humidity_Low_Stress",
                    "K_Deficiency",
                    "N_Deficiency",
                    "Normal",
                    "P_Deficiency",
                    "Water_Temp_High",
                    "pH_Imbalance",
                ),
            ),
        ],
    )
    def test_all_sensor_classes_exact_names(
        self, crop: str, expected_classes: tuple[str, ...]
    ) -> None:
        """Verify registered sensor classes exactly match Overview Section 6.4."""
        registered = get_classes(crop, "sensor")
        assert registered == expected_classes
        assert num_classes(crop, "sensor") == len(expected_classes)

    @pytest.mark.parametrize(
        "crop,sensor_classes",
        [
            ("lettuce", LETTUCE_SENSOR_CLASSES),
            ("spinach", SPINACH_SENSOR_CLASSES),
            ("tomato", TOMATO_SENSOR_CLASSES),
        ],
    )
    def test_sensor_display_mapping(
        self, crop: str, sensor_classes: tuple[str, ...]
    ) -> None:
        """Every sensor class maps to healthy (for Normal) or unhealthy/<original class>."""
        for label in sensor_classes:
            display = get_display_label(crop, label, "sensor")
            if label == "Normal":
                assert display == "healthy"
            else:
                assert display == f"unhealthy/{label}"
            assert validate_label(crop, label, "sensor") == label

    def test_healthy_sensor_labels_set(self) -> None:
        """Verify HEALTHY_SENSOR_LABELS contains 'Normal'."""
        assert "Normal" in HEALTHY_SENSOR_LABELS

    def test_sensor_and_image_taxonomies_are_not_interchangeable(self) -> None:
        """Overview Section 6.4: Sensor and image taxonomies must not be confused."""
        # Sensor class cannot be queried as image class
        with pytest.raises(UnknownLabelError):
            validate_label("lettuce", "EC_High", "image")

        with pytest.raises(UnknownLabelError):
            validate_label("spinach", "EC_High_Toxicity", "image")

        # Image class cannot be queried as sensor class
        with pytest.raises(UnknownLabelError):
            validate_label("lettuce", "Healthier", "sensor")

        with pytest.raises(UnknownLabelError):
            validate_label("tomato", "Tomato_Healthy", "sensor")


# ==============================================================================
# Unknown Crop, Modality, and Label Rejections
# ==============================================================================


class TestUnknownRejections:
    """Validate error behaviour on unknown crops, modalities, and labels."""

    def test_unknown_crop_raises_unknown_crop_error(self) -> None:
        with pytest.raises(UnknownCropError, match="Unknown crop 'cucumber'"):
            get_classes("cucumber", "image")

        with pytest.raises(UnknownCropError, match="Unknown crop 'kale'"):
            validate_label("kale", "Healthier", "image")

        with pytest.raises(UnknownCropError, match="Unknown crop 'kale'"):
            get_display_label("kale", "Healthier", "image")

    def test_unknown_modality_raises_unknown_modality_error(self) -> None:
        with pytest.raises(UnknownModalityError, match="Unknown modality 'audio'"):
            get_classes("tomato", "audio")

        with pytest.raises(UnknownModalityError, match="Unknown modality 'video'"):
            validate_label("tomato", "Tomato_Healthy", "video")

    def test_unknown_label_raises_unknown_label_error(self) -> None:
        with pytest.raises(UnknownLabelError, match="Unknown image label 'Rotten'"):
            validate_label("lettuce", "Rotten", "image")

        with pytest.raises(UnknownLabelError, match="Unknown sensor label 'Root_Rot'"):
            validate_label("tomato", "Root_Rot", "sensor")


# ==============================================================================
# Sensor Feature Ordering & Schema Representation
# ==============================================================================


class TestSensorFeatureOrdering:
    """Validate feature order across all 3 crops matches audited datasets."""

    def test_lettuce_feature_order(self) -> None:
        payload = {
            "ph": 6.2,
            "ec": 1.8,
            "water_temp": 21.5,
            "humidity": 65.0,
            "air_temp": 23.0,
        }
        req = LettuceSensorRequest.model_validate(payload)
        feature_dict = req.to_feature_dict()
        assert tuple(feature_dict.keys()) == LETTUCE_ORDERED_FEATURE_NAMES
        assert tuple(feature_dict.keys()) == (
            "ph",
            "ec",
            "water_temp",
            "humidity",
            "air_temp",
        )

    def test_spinach_feature_order(self) -> None:
        payload = {
            "air_temperature_C": 20.0,
            "relative_humidity_percent": 60.0,
            "water_temperature_C": 19.0,
            "ph": 6.5,
            "ec_mS_cm": 2.1,
        }
        req = SpinachSensorRequest.model_validate(payload)
        feature_dict = req.to_feature_dict()
        assert tuple(feature_dict.keys()) == SPINACH_ORDERED_FEATURE_NAMES
        assert tuple(feature_dict.keys()) == (
            "air_temperature_C",
            "relative_humidity_percent",
            "water_temperature_C",
            "ph",
            "ec_mS_cm",
        )

    def test_tomato_feature_order(self) -> None:
        payload = {
            "air_temperature_C": 24.0,
            "relative_humidity_percent": 70.0,
            "water_temperature_C": 22.0,
            "ph": 6.3,
            "ec_mS_cm": 2.8,
            "nitrogen_mg_L": 150.0,
            "phosphorus_mg_L": 50.0,
            "potassium_mg_L": 250.0,
        }
        req = TomatoSensorRequest.model_validate(payload)
        feature_dict = req.to_feature_dict()
        assert tuple(feature_dict.keys()) == TOMATO_ORDERED_FEATURE_NAMES
        assert tuple(feature_dict.keys()) == (
            "air_temperature_C",
            "relative_humidity_percent",
            "water_temperature_C",
            "ph",
            "ec_mS_cm",
            "nitrogen_mg_L",
            "phosphorus_mg_L",
            "potassium_mg_L",
        )

    def test_factory_and_convenience_functions(self) -> None:
        """Verify get_sensor_request_model and validate_sensor_request."""
        assert get_sensor_request_model("lettuce") is LettuceSensorRequest
        assert get_sensor_request_model("spinach") is SpinachSensorRequest
        assert get_sensor_request_model("tomato") is TomatoSensorRequest

        with pytest.raises(ValueError, match="Unknown crop 'cucumber'"):
            get_sensor_request_model("cucumber")

        req = validate_sensor_request(
            "lettuce",
            {
                "ph": 6.0,
                "ec": 1.8,
                "water_temp": 20.0,
                "humidity": 60.0,
                "air_temp": 22.0,
            },
        )
        assert isinstance(req, LettuceSensorRequest)
        assert req.ph == 6.0


# ==============================================================================
# Missing vs Zero Handling (Acceptance Criterion 2)
# ==============================================================================


class TestMissingVsZeroHandling:
    """Strictly assert null/None is distinguished from 0.0 and never coerced."""

    @pytest.mark.parametrize(
        "crop,req_cls,valid_base",
        [
            (
                "lettuce",
                LettuceSensorRequest,
                {
                    "ph": 6.0,
                    "ec": 1.8,
                    "water_temp": 20.0,
                    "humidity": 60.0,
                    "air_temp": 22.0,
                },
            ),
            (
                "spinach",
                SpinachSensorRequest,
                {
                    "air_temperature_C": 20.0,
                    "relative_humidity_percent": 60.0,
                    "water_temperature_C": 19.0,
                    "ph": 6.5,
                    "ec_mS_cm": 2.1,
                },
            ),
            (
                "tomato",
                TomatoSensorRequest,
                {
                    "air_temperature_C": 24.0,
                    "relative_humidity_percent": 70.0,
                    "water_temperature_C": 22.0,
                    "ph": 6.3,
                    "ec_mS_cm": 2.8,
                    "nitrogen_mg_L": 150.0,
                    "phosphorus_mg_L": 50.0,
                    "potassium_mg_L": 250.0,
                },
            ),
        ],
    )
    def test_null_value_raises_missing_field_for_all_crops(
        self, crop: str, req_cls: type[Any], valid_base: dict[str, float]
    ) -> None:
        """Every sensor field when set to None/null raises MISSING_FIELD (not 0.0)."""
        for field_name in valid_base:
            payload = dict(valid_base)
            payload[field_name] = None
            with pytest.raises(ValidationError) as exc_info:
                req_cls.model_validate(payload)
            errors = exc_info.value.errors()
            err_for_field = [e for e in errors if e["loc"] == (field_name,)]
            assert len(err_for_field) == 1
            msg = err_for_field[0]["msg"]
            assert "MISSING_FIELD" in msg
            assert f"Field '{field_name}'" in msg
            assert "cannot be null" in msg

    @pytest.mark.parametrize(
        "crop,req_cls,valid_base",
        [
            (
                "lettuce",
                LettuceSensorRequest,
                {
                    "ph": 6.0,
                    "ec": 1.8,
                    "water_temp": 20.0,
                    "humidity": 60.0,
                    "air_temp": 22.0,
                },
            ),
            (
                "spinach",
                SpinachSensorRequest,
                {
                    "air_temperature_C": 20.0,
                    "relative_humidity_percent": 60.0,
                    "water_temperature_C": 19.0,
                    "ph": 6.5,
                    "ec_mS_cm": 2.1,
                },
            ),
            (
                "tomato",
                TomatoSensorRequest,
                {
                    "air_temperature_C": 24.0,
                    "relative_humidity_percent": 70.0,
                    "water_temperature_C": 22.0,
                    "ph": 6.3,
                    "ec_mS_cm": 2.8,
                    "nitrogen_mg_L": 150.0,
                    "phosphorus_mg_L": 50.0,
                    "potassium_mg_L": 250.0,
                },
            ),
        ],
    )
    def test_omitted_field_raises_missing_field_for_all_crops(
        self, crop: str, req_cls: type[Any], valid_base: dict[str, float]
    ) -> None:
        """Every sensor field when omitted raises MISSING_FIELD."""
        for field_name in valid_base:
            payload = dict(valid_base)
            del payload[field_name]
            with pytest.raises(ValidationError) as exc_info:
                req_cls.model_validate(payload)
            errors = exc_info.value.errors()
            err_for_field = [e for e in errors if e["loc"] == (field_name,)]
            assert len(err_for_field) == 1
            msg = err_for_field[0]["msg"]
            assert "MISSING_FIELD" in msg

    def test_zero_is_accepted_when_within_physical_bounds(self) -> None:
        """0.0 is valid numeric data and must NOT be treated as missing if within bounds."""
        # EC [0.0, 6.0], humidity [0.0, 100.0], water_temp [0.0, 45.0], air_temp [0.0, 55.0]
        payload = {
            "ph": 6.0,
            "ec": 0.0,
            "water_temp": 0.0,
            "humidity": 0.0,
            "air_temp": 0.0,
        }
        req = LettuceSensorRequest.model_validate(payload)
        assert req.ec == 0.0
        assert req.water_temp == 0.0
        assert req.humidity == 0.0
        assert req.air_temp == 0.0
        # Assert type is strictly float, not None
        assert isinstance(req.ec, float)

    def test_zero_raises_out_of_bounds_when_outside_physical_bounds(self) -> None:
        """0.0 raises OUT_OF_BOUNDS (and NEVER MISSING_FIELD) when below physical minimum."""
        # pH physical bounds are [3.0, 10.0]
        payload = {
            "ph": 0.0,
            "ec": 1.8,
            "water_temp": 20.0,
            "humidity": 60.0,
            "air_temp": 22.0,
        }
        with pytest.raises(ValidationError) as exc_info:
            LettuceSensorRequest.model_validate(payload)
        msg = exc_info.value.errors()[0]["msg"]
        assert "OUT_OF_BOUNDS" in msg
        assert "MISSING_FIELD" not in msg
        assert "Field 'ph'" in msg
        assert "0.0" in msg

    def test_boolean_values_rejected_not_coerced_to_zero_or_one(self) -> None:
        """Booleans (True/False) must be rejected with INVALID_TYPE, never coerced to 1.0/0.0."""
        for bool_val in (True, False):
            payload = {
                "ph": bool_val,
                "ec": 1.8,
                "water_temp": 20.0,
                "humidity": 60.0,
                "air_temp": 22.0,
            }
            with pytest.raises(ValidationError) as exc_info:
                LettuceSensorRequest.model_validate(payload)
            msg = exc_info.value.errors()[0]["msg"]
            assert "INVALID_TYPE" in msg
            assert "boolean" in msg


# ==============================================================================
# Unit & Physical Range Validation Across All Features
# ==============================================================================


class TestUnitAndRangeValidation:
    """Validate physical units and hard limits across all 3 crop schemas."""

    def test_all_lettuce_features_bounds(self) -> None:
        schema = get_sensor_schema("lettuce")
        base = {
            "ph": 6.0,
            "ec": 1.8,
            "water_temp": 20.0,
            "humidity": 60.0,
            "air_temp": 22.0,
        }

        for feat in schema.features:
            lo, hi = feat.hard_bounds
            unit = feat.unit

            # Boundary values must succeed
            valid_lo = dict(base)
            valid_lo[feat.name] = lo
            assert (
                getattr(LettuceSensorRequest.model_validate(valid_lo), feat.name) == lo
            )

            valid_hi = dict(base)
            valid_hi[feat.name] = hi
            assert (
                getattr(LettuceSensorRequest.model_validate(valid_hi), feat.name) == hi
            )

            # Below minimum
            invalid_below = dict(base)
            invalid_below[feat.name] = lo - 0.5
            with pytest.raises(ValidationError) as exc_lo:
                LettuceSensorRequest.model_validate(invalid_below)
            msg_lo = exc_lo.value.errors()[0]["msg"]
            assert "OUT_OF_BOUNDS" in msg_lo
            assert feat.name in msg_lo
            assert unit in msg_lo
            assert f"[{lo}, {hi}]" in msg_lo

            # Above maximum
            invalid_above = dict(base)
            invalid_above[feat.name] = hi + 0.5
            with pytest.raises(ValidationError) as exc_hi:
                LettuceSensorRequest.model_validate(invalid_above)
            msg_hi = exc_hi.value.errors()[0]["msg"]
            assert "OUT_OF_BOUNDS" in msg_hi
            assert feat.name in msg_hi
            assert unit in msg_hi
            assert f"[{lo}, {hi}]" in msg_hi

    def test_all_spinach_features_bounds(self) -> None:
        schema = get_sensor_schema("spinach")
        base = {
            "air_temperature_C": 20.0,
            "relative_humidity_percent": 60.0,
            "water_temperature_C": 19.0,
            "ph": 6.5,
            "ec_mS_cm": 2.1,
        }

        for feat in schema.features:
            lo, hi = feat.hard_bounds
            unit = feat.unit

            # Boundary values must succeed
            valid_lo = dict(base)
            valid_lo[feat.name] = lo
            assert (
                getattr(SpinachSensorRequest.model_validate(valid_lo), feat.name) == lo
            )

            valid_hi = dict(base)
            valid_hi[feat.name] = hi
            assert (
                getattr(SpinachSensorRequest.model_validate(valid_hi), feat.name) == hi
            )

            # Below minimum
            invalid_below = dict(base)
            invalid_below[feat.name] = lo - 1.0
            with pytest.raises(ValidationError) as exc_lo:
                SpinachSensorRequest.model_validate(invalid_below)
            msg_lo = exc_lo.value.errors()[0]["msg"]
            assert "OUT_OF_BOUNDS" in msg_lo
            assert feat.name in msg_lo
            assert unit in msg_lo

            # Above maximum
            invalid_above = dict(base)
            invalid_above[feat.name] = hi + 1.0
            with pytest.raises(ValidationError) as exc_hi:
                SpinachSensorRequest.model_validate(invalid_above)
            msg_hi = exc_hi.value.errors()[0]["msg"]
            assert "OUT_OF_BOUNDS" in msg_hi
            assert feat.name in msg_hi
            assert unit in msg_hi

    def test_all_tomato_features_bounds(self) -> None:
        schema = get_sensor_schema("tomato")
        base = {
            "air_temperature_C": 24.0,
            "relative_humidity_percent": 70.0,
            "water_temperature_C": 22.0,
            "ph": 6.3,
            "ec_mS_cm": 2.8,
            "nitrogen_mg_L": 150.0,
            "phosphorus_mg_L": 50.0,
            "potassium_mg_L": 250.0,
        }

        for feat in schema.features:
            lo, hi = feat.hard_bounds
            unit = feat.unit

            # Boundary values must succeed
            valid_lo = dict(base)
            valid_lo[feat.name] = lo
            assert (
                getattr(TomatoSensorRequest.model_validate(valid_lo), feat.name) == lo
            )

            valid_hi = dict(base)
            valid_hi[feat.name] = hi
            assert (
                getattr(TomatoSensorRequest.model_validate(valid_hi), feat.name) == hi
            )

            # Below minimum
            invalid_below = dict(base)
            invalid_below[feat.name] = lo - 1.0
            with pytest.raises(ValidationError) as exc_lo:
                TomatoSensorRequest.model_validate(invalid_below)
            msg_lo = exc_lo.value.errors()[0]["msg"]
            assert "OUT_OF_BOUNDS" in msg_lo
            assert feat.name in msg_lo
            assert unit in msg_lo

            # Above maximum
            invalid_above = dict(base)
            invalid_above[feat.name] = hi + 10.0
            with pytest.raises(ValidationError) as exc_hi:
                TomatoSensorRequest.model_validate(invalid_above)
            msg_hi = exc_hi.value.errors()[0]["msg"]
            assert "OUT_OF_BOUNDS" in msg_hi
            assert feat.name in msg_hi
            assert unit in msg_hi


# ==============================================================================
# Unknown-Field & Data-Leakage Rejection (extra='forbid')
# ==============================================================================


class TestUnknownFieldRejection:
    """Verify extra='forbid' strictly rejects unknown fields and data leakage columns."""

    def test_lettuce_rejects_leakage_and_unknown_fields(self) -> None:
        base = {
            "ph": 6.0,
            "ec": 1.8,
            "water_temp": 20.0,
            "humidity": 60.0,
            "air_temp": 22.0,
        }

        # Data leakage blacklisted fields
        blacklisted = ["co2", "tds", "lux_meter", "label", "label_next"]
        for field in blacklisted:
            payload = dict(base)
            payload[field] = 500.0
            with pytest.raises(ValidationError) as exc_info:
                LettuceSensorRequest.model_validate(payload)
            errors = exc_info.value.errors()
            assert any(
                e["type"] == "extra_forbidden" and e["loc"] == (field,) for e in errors
            )

    def test_spinach_rejects_leakage_and_unknown_fields(self) -> None:
        base = {
            "air_temperature_C": 20.0,
            "relative_humidity_percent": 60.0,
            "water_temperature_C": 19.0,
            "ph": 6.5,
            "ec_mS_cm": 2.1,
        }

        blacklisted = ["tds_ppm", "predicted_class", "timestamp", "lux"]
        for field in blacklisted:
            payload = dict(base)
            payload[field] = 120.0
            with pytest.raises(ValidationError) as exc_info:
                SpinachSensorRequest.model_validate(payload)
            errors = exc_info.value.errors()
            assert any(
                e["type"] == "extra_forbidden" and e["loc"] == (field,) for e in errors
            )

    def test_tomato_rejects_leakage_and_unknown_fields(self) -> None:
        base = {
            "air_temperature_C": 24.0,
            "relative_humidity_percent": 70.0,
            "water_temperature_C": 22.0,
            "ph": 6.3,
            "ec_mS_cm": 2.8,
            "nitrogen_mg_L": 150.0,
            "phosphorus_mg_L": 50.0,
            "potassium_mg_L": 250.0,
        }

        blacklisted = ["sample_id", "primary_fault", "leaf_temp_C", "target"]
        for field in blacklisted:
            payload = dict(base)
            payload[field] = 999.0
            with pytest.raises(ValidationError) as exc_info:
                TomatoSensorRequest.model_validate(payload)
            errors = exc_info.value.errors()
            assert any(
                e["type"] == "extra_forbidden" and e["loc"] == (field,) for e in errors
            )


# ==============================================================================
# Error-Envelope Structure, Status Codes, and Retryable Flags
# ==============================================================================


class TestErrorEnvelopeAndCodes:
    """Verify all typed exceptions conform to the unified error envelope."""

    ALL_ERROR_CLASSES: ClassVar[list[tuple[type[AppException], str, int, bool]]] = [
        (UnsupportedMediaError, "UNSUPPORTED_MEDIA", 415, False),
        (PayloadTooLargeError, "PAYLOAD_TOO_LARGE", 413, False),
        (InvalidInputError, "INVALID_INPUT", 422, False),
        (MissingFieldError, "MISSING_FIELD", 422, False),
        (SegmentationFailedError, "SEGMENTATION_FAILED", 422, True),
        (SchemaMismatchError, "SCHEMA_MISMATCH", 409, False),
        (ModelUnavailableError, "MODEL_UNAVAILABLE", 503, False),
        (DeviceUnavailableError, "DEVICE_UNAVAILABLE", 503, True),
        (PersistenceFailedError, "PERSISTENCE_FAILED", 503, False),
        (RateLimitExceededError, "RATE_LIMIT_EXCEEDED", 429, False),
    ]

    @pytest.mark.parametrize(
        "exc_cls,expected_code,expected_status,expected_retryable",
        ALL_ERROR_CLASSES,
    )
    def test_each_error_code_and_retryable_flag(
        self,
        exc_cls: type[AppException],
        expected_code: str,
        expected_status: int,
        expected_retryable: bool,
    ) -> None:
        """Every typed error class produces the standard envelope structure."""
        exc = exc_cls(message="Test error message", details={"field": "ph"})
        assert exc.status_code == expected_status
        assert exc.retryable is expected_retryable
        assert exc.code == expected_code

        envelope = exc.to_envelope()
        assert "error" in envelope
        err_obj = envelope["error"]
        assert err_obj["code"] == expected_code
        assert err_obj["message"] == "Test error message"
        assert err_obj["details"] == {"field": "ph"}
        assert err_obj["retryable"] is expected_retryable

    def test_retryable_strictly_true_only_for_segmentation_and_device(
        self,
    ) -> None:
        """retryable is strictly True only for SEGMENTATION_FAILED and DEVICE_UNAVAILABLE."""
        retryable_codes = {
            expected_code
            for _, expected_code, _, expected_retryable in self.ALL_ERROR_CLASSES
            if expected_retryable
        }
        assert retryable_codes == {"SEGMENTATION_FAILED", "DEVICE_UNAVAILABLE"}

    def test_fastapi_error_handlers_return_envelope(self) -> None:
        """FastAPI app exception handlers wrap typed exceptions into unified envelope."""
        app = FastAPI()
        register_error_handlers(app)

        @app.get("/trigger-unsupported")
        def _trigger_unsupported() -> None:
            raise UnsupportedMediaError("Only JPEG and PNG allowed.")

        @app.get("/trigger-device")
        def _trigger_device() -> None:
            raise DeviceUnavailableError("Modbus timeout on port 5000.")

        client = TestClient(app)

        res415 = client.get("/trigger-unsupported")
        assert res415.status_code == 415
        data415 = res415.json()
        assert data415["error"]["code"] == "UNSUPPORTED_MEDIA"
        assert data415["error"]["retryable"] is False

        res503 = client.get("/trigger-device")
        assert res503.status_code == 503
        data503 = res503.json()
        assert data503["error"]["code"] == "DEVICE_UNAVAILABLE"
        assert data503["error"]["retryable"] is True
