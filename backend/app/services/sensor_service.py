"""SensorService for validating telemetry, ordering features, and executing inference (T6-004).

Conforms to TECH_RULES and T6-004 Acceptance Criteria:
- Validates sensor payloads per crop schema; rejects missing/null values with MISSING_FIELD.
- Rejects unknown fields with INVALID_INPUT (extra='forbid').
- Maps out-of-order JSON keys into the exact feature order defined by the model schema.
- Applies the saved pipeline scaler and classifier.
- Computes prediction confidence and uncertainty flag based on metadata thresholds.
- Attaches the 'experimental' boolean flag from model metadata (True for tomato).
- Attaches operational warnings and agronomic limitations ('rule-derived labels').
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import ValidationError

from backend.app.core.errors import (
    InvalidInputError,
    MissingFieldError,
    ModelUnavailableError,
    SchemaMismatchError,
)
from backend.app.schemas.prediction import PredictionResponse
from backend.app.schemas.sensors import (
    CropSensorRequest,
    get_sensor_request_model,
    validate_sensor_request,
)
from backend.app.services.model_service import (
    ModelRegistryService,
    get_model_registry_service,
)
from ml.sensor_models.schemas import get_sensor_schema

logger = logging.getLogger(__name__)

STANDARD_LIMITATIONS: list[str] = [
    "rule-derived labels: Predictions reflect statistical correlation with project rule taxonomy on a preserved split, not a clinical veterinary diagnosis.",
    "No multimodal fusion without verified paired data (TECH_RULES §15, Overview §19).",
]


class SensorService:
    """Service managing sensor observation validation and model inference execution."""

    def __init__(self, model_registry_service: ModelRegistryService | None = None) -> None:
        self.registry_service = model_registry_service or get_model_registry_service()

    def predict(
        self,
        crop: str,
        readings: dict[str, Any] | CropSensorRequest,
        provenance: Literal["live", "manual", "imported"] = "manual",
    ) -> PredictionResponse:
        """Validate sensor payload, align feature order, execute inference, and return PredictionResponse.

        Args:
            crop: Crop family ('lettuce', 'spinach', 'tomato', 'tomato_live').
            readings: Dictionary of sensor readings or validated CropSensorRequest instance.
            provenance: Telemetry provenance ('live', 'manual', 'imported').

        Returns:
            PredictionResponse with predicted class, display label, confidence, uncertainty,
            experimental flag, warnings, and limitations.

        Raises:
            MissingFieldError: If a required sensor reading is omitted or null.
            InvalidInputError: If payload has unknown keys, invalid types, or out-of-bounds values.
            ModelUnavailableError: If promoted model artifact is missing or corrupted.
            SchemaMismatchError: If model schema does not align with runtime expected schema.
        """
        clean_crop = crop.strip().lower()

        # 1. Validate payload using per-crop schema (T3-003)
        validated_request: CropSensorRequest
        if isinstance(readings, CropSensorRequest):
            validated_request = readings
        elif isinstance(readings, dict):
            validated_request = self._validate_payload_dict(clean_crop, readings)
        else:
            raise InvalidInputError(
                message=f"Payload for '{clean_crop}' must be a dictionary or CropSensorRequest, got {type(readings).__name__}.",
                details={"crop": clean_crop, "code": "INVALID_INPUT"},
            )

        feature_dict = validated_request.to_feature_dict()

        # 2. Retrieve promoted active model and metadata from registry (T6-002)
        model_obj, metadata = self.registry_service.get_model(clean_crop, "sensor")

        ordered_features = metadata.ordered_features
        if not ordered_features:
            raise SchemaMismatchError(
                message=f"Model metadata for '{clean_crop}' does not define ordered_features.",
                details={"crop": clean_crop, "code": "SCHEMA_MISMATCH"},
            )

        # 3. Order features per schema: guarantees arbitrary JSON key order maps to correct order
        # Acceptance Criterion 1: Out-of-order JSON keys still map to the correct feature order
        try:
            ordered_row = {feat: float(feature_dict[feat]) for feat in ordered_features}
        except KeyError as err:
            missing_feat = str(err).strip("'")
            raise MissingFieldError(
                message=f"Missing required sensor field '{missing_feat}' for crop '{clean_crop}'.",
                details={"crop": clean_crop, "field": missing_feat, "code": "MISSING_FIELD"},
            ) from err

        # Pass as DataFrame with column names to preserve feature names for scikit-learn transformers
        X = pd.DataFrame([ordered_row], columns=ordered_features)

        # 4. Predict using the pipeline (applies the saved scaler automatically)
        predicted_class: str
        confidence: float | None = None
        confidence_semantics: str | None = None

        if hasattr(model_obj, "predict_proba"):
            probabilities = model_obj.predict_proba(X)[0]
            classes = list(getattr(model_obj, "classes_", metadata.classes))
            max_idx = int(np.argmax(probabilities))
            confidence = round(float(probabilities[max_idx]), 4)
            predicted_class = str(classes[max_idx])
            confidence_semantics = "multinomial_probability"
        else:
            predictions = model_obj.predict(X)
            predicted_class = str(predictions[0])
            confidence = None
            confidence_semantics = None

        # 5. Compute uncertainty from metadata thresholds
        uncertainty_threshold = 0.40
        if isinstance(metadata.confidence_settings, dict):
            uncertainty_threshold = float(
                metadata.confidence_settings.get("uncertainty_threshold", 0.40)
            )

        uncertain = (
            bool(confidence < uncertainty_threshold)
            if confidence is not None
            else False
        )

        # 6. Attach experimental flag from metadata
        # Acceptance Criterion 3: Tomato result carries experimental=true
        experimental = bool(metadata.experimental)

        # 7. Map display label
        display_label = metadata.display_mapping.get(predicted_class, predicted_class)

        # 8. Check operational agronomic warnings
        warnings = self._collect_agronomic_warnings(clean_crop, ordered_row)

        logger.info(
            "Sensor prediction completed: crop=%s class=%s conf=%s uncertain=%s experimental=%s",
            clean_crop,
            predicted_class,
            confidence,
            uncertain,
            experimental,
        )

        return PredictionResponse(
            crop=clean_crop,
            modality="sensor",
            predicted_class=predicted_class,
            display_label=display_label,
            confidence=confidence,
            confidence_semantics=confidence_semantics,
            uncertain=uncertain,
            experimental=experimental,
            model_id=metadata.model_id,
            model_version=f"v{metadata.artifact_version}",
            timestamp=datetime.now(timezone.utc),
            provenance=provenance,
            warnings=warnings,
            limitations=list(STANDARD_LIMITATIONS),
        )

    def _validate_payload_dict(
        self,
        crop: str,
        payload: dict[str, Any],
    ) -> CropSensorRequest:
        """Validate raw dictionary against crop request model, mapping errors to typed exceptions."""
        try:
            return validate_sensor_request(crop, payload)
        except ValidationError as err:
            errors = err.errors()
            for e in errors:
                msg = e.get("msg", "")
                loc = ".".join(str(x) for x in e.get("loc", []))
                err_type = e.get("type", "")

                # Missing field detection (Acceptance Criterion 2)
                if err_type == "missing" or "MISSING_FIELD" in msg:
                    field_name = loc or "unknown"
                    raise MissingFieldError(
                        message=f"Missing required sensor field '{field_name}' for crop '{crop}'.",
                        details={"crop": crop, "field": field_name, "code": "MISSING_FIELD"},
                    ) from err

                # Extra forbidden fields (extra='forbid')
                if err_type == "extra_forbidden":
                    field_name = loc or "unknown"
                    raise InvalidInputError(
                        message=f"Unknown field '{field_name}' is forbidden for crop '{crop}'.",
                        details={"crop": crop, "field": field_name, "code": "INVALID_INPUT"},
                    ) from err

                # Out of bounds or invalid numeric type
                if "OUT_OF_BOUNDS" in msg or "INVALID_TYPE" in msg or "value_error" in err_type:
                    raise InvalidInputError(
                        message=msg,
                        details={"crop": crop, "field": loc, "code": "INVALID_INPUT"},
                    ) from err

            raise InvalidInputError(
                message="Validation failed for sensor input parameters.",
                details={"crop": crop, "validation_errors": errors, "code": "INVALID_INPUT"},
            ) from err
        except ValueError as err:
            err_str = str(err)
            if "Unknown crop" in err_str:
                raise InvalidInputError(
                    message=err_str,
                    details={"crop": crop, "code": "INVALID_INPUT"},
                ) from err
            if "MISSING_FIELD" in err_str:
                raise MissingFieldError(
                    message=err_str,
                    details={"crop": crop, "code": "MISSING_FIELD"},
                ) from err
            raise InvalidInputError(
                message=err_str,
                details={"crop": crop, "code": "INVALID_INPUT"},
            ) from err

    def _collect_agronomic_warnings(
        self,
        crop: str,
        readings: dict[str, float],
    ) -> list[str]:
        """Agronomic parameter bounds are rendered directly in the sensor telemetry report below.
        
        Per user directive, omit top-level redundant sensor notices.
        """
        return []


# Singleton access
_sensor_service_instance: SensorService | None = None


def get_sensor_service(
    model_registry_service: ModelRegistryService | None = None,
) -> SensorService:
    """Return the application singleton SensorService instance."""
    global _sensor_service_instance
    if _sensor_service_instance is None or model_registry_service is not None:
        _sensor_service_instance = SensorService(
            model_registry_service=model_registry_service
        )
    return _sensor_service_instance


def reset_sensor_service() -> None:
    """Reset the singleton instance (primarily for test fixtures)."""
    global _sensor_service_instance
    _sensor_service_instance = None
