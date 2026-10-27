"""Verification script for T6-001: Pydantic Request and Response Schemas.

Verifies acceptance criteria:
1. Schemas validate the example JSON in TECH_RULES.
2. confidence_semantics is mandatory whenever confidence is present.
3. OpenAPI renders without errors.
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime, timezone

from pydantic import ValidationError

from backend.app.main import create_app
from backend.app.schemas import (
    ErrorResponse,
    LeafPrediction,
    PredictionResponse,
    RecordCreate,
)


def verify_tech_rules_json_examples() -> None:
    # 1. Prediction Response JSON matching TECH_RULES line 174
    example_sensor_prediction = {
        "crop": "lettuce",
        "modality": "sensor",
        "predicted_class": "NitrogenDeficiency",
        "display_label": "Nitrogen Deficiency",
        "confidence": 0.892,
        "confidence_semantics": "multinomial_probability",
        "uncertain": False,
        "experimental": False,
        "model_id": "lettuce_sensor_rf",
        "model_version": "v1.0.0",
        "timestamp": "2026-10-27T18:00:00Z",
        "provenance": "live",
        "warnings": ["Low reservoir temperature"],
        "limitations": ["Calibrated for NFT hydro-channels"],
    }
    pred = PredictionResponse.model_validate(example_sensor_prediction)
    assert pred.crop == "lettuce"
    assert pred.confidence == 0.892

    # Image prediction with leaves, rejected_leaves, and summary
    example_image_prediction = {
        **example_sensor_prediction,
        "modality": "image",
        "input_mode": "plant",
        "leaves": [
            {
                "leaf_id": "leaf_001",
                "bbox": [50, 40, 200, 220],
                "mask_ref": "artifacts/masks/leaf_001.png",
                "predicted_class": "NitrogenDeficiency",
                "display_label": "Nitrogen Deficiency",
                "confidence": 0.915,
                "confidence_semantics": "tree_agreement_ratio",
                "uncertain": False,
            }
        ],
        "rejected_leaves": [
            {
                "leaf_id": "leaf_002",
                "reason": "Mask is too small (0.0005 < min_area_ratio 0.0100)",
                "verdict": "reject",
            }
        ],
        "summary": {
            "leaf_count": 1,
            "class_counts": {"NitrogenDeficiency": 1},
        },
    }
    img_pred = PredictionResponse.model_validate(example_image_prediction)
    assert img_pred.summary is not None
    assert img_pred.summary.leaf_count == 1
    assert len(img_pred.leaves) == 1
    assert len(img_pred.rejected_leaves) == 1

    # 2. Error envelope matching TECH_RULES line 180
    example_error = {
        "error": {
            "code": "SEGMENTATION_FAILED",
            "message": "Candidate leaf mask failed boundary check",
            "details": {
                "rejected_leaves": [{"leaf_id": "leaf_001", "reason": "border touch"}]
            },
            "retryable": True,
        }
    }
    err = ErrorResponse.model_validate(example_error)
    assert err.error.code == "SEGMENTATION_FAILED"
    assert err.error.retryable is True

    print("TECH_RULES example JSON validation: VERIFIED.")


def verify_mandatory_confidence_semantics() -> None:
    # Present confidence without semantics MUST raise ValidationError
    try:
        PredictionResponse(
            crop="tomato",
            modality="sensor",
            predicted_class="Healthy",
            display_label="Healthy",
            confidence=0.88,
            confidence_semantics=None,  # Invalid!
            uncertain=False,
            model_id="m1",
            model_version="v1",
            timestamp=datetime.now(timezone.utc),
            provenance="live",
        )
        raise AssertionError(
            "Expected ValidationError when confidence_semantics is missing"
        )
    except ValidationError:
        pass

    # Empty string confidence_semantics MUST also fail
    try:
        PredictionResponse(
            crop="tomato",
            modality="sensor",
            predicted_class="Healthy",
            display_label="Healthy",
            confidence=0.88,
            confidence_semantics="",
            uncertain=False,
            model_id="m1",
            model_version="v1",
            timestamp=datetime.now(timezone.utc),
            provenance="live",
        )
        raise AssertionError(
            "Expected ValidationError when confidence_semantics is empty string"
        )
    except ValidationError:
        pass

    # LeafPrediction also enforces mandatory confidence_semantics
    try:
        LeafPrediction(
            leaf_id="l1",
            bbox=[0, 0, 10, 10],
            predicted_class="Healthy",
            display_label="Healthy",
            confidence=0.9,
            confidence_semantics=None,
            uncertain=False,
        )
        raise AssertionError(
            "Expected ValidationError on LeafPrediction without confidence_semantics"
        )
    except ValidationError:
        pass

    print("Mandatory confidence_semantics invariant: VERIFIED.")


def verify_extra_forbid_on_requests() -> None:
    try:
        RecordCreate(
            crop="spinach",
            mode="sensor",
            source="manual",
            observed_at=datetime.now(timezone.utc),
            observed_tz="UTC",
            unauthorized_field="disallowed",
        )
        raise AssertionError(
            "Expected ValidationError when extra field provided to RecordCreate"
        )
    except ValidationError:
        pass

    print("Request extra='forbid' validation: VERIFIED.")


def verify_openapi_rendering() -> None:
    app = create_app()
    openapi_doc = app.openapi()
    assert openapi_doc.get("openapi") is not None
    assert "paths" in openapi_doc
    print("FastAPI OpenAPI schema generation: VERIFIED.")


def main() -> int:
    print("=== Verifying T6-001: Pydantic Request/Response Schemas ===")
    verify_tech_rules_json_examples()
    verify_mandatory_confidence_semantics()
    verify_extra_forbid_on_requests()
    verify_openapi_rendering()
    print("=== All T6-001 Acceptance Criteria PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
