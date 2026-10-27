"""Unit tests for Pydantic request and response schemas (T6-001)."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.main import create_app
from backend.app.schemas import (
    ErrorResponse,
    LeafPrediction,
    PredictionResponse,
    PredictionSummary,
    RecommendationRead,
    RecordCreate,
    RecordRead,
    RejectedLeaf,
)


def test_prediction_response_valid():
    payload = {
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
        "warnings": ["Low water temp"],
        "limitations": ["Greenhouse only"],
    }
    resp = PredictionResponse.model_validate(payload)
    assert resp.crop == "lettuce"
    assert resp.confidence == 0.892
    assert resp.confidence_semantics == "multinomial_probability"
    assert resp.warnings == ["Low water temp"]


def test_prediction_response_mandatory_confidence_semantics():
    # If confidence is present, confidence_semantics MUST be provided
    payload = {
        "crop": "lettuce",
        "modality": "sensor",
        "predicted_class": "NitrogenDeficiency",
        "display_label": "Nitrogen Deficiency",
        "confidence": 0.85,
        # Missing confidence_semantics
        "uncertain": False,
        "model_id": "m1",
        "model_version": "v1",
        "timestamp": datetime.now(timezone.utc),
        "provenance": "live",
    }
    with pytest.raises(ValidationError, match="`confidence_semantics` is mandatory"):
        PredictionResponse.model_validate(payload)

    # Empty string confidence_semantics should also be rejected
    payload["confidence_semantics"] = "   "
    with pytest.raises(ValidationError, match="`confidence_semantics` is mandatory"):
        PredictionResponse.model_validate(payload)


def test_prediction_response_null_confidence_allows_null_semantics():
    # If confidence is None (e.g. uncalibrated rule fallback), confidence_semantics should be None
    payload = {
        "crop": "lettuce",
        "modality": "sensor",
        "predicted_class": "Healthy",
        "display_label": "Healthy",
        "confidence": None,
        "confidence_semantics": None,
        "uncertain": True,
        "model_id": "rule_engine",
        "model_version": "v1.0.0",
        "timestamp": datetime.now(timezone.utc),
        "provenance": "manual",
    }
    resp = PredictionResponse.model_validate(payload)
    assert resp.confidence is None
    assert resp.confidence_semantics is None


def test_prediction_response_null_confidence_with_semantics_rejected():
    payload = {
        "crop": "lettuce",
        "modality": "sensor",
        "predicted_class": "Healthy",
        "display_label": "Healthy",
        "confidence": None,
        "confidence_semantics": "multinomial_probability",
        "uncertain": True,
        "model_id": "m1",
        "model_version": "v1",
        "timestamp": datetime.now(timezone.utc),
        "provenance": "manual",
    }
    with pytest.raises(
        ValidationError, match="cannot be specified when `confidence` is null"
    ):
        PredictionResponse.model_validate(payload)


def test_leaf_prediction_and_summary_invariants():
    leaf = LeafPrediction(
        leaf_id="leaf_01",
        bbox=[10, 20, 100, 150],
        mask_ref="masks/leaf_01.png",
        predicted_class="PhosphorusDeficiency",
        display_label="Phosphorus Deficiency",
        confidence=0.91,
        confidence_semantics="random_forest_tree_agreement",
        uncertain=False,
    )
    assert leaf.leaf_id == "leaf_01"
    assert leaf.confidence == 0.91

    rejected = RejectedLeaf(
        leaf_id="leaf_02",
        reason="Mask is too small (0.0002 < min_area_ratio 0.0100)",
        verdict="reject",
    )
    assert rejected.verdict == "reject"

    summary = PredictionSummary(
        leaf_count=1,
        class_counts={"PhosphorusDeficiency": 1},
    )
    assert summary.leaf_count == 1

    # Attempting to add an averaged confidence to summary MUST fail (extra='forbid')
    with pytest.raises(ValidationError):
        PredictionSummary(
            leaf_count=1,
            class_counts={"PhosphorusDeficiency": 1},
            averaged_confidence=0.91,  # type: ignore
        )


def test_error_envelope_schema():
    envelope_data = {
        "error": {
            "code": "SEGMENTATION_FAILED",
            "message": "No valid leaf mask found",
            "details": {"rejected_leaves": [{"leaf_id": "leaf_0", "reason": "empty"}]},
            "retryable": True,
        }
    }
    err = ErrorResponse.model_validate(envelope_data)
    assert err.error.code == "SEGMENTATION_FAILED"
    assert err.error.retryable is True
    assert err.error.details["rejected_leaves"][0]["leaf_id"] == "leaf_0"


def test_record_create_extra_forbid():
    valid_data = {
        "crop": "lettuce",
        "mode": "sensor",
        "source": "live",
        "observed_at": datetime.now(timezone.utc),
        "observed_tz": "UTC",
        "notes": "Testing record",
        "inputs": {"ph": 6.0, "ec": 1.5},
    }
    rec = RecordCreate.model_validate(valid_data)
    assert rec.crop == "lettuce"

    # Reject unexpected extra fields
    invalid_data = dict(valid_data)
    invalid_data["unknown_extra_field"] = "malicious_or_unexpected"
    with pytest.raises(ValidationError):
        RecordCreate.model_validate(invalid_data)


def test_record_read_and_recommendation_read():
    rec_id = uuid4()
    recommendation = RecommendationRead(
        record_id=rec_id,
        source="rules",
        rules_version="v1.0.0",
        content={"action": "Add pH down"},
        evidence={"ph": 7.8},
    )

    pred = PredictionResponse(
        crop="lettuce",
        modality="sensor",
        predicted_class="Healthy",
        display_label="Healthy",
        confidence=0.95,
        confidence_semantics="calibrated_score",
        uncertain=False,
        model_id="sensor_model_v1",
        model_version="v1.0.0",
        timestamp=datetime.now(timezone.utc),
        provenance="live",
    )

    record = RecordRead(
        id=rec_id,
        crop="lettuce",
        mode="sensor",
        observed_at=datetime.now(timezone.utc),
        observed_tz="Asia/Kolkata",
        source="live",
        predictions=[pred],
        recommendation=recommendation,
    )
    assert record.id == rec_id
    assert len(record.predictions) == 1
    assert record.recommendation is not None
    assert record.recommendation.source == "rules"


def test_openapi_schema_generation():
    """Verify OpenAPI generates without schema errors when schemas are mounted."""
    app = create_app()
    openapi = app.openapi()
    assert "openapi" in openapi
    assert "info" in openapi
    assert "paths" in openapi
