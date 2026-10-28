"""Unit tests for RuleProvider and recommendation rules matrix (T6-008).

Acceptance Criteria:
- Every class of every crop has a rule or an explicit 'insufficient information'.
- No rule text contains dosage numbers.
- Output cites which prediction/rule produced it.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from backend.app.schemas.prediction import PredictionResponse
from backend.app.schemas.recommendation import RecommendationRead
from ml.labels import (
    LETTUCE_IMAGE_CLASSES,
    LETTUCE_SENSOR_CLASSES,
    SPINACH_IMAGE_CLASSES,
    SPINACH_SENSOR_CLASSES,
    TOMATO_IMAGE_CLASSES,
    TOMATO_SENSOR_CLASSES,
)
from ml.recommendations.rule_provider import (
    RuleProvider,
    get_rule_provider,
    reset_rule_provider,
)


@pytest.fixture(autouse=True)
def clean_rule_provider():
    """Reset singleton before and after each test."""
    reset_rule_provider()
    yield
    reset_rule_provider()


def test_criterion_1_every_class_of_every_crop_covered():
    """Acceptance Criterion 1: Every class of every crop has a rule or explicit insufficient info."""
    provider = get_rule_provider()

    crop_classes = {
        "lettuce": (LETTUCE_IMAGE_CLASSES, LETTUCE_SENSOR_CLASSES),
        "spinach": (SPINACH_IMAGE_CLASSES, SPINACH_SENSOR_CLASSES),
        "tomato": (TOMATO_IMAGE_CLASSES, TOMATO_SENSOR_CLASSES),
    }

    for crop, (image_classes, sensor_classes) in crop_classes.items():
        # Test all image classes
        for cls in image_classes:
            rec = provider.recommend(crop=crop, predicted_class=cls, modality="image")
            assert isinstance(rec, RecommendationRead)
            assert rec.source == "rules"
            assert rec.rules_version == "v1.0.0"
            assert "insufficient information" not in rec.content["summary"].lower(), (
                f"Missing specific rule for {crop}/image/{cls}"
            )
            assert len(rec.content["checks"]) > 0
            assert len(rec.content["actions"]) > 0

        # Test all sensor classes
        for cls in sensor_classes:
            rec = provider.recommend(crop=crop, predicted_class=cls, modality="sensor")
            assert isinstance(rec, RecommendationRead)
            assert rec.source == "rules"
            assert rec.rules_version == "v1.0.0"
            assert "insufficient information" not in rec.content["summary"].lower(), (
                f"Missing specific rule for {crop}/sensor/{cls}"
            )
            assert len(rec.content["checks"]) > 0
            assert len(rec.content["actions"]) > 0


def test_criterion_1_unsupported_returns_insufficient_information():
    """Verify unknown crop or unsupported class returns explicit 'insufficient information'."""
    provider = get_rule_provider()

    # Unsupported class on known crop
    rec_bad_class = provider.recommend("lettuce", "AlienDeficiency", "sensor")
    assert "insufficient information" in rec_bad_class.content["summary"].lower()
    assert rec_bad_class.evidence["rule_citation"] == "insufficient_information"

    # Unsupported crop
    rec_bad_crop = provider.recommend("kale", "Normal", "sensor")
    assert "insufficient information" in rec_bad_crop.content["summary"].lower()
    assert rec_bad_crop.evidence["rule_citation"] == "insufficient_information"


def test_criterion_2_no_rule_text_contains_dosage_numbers():
    """Acceptance Criterion 2: No rule text contains dosage numbers."""
    rules_dir = Path("ml/recommendations/rules")
    yaml_files = list(rules_dir.glob("*.v1.yaml"))
    assert len(yaml_files) >= 3

    digit_pattern = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ml|mg|g|ppm|%|ms|l|liters?|drops?|oz)\b", re.IGNORECASE)

    for yf in yaml_files:
        data = yaml.safe_load(yf.read_text(encoding="utf-8"))
        rules = data.get("rules", {})
        for modality, class_dict in rules.items():
            for cls_name, rule in class_dict.items():
                for field in ("actions", "checks", "summary", "low_confidence_action"):
                    raw_val = rule.get(field, [])
                    items = raw_val if isinstance(raw_val, list) else [raw_val]
                    for item in items:
                        # Test for dosage units with numbers
                        dosage_matches = digit_pattern.findall(str(item))
                        assert not dosage_matches, (
                            f"Dosage number found in {yf.name} -> {modality}/{cls_name} -> {field}: {dosage_matches}"
                        )
                        # Also test for any raw numbers in actions
                        if field in ("actions", "low_confidence_action"):
                            raw_digits = re.findall(r"\d+", str(item))
                            assert not raw_digits, (
                                f"Numeric amount found in action of {yf.name} -> {modality}/{cls_name}: {raw_digits}"
                            )


def test_criterion_3_output_cites_prediction_and_rule():
    """Acceptance Criterion 3: Output cites which prediction/rule produced it."""
    provider = get_rule_provider()

    rec = provider.recommend(
        crop="lettuce",
        predicted_class="EC_High",
        modality="sensor",
        confidence=0.88,
        uncertain=False,
    )

    # Citation in content
    assert rec.content["rule_citation"] == "rules/lettuce.v1.yaml#sensor/EC_High"

    # Citation in evidence
    assert rec.evidence["rule_citation"] == "rules/lettuce.v1.yaml#sensor/EC_High"
    assert rec.evidence["crop"] == "lettuce"
    assert rec.evidence["modality"] == "sensor"
    assert rec.evidence["predicted_class"] == "EC_High"
    assert rec.evidence["confidence"] == 0.88
    assert rec.evidence["uncertain"] is False
    assert rec.evidence["status"] == "advisory, unreviewed"


def test_low_confidence_triggers_remeasure_advice():
    """Verify that low confidence or uncertain=True yields re-measure/inspect advice."""
    provider = get_rule_provider()

    rec_low = provider.recommend(
        crop="lettuce",
        predicted_class="EC_High",
        modality="sensor",
        confidence=0.25,  # Low confidence
        uncertain=True,
    )

    assert "uncertain" in rec_low.content["summary"].lower()
    assert rec_low.evidence["is_low_confidence"] is True
    # Actions emphasize re-measurement
    combined_actions = " ".join(rec_low.content["actions"]).lower()
    assert "re-measure" in combined_actions or "verify" in combined_actions


def test_recommend_from_prediction_object():
    """Verify recommend_from_prediction seamlessly handles PredictionResponse."""
    provider = get_rule_provider()

    prediction = PredictionResponse(
        crop="spinach",
        modality="sensor",
        predicted_class="Humidity_High_Mildew_Risk",
        display_label="unhealthy/Humidity_High_Mildew_Risk",
        confidence=0.92,
        confidence_semantics="multinomial_probability",
        uncertain=False,
        experimental=False,
        model_id="spinach_sensor_dt_v1.0.0",
        model_version="v1.0.0",
        timestamp=datetime.now(timezone.utc),
        provenance="live",
        warnings=[],
        limitations=["rule-derived labels"],
    )

    rec = provider.recommend_from_prediction(prediction)
    assert rec.evidence["crop"] == "spinach"
    assert rec.evidence["predicted_class"] == "Humidity_High_Mildew_Risk"
    assert rec.evidence["confidence"] == 0.92
    assert "rules/spinach.v1.yaml#sensor/Humidity_High_Mildew_Risk" in rec.evidence["rule_citation"]
