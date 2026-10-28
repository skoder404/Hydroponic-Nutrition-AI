"""Unit tests for RecommendationService with provider interface (T6-009)."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import Settings  # noqa: E402
from backend.app.schemas.prediction import PredictionResponse  # noqa: E402
from backend.app.schemas.recommendation import RecommendationRead  # noqa: E402
from backend.app.services.recommendation_service import (  # noqa: E402
    GeminiProviderSlot,
    RecommendationProvider,
    RecommendationService,
    get_recommendation_service,
    reset_recommendation_service,
)
from ml.recommendations.rule_provider import (  # noqa: E402
    RuleProvider,
    get_rule_provider,
)


class DummyCustomProvider:
    """Custom provider for testing registration and swapping."""

    provider_name: str = "dummy"

    def recommend(self, **kwargs) -> RecommendationRead:
        return RecommendationRead(
            id=uuid4(),
            record_id=kwargs.get("record_id") or uuid4(),
            source="rules",
            rules_version="dummy_v1.0.0",
            content={"summary": "Dummy custom advice", "actions": []},
            evidence={"dummy": True},
        )

    def recommend_from_prediction(
        self, prediction: PredictionResponse, **kwargs
    ) -> RecommendationRead:
        return self.recommend(**kwargs)


@pytest.fixture(autouse=True)
def cleanup_service():
    """Reset recommendation service singleton before and after each test."""
    reset_recommendation_service()
    yield
    reset_recommendation_service()


def test_provider_protocol_conformance() -> None:
    """Verify that RuleProvider, GeminiProviderSlot, and custom providers conform to the protocol."""
    assert isinstance(get_rule_provider(), RecommendationProvider)
    assert isinstance(GeminiProviderSlot(), RecommendationProvider)
    assert isinstance(DummyCustomProvider(), RecommendationProvider)


def test_service_initialization_defaults() -> None:
    """Verify service initializes with RuleProvider default when no Gemini is configured."""
    settings = Settings(gemini_api_key=None, recommendation_provider="rules")
    service = RecommendationService(settings=settings)

    assert service.active_provider_name == "rules"
    assert isinstance(service.active_provider, RuleProvider)


def test_service_works_with_no_gemini_configured() -> None:
    """Acceptance criterion 1: Service works cleanly with no Gemini API key configured."""
    settings = Settings(gemini_api_key=None, recommendation_provider="rules")
    service = RecommendationService(settings=settings)

    rec = service.recommend(
        crop="lettuce",
        predicted_class="NitrogenDeficiency",
        modality="sensor",
        confidence=0.85,
    )
    assert rec is not None
    assert rec.source == "rules"
    assert rec.rules_version == "v1.0.0"
    assert "actions" in rec.content
    assert "checks" in rec.content


def test_output_includes_source_rules() -> None:
    """Acceptance criterion 2: Output includes source: 'rules' and rules_version."""
    service = RecommendationService()

    rec = service.recommend(
        crop="spinach",
        predicted_class="Spinach_Healthy",
        modality="image",
    )
    assert rec.source == "rules"
    assert rec.rules_version == "v1.0.0"
    assert rec.evidence["status"] == "advisory, unreviewed"


def test_provider_swapped_by_config() -> None:
    """Acceptance criterion 3: Provider can be swapped via application configuration."""
    # Configure provider='gemini' with no key -> should set active_provider_name to 'gemini'
    # and delegate to GeminiProviderSlot (which falls back to rules)
    settings_gemini = Settings(gemini_api_key=None, recommendation_provider="gemini")
    service = RecommendationService(settings=settings_gemini)

    assert service.active_provider_name == "gemini"
    assert isinstance(service.active_provider, GeminiProviderSlot)

    rec = service.recommend(
        crop="lettuce",
        predicted_class="Healthier",
        modality="sensor",
    )
    assert rec.source == "rules"  # Slot cleanly delegates to rules


def test_provider_swapped_at_runtime() -> None:
    """Verify active provider can be switched via set_provider."""
    service = RecommendationService()
    assert service.active_provider_name == "rules"

    service.set_provider("gemini")
    assert service.active_provider_name == "gemini"

    service.set_provider("rules")
    assert service.active_provider_name == "rules"

    # Register and swap to custom provider
    dummy = DummyCustomProvider()
    service.register_provider("dummy", dummy)
    service.set_provider("dummy")
    assert service.active_provider_name == "dummy"

    rec = service.recommend(crop="lettuce", predicted_class="Healthier")
    assert rec.rules_version == "dummy_v1.0.0"


def test_set_provider_unknown_raises_error() -> None:
    """Verify setting an unknown provider name raises ValueError with available list."""
    service = RecommendationService()
    with pytest.raises(ValueError, match="Unknown recommendation provider 'unknown'"):
        service.set_provider("unknown")


def test_set_provider_invalid_type_raises_error() -> None:
    """Verify passing an object that does not implement RecommendationProvider raises TypeError."""
    service = RecommendationService()
    with pytest.raises(TypeError, match="Expected RecommendationProvider"):
        service.set_provider(12345)  # type: ignore[arg-type]


def test_recommend_from_prediction() -> None:
    """Verify recommend_from_prediction delegates correctly with PredictionResponse."""
    service = RecommendationService()

    pred = PredictionResponse(
        crop="spinach",
        predicted_class="Spinach_Healthy",
        display_label="Healthy",
        modality="image",
        confidence=0.92,
        confidence_semantics="multinomial_probability",
        uncertain=False,
        model_id="spinach_rf_v1.0.0",
        model_version="v1.0.0",
        timestamp="2026-10-28T16:00:00Z",
        provenance="live",
    )
    rec = service.recommend_from_prediction(pred)
    assert rec.source == "rules"
    assert rec.evidence["crop"] == "spinach"
    assert rec.evidence["predicted_class"] == "Spinach_Healthy"


def test_singleton_helpers() -> None:
    """Verify get_recommendation_service and reset_recommendation_service."""
    s1 = get_recommendation_service()
    s2 = get_recommendation_service()
    assert s1 is s2

    reset_recommendation_service()
    s3 = get_recommendation_service()
    assert s3 is not s1
