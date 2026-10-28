"""RecommendationService orchestrating agronomic recommendations across selectable providers (T6-009).

Conforms to TECH_RULES and PRD Section 20:
- Defines `RecommendationProvider` protocol.
- Wires `RuleProvider` as the default recommendation provider.
- Leaves a `GeminiProviderSlot` for Phase 10 / T10-001 narrative LLM recommendations.
- Guarantees service works with no Gemini configured (falls back cleanly).
- Guarantees output includes `source` ('rules' or 'gemini') and `rules_version`.
- Allows provider swapping by configuration (Settings.recommendation_provider) or runtime injection.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import UUID

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.prediction import PredictionResponse
from backend.app.schemas.recommendation import RecommendationRead
from ml.recommendations.rule_provider import get_rule_provider

logger = logging.getLogger(__name__)


@runtime_checkable
class RecommendationProvider(Protocol):
    """Protocol for agronomic recommendation providers (T6-009)."""

    def recommend(
        self,
        crop: str,
        predicted_class: str,
        modality: Literal["sensor", "image", "image_features"] = "sensor",
        confidence: float | None = None,
        uncertain: bool = False,
        sensor_readings: dict[str, Any] | None = None,
        record_id: UUID | None = None,
    ) -> RecommendationRead:
        """Generate a recommendation given crop prediction details and telemetry."""
        ...

    def recommend_from_prediction(
        self,
        prediction: PredictionResponse,
        sensor_readings: dict[str, Any] | None = None,
        record_id: UUID | None = None,
    ) -> RecommendationRead:
        """Generate a recommendation directly from a validated PredictionResponse."""
        ...


class GeminiProviderSlot:
    """Placeholder slot for GeminiRecommendationProvider (implemented in Phase 10 / T10-001).

    If invoked before Phase 10 or when GEMINI_API_KEY is not configured,
    it falls back cleanly to the deterministic RuleProvider.
    """

    provider_name: str = "gemini"

    def __init__(
        self,
        fallback_provider: RecommendationProvider | None = None,
        api_key_configured: bool = False,
    ) -> None:
        self._fallback_provider = fallback_provider or get_rule_provider()
        self.api_key_configured = api_key_configured

    def recommend(
        self,
        crop: str,
        predicted_class: str,
        modality: Literal["sensor", "image", "image_features"] = "sensor",
        confidence: float | None = None,
        uncertain: bool = False,
        sensor_readings: dict[str, Any] | None = None,
        record_id: UUID | None = None,
    ) -> RecommendationRead:
        logger.info(
            "GeminiProviderSlot invoked (Phase 10 pending, api_key_configured=%s); "
            "delegating to deterministic RuleProvider.",
            self.api_key_configured,
        )
        return self._fallback_provider.recommend(
            crop=crop,
            predicted_class=predicted_class,
            modality=modality,
            confidence=confidence,
            uncertain=uncertain,
            sensor_readings=sensor_readings,
            record_id=record_id,
        )

    def recommend_from_prediction(
        self,
        prediction: PredictionResponse,
        sensor_readings: dict[str, Any] | None = None,
        record_id: UUID | None = None,
    ) -> RecommendationRead:
        return self.recommend(
            crop=prediction.crop,
            predicted_class=prediction.predicted_class,
            modality=prediction.modality,
            confidence=prediction.confidence,
            uncertain=prediction.uncertain,
            sensor_readings=sensor_readings,
            record_id=record_id,
        )


class RecommendationService:
    """Service managing agronomic recommendation generation across selectable providers."""

    def __init__(
        self,
        provider: RecommendationProvider | str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._providers: dict[str, RecommendationProvider] = {}

        # 1. Wire RuleProvider as default
        self._rule_provider = get_rule_provider()
        self._providers["rules"] = self._rule_provider

        # 2. Gemini provider slot (Phase 10)
        has_gemini_key = bool(
            self.settings.gemini_api_key
            and self.settings.gemini_api_key.get_secret_value().strip()
        )
        self._gemini_slot = GeminiProviderSlot(
            fallback_provider=self._rule_provider,
            api_key_configured=has_gemini_key,
        )
        self._providers["gemini"] = self._gemini_slot

        # 3. Determine and set active provider
        target_provider = provider or self.settings.recommendation_provider
        self._active_provider: RecommendationProvider = self._rule_provider
        self._active_provider_name: str = "rules"
        self.set_provider(target_provider)

    @property
    def active_provider(self) -> RecommendationProvider:
        """Return the currently active recommendation provider."""
        return self._active_provider

    @property
    def active_provider_name(self) -> str:
        """Return the name of the currently active recommendation provider."""
        return self._active_provider_name

    def set_provider(self, provider: RecommendationProvider | str) -> None:
        """Swap the active recommendation provider by instance or registered name.

        Args:
            provider: Either a registered provider name ('rules', 'gemini') or
                      a custom object implementing the RecommendationProvider protocol.
        """
        if isinstance(provider, str):
            clean_name = provider.strip().lower()
            if clean_name not in self._providers:
                available = sorted(self._providers.keys())
                raise ValueError(
                    f"Unknown recommendation provider '{provider}'. Available providers: {available}"
                )
            self._active_provider = self._providers[clean_name]
            self._active_provider_name = clean_name
            logger.info(
                "RecommendationService active provider swapped to '%s'", clean_name
            )
        elif isinstance(provider, RecommendationProvider):
            self._active_provider = provider
            self._active_provider_name = getattr(provider, "provider_name", "custom")
            self._providers[self._active_provider_name] = provider
            logger.info(
                "RecommendationService active provider set to custom instance '%s'",
                self._active_provider_name,
            )
        else:
            raise TypeError(
                f"Expected RecommendationProvider instance or str, got {type(provider).__name__}"
            )

    def register_provider(self, name: str, provider: RecommendationProvider) -> None:
        """Register a new named provider (e.g. for testing, plugins, or Phase 10)."""
        clean_name = name.strip().lower()
        self._providers[clean_name] = provider
        logger.info("Registered recommendation provider '%s'", clean_name)

    def recommend(
        self,
        crop: str,
        predicted_class: str,
        modality: Literal["sensor", "image", "image_features"] = "sensor",
        confidence: float | None = None,
        uncertain: bool = False,
        sensor_readings: dict[str, Any] | None = None,
        record_id: UUID | None = None,
    ) -> RecommendationRead:
        """Generate a recommendation using the active provider."""
        return self._active_provider.recommend(
            crop=crop,
            predicted_class=predicted_class,
            modality=modality,
            confidence=confidence,
            uncertain=uncertain,
            sensor_readings=sensor_readings,
            record_id=record_id,
        )

    def recommend_from_prediction(
        self,
        prediction: PredictionResponse,
        sensor_readings: dict[str, Any] | None = None,
        record_id: UUID | None = None,
    ) -> RecommendationRead:
        """Generate a recommendation directly from a PredictionResponse using the active provider."""
        return self._active_provider.recommend_from_prediction(
            prediction=prediction,
            sensor_readings=sensor_readings,
            record_id=record_id,
        )


# Singleton access
_recommendation_service_instance: RecommendationService | None = None


def get_recommendation_service(
    provider: RecommendationProvider | str | None = None,
    settings: Settings | None = None,
) -> RecommendationService:
    """Return the application singleton RecommendationService instance."""
    global _recommendation_service_instance
    if (
        _recommendation_service_instance is None
        or provider is not None
        or settings is not None
    ):
        _recommendation_service_instance = RecommendationService(
            provider=provider, settings=settings
        )
    return _recommendation_service_instance


def reset_recommendation_service() -> None:
    """Reset the singleton instance (primarily for testing fixtures)."""
    global _recommendation_service_instance
    _recommendation_service_instance = None
