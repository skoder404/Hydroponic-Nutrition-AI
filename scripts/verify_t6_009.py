"""Verification script for Task T6-009: RecommendationService with provider interface.

Verifies:
1. RecommendationProvider protocol is defined and adhered to by RuleProvider and GeminiProviderSlot.
2. Service works with no Gemini configured (null key, default config).
3. Output includes `source: rules` and `rules_version` by default.
4. Provider can be swapped by config (Settings.recommendation_provider = "rules" or "gemini").
5. Custom provider can be registered and swapped via service.set_provider().
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    get_rule_provider,
)


class MockCustomProvider:
    """Mock provider to test runtime swapping and protocol adherence."""

    provider_name: str = "custom_test"

    def recommend(self, **kwargs) -> RecommendationRead:
        return RecommendationRead(
            id=uuid4(),
            record_id=kwargs.get("record_id") or uuid4(),
            source="rules",  # Must be 'rules' or 'gemini' per schema CheckConstraint
            rules_version="custom_v1",
            content={"summary": "Custom mock recommendation"},
            evidence={"test": True},
        )

    def recommend_from_prediction(
        self, prediction: PredictionResponse, **kwargs
    ) -> RecommendationRead:
        return self.recommend(**kwargs)


def verify_t6_009() -> None:
    print("=== Verifying T6-009: RecommendationService with Provider Interface ===")

    # 1. Verify RecommendationProvider protocol adherence
    rule_prov = get_rule_provider()
    gemini_slot = GeminiProviderSlot()
    mock_prov = MockCustomProvider()

    assert isinstance(rule_prov, RecommendationProvider), (
        "RuleProvider does not adhere to RecommendationProvider protocol"
    )
    assert isinstance(gemini_slot, RecommendationProvider), (
        "GeminiProviderSlot does not adhere to RecommendationProvider protocol"
    )
    assert isinstance(mock_prov, RecommendationProvider), (
        "MockCustomProvider does not adhere to RecommendationProvider protocol"
    )
    print(
        "[PASS] RecommendationProvider protocol verified across RuleProvider and GeminiProviderSlot."
    )

    # 2. Acceptance Criterion 1: Service works with no Gemini configured
    reset_recommendation_service()
    settings_no_gemini = Settings(gemini_api_key=None, recommendation_provider="rules")
    service = RecommendationService(settings=settings_no_gemini)

    rec = service.recommend(
        crop="lettuce",
        predicted_class="NitrogenDeficiency",
        modality="sensor",
        confidence=0.88,
        uncertain=False,
    )
    assert rec is not None, "Service failed to generate recommendation without Gemini"
    print("[PASS] Criterion 1 verified: Service works with no Gemini configured.")

    # 3. Acceptance Criterion 2: Output includes source: rules and rules_version
    assert rec.source == "rules", f"Expected source 'rules', got '{rec.source}'"
    assert rec.rules_version, "rules_version is missing in recommendation output"
    assert "actions" in rec.content
    assert "checks" in rec.content
    print(
        f"[PASS] Criterion 2 verified: Output includes source='{rec.source}' and rules_version='{rec.rules_version}'."
    )

    # 4. Acceptance Criterion 3: Provider can be swapped by config
    # Test A: Swapping to 'gemini' via config when no key is configured (falls back cleanly to rules)
    settings_gemini_cfg = Settings(
        gemini_api_key=None, recommendation_provider="gemini"
    )
    service_gemini = RecommendationService(settings=settings_gemini_cfg)
    assert service_gemini.active_provider_name == "gemini", (
        f"Expected active provider 'gemini', got '{service_gemini.active_provider_name}'"
    )
    rec_gemini = service_gemini.recommend(
        crop="spinach",
        predicted_class="Spinach_Healthy",
        modality="image",
        confidence=0.95,
        uncertain=False,
    )
    assert rec_gemini.source == "rules", (
        "Gemini slot with no key should fall back to rules output"
    )

    # Test B: Swapping active provider at runtime via service.set_provider()
    service.set_provider("gemini")
    assert service.active_provider_name == "gemini"
    service.set_provider("rules")
    assert service.active_provider_name == "rules"

    # Test C: Register and swap to custom provider
    service.register_provider("mock", mock_prov)
    service.set_provider("mock")
    assert service.active_provider_name == "mock"
    mock_rec = service.recommend(
        crop="tomato",
        predicted_class="Healthy",
        modality="sensor",
    )
    assert mock_rec.rules_version == "custom_v1"
    print(
        "[PASS] Criterion 3 verified: Provider swapped by config and runtime registration."
    )

    # 5. Verify recommend_from_prediction method
    service.set_provider("rules")
    pred = PredictionResponse(
        crop="tomato",
        predicted_class="Tomato___Bacterial_spot",
        display_label="Bacterial Spot",
        modality="image",
        confidence=0.91,
        confidence_semantics="multinomial_probability",
        uncertain=False,
        model_id="tomato_test_model",
        model_version="v1.0.0",
        timestamp="2026-10-28T16:00:00Z",
        provenance="live",
    )
    rec_from_pred = service.recommend_from_prediction(pred)
    assert rec_from_pred.source == "rules"
    assert rec_from_pred.evidence["crop"] == "tomato"
    assert rec_from_pred.evidence["predicted_class"] == "Tomato___Bacterial_spot"
    print("[PASS] recommend_from_prediction verified with valid PredictionResponse.")

    # 6. Verify singleton access functions
    reset_recommendation_service()
    s1 = get_recommendation_service()
    s2 = get_recommendation_service()
    assert s1 is s2, "get_recommendation_service did not return singleton instance"
    reset_recommendation_service()
    print(
        "[PASS] Singleton access get_recommendation_service and reset_recommendation_service verified."
    )

    print("=== All T6-009 Acceptance Criteria Successfully Verified ===")


if __name__ == "__main__":
    verify_t6_009()
