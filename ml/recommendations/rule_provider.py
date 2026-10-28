"""Deterministic rule-based recommendation provider (T6-008).

Conforms to TECH_RULES and PROJECT_OVERVIEW Section 20:
- Maps (crop, predicted_class, confidence_tier, modality) to conservative checks and actions.
- Loads rules from ml/recommendations/rules/<crop>.v1.yaml.
- Returns re-measure and inspection advice when confidence is low or uncertain=True.
- Returns explicit 'insufficient information' for unsupported or unknown combinations.
- Strictly forbids nutrient dosage or chemical concentration numbers.
- Cites the specific prediction and rule ID that produced the recommendation.
- Marks every rule as 'advisory, unreviewed' per Section 20.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

import yaml  # type: ignore[import-untyped]

from backend.app.schemas.prediction import PredictionResponse
from backend.app.schemas.recommendation import RecommendationRead

logger = logging.getLogger(__name__)

RULES_DIR = Path(__file__).resolve().parent / "rules"


class RuleProvider:
    """Deterministic rule-based recommendation provider implementing Overview §20."""

    provider_name: str = "rules"

    def __init__(self, rules_dir: Path | str | None = None) -> None:
        self.rules_dir = Path(rules_dir).resolve() if rules_dir else RULES_DIR
        self._rules_cache: dict[str, dict[str, Any]] = {}
        self._load_all_rules()

    def _load_all_rules(self) -> None:
        """Preload and validate all crop rule YAML files in the rules directory."""
        if not self.rules_dir.is_dir():
            logger.warning("Rules directory '%s' not found.", self.rules_dir)
            return

        for yaml_file in sorted(self.rules_dir.glob("*.v1.yaml")):
            crop_name = yaml_file.stem.split(".")[0].lower()
            try:
                data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._rules_cache[crop_name] = data
                    logger.info(
                        "Loaded recommendation rules for '%s' (v%s)",
                        crop_name,
                        data.get("rules_version"),
                    )
            except Exception as exc:
                logger.error("Failed to load rules YAML from '%s': %s", yaml_file, exc)

    def get_crop_rules(self, crop: str) -> dict[str, Any] | None:
        """Retrieve the rules mapping for a given crop."""
        clean_crop = crop.strip().lower()
        if clean_crop not in self._rules_cache:
            file_path = self.rules_dir / f"{clean_crop}.v1.yaml"
            if file_path.is_file():
                try:
                    data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        self._rules_cache[clean_crop] = data
                except Exception as exc:
                    logger.error("Failed loading %s: %s", file_path, exc)
                    return None
        return self._rules_cache.get(clean_crop)

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
        """Formulate a conservative recommendation citing the applied rule and prediction evidence.

        Args:
            crop: Target crop cultivar ('lettuce', 'spinach', 'tomato').
            predicted_class: Output class from model prediction.
            modality: Input modality ('sensor' or 'image').
            confidence: Model confidence score in [0.0, 1.0].
            uncertain: Whether the prediction was flagged uncertain or low-confidence.
            sensor_readings: Optional telemetry observations to attach as evidence.
            record_id: Optional UUID of the parent analysis record.

        Returns:
            RecommendationRead conforming to TECH_RULES error envelope and database schema.
        """
        clean_crop = crop.strip().lower()
        clean_modality = (
            "image" if modality in ("image", "image_features") else "sensor"
        )
        crop_rules = self.get_crop_rules(clean_crop)

        # 1. Check for missing or unsupported crop rules
        if not crop_rules:
            return self._build_insufficient_info(
                crop=clean_crop,
                predicted_class=predicted_class,
                modality=clean_modality,
                confidence=confidence,
                uncertain=uncertain,
                reason=f"No validated ruleset available for crop '{clean_crop}'.",
                sensor_readings=sensor_readings,
                record_id=record_id,
            )

        rules_version = str(crop_rules.get("rules_version", "v1.0.0"))
        disclaimer = str(crop_rules.get("disclaimer", "Advisory recommendation."))
        rules_matrix = crop_rules.get("rules", {}).get(clean_modality, {})

        # 2. Check for unsupported class combination
        if predicted_class not in rules_matrix:
            return self._build_insufficient_info(
                crop=clean_crop,
                predicted_class=predicted_class,
                modality=clean_modality,
                confidence=confidence,
                uncertain=uncertain,
                reason=f"Unsupported combination: No validated rule for {clean_crop}/{clean_modality} class '{predicted_class}'.",
                rules_version=rules_version,
                sensor_readings=sensor_readings,
                record_id=record_id,
            )

        rule = rules_matrix[predicted_class]
        rule_citation = f"rules/{clean_crop}.v1.yaml#{clean_modality}/{predicted_class}"

        # 3. Handle low-confidence / uncertain predictions
        # Low confidence returns re-measure and inspect advice rather than strong actions
        is_low_confidence = uncertain or (confidence is not None and confidence < 0.40)

        if is_low_confidence:
            summary = (
                f"Uncertain prediction ({predicted_class}): Model confidence is low. "
                "Secondary verification and inspection advised before taking corrective measures."
            )
            checks = list(rule.get("checks", [])) + [
                "Re-measure parameters using independent calibrated handheld instrumentation.",
                "Inspect physical plant root and foliar tissue in person.",
            ]
            actions = list(
                rule.get(
                    "low_confidence_action",
                    [
                        "Verify environmental and solution readings before adjusting cultivation setpoints."
                    ],
                )
            )
        else:
            summary = str(
                rule.get("summary", f"Condition '{predicted_class}' detected.")
            )
            checks = list(rule.get("checks", []))
            actions = list(rule.get("actions", []))

        evidence: dict[str, Any] = {
            "crop": clean_crop,
            "modality": clean_modality,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "uncertain": uncertain,
            "is_low_confidence": is_low_confidence,
            "rule_citation": rule_citation,
            "rules_version": rules_version,
            "status": "advisory, unreviewed",
        }
        if sensor_readings:
            evidence["sensor_readings"] = sensor_readings

        content: dict[str, Any] = {
            "summary": summary,
            "checks": checks,
            "actions": actions,
            "status": "advisory, unreviewed",
            "disclaimer": disclaimer,
            "rule_citation": rule_citation,
        }

        return RecommendationRead(
            id=uuid4(),
            record_id=record_id or uuid4(),
            source="rules",
            rules_version=rules_version,
            content=content,
            evidence=evidence,
            created_at=datetime.now(timezone.utc),
        )

    def recommend_from_prediction(
        self,
        prediction: PredictionResponse,
        sensor_readings: dict[str, Any] | None = None,
        record_id: UUID | None = None,
    ) -> RecommendationRead:
        """Convenience method accepting a validated PredictionResponse directly."""
        return self.recommend(
            crop=prediction.crop,
            predicted_class=prediction.predicted_class,
            modality=prediction.modality,
            confidence=prediction.confidence,
            uncertain=prediction.uncertain,
            sensor_readings=sensor_readings,
            record_id=record_id,
        )

    def _build_insufficient_info(
        self,
        crop: str,
        predicted_class: str,
        modality: str,
        confidence: float | None,
        uncertain: bool,
        reason: str,
        rules_version: str = "v1.0.0",
        sensor_readings: dict[str, Any] | None = None,
        record_id: UUID | None = None,
    ) -> RecommendationRead:
        """Formulate an explicit 'insufficient information' response per Section 20.3."""
        evidence: dict[str, Any] = {
            "crop": crop,
            "modality": modality,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "uncertain": uncertain,
            "rule_citation": "insufficient_information",
            "rules_version": rules_version,
            "status": "advisory, unreviewed",
            "reason": reason,
        }
        if sensor_readings:
            evidence["sensor_readings"] = sensor_readings

        content: dict[str, Any] = {
            "summary": f"insufficient information: {reason}",
            "checks": [
                "Inspect sensor probes and camera positioning to confirm valid data capture.",
                "Review system logs and crop cultivar configuration.",
            ],
            "actions": [
                "Consult grower logbook or certified diagnostic service; insufficient information to recommend corrective steps.",
            ],
            "status": "advisory, unreviewed",
            "rule_citation": "insufficient_information",
        }

        return RecommendationRead(
            id=uuid4(),
            record_id=record_id or uuid4(),
            source="rules",
            rules_version=rules_version,
            content=content,
            evidence=evidence,
            created_at=datetime.now(timezone.utc),
        )


# Singleton access
_rule_provider_instance: RuleProvider | None = None


def get_rule_provider(rules_dir: Path | str | None = None) -> RuleProvider:
    """Return the application singleton RuleProvider instance."""
    global _rule_provider_instance
    if _rule_provider_instance is None or rules_dir is not None:
        _rule_provider_instance = RuleProvider(rules_dir=rules_dir)
    return _rule_provider_instance


def reset_rule_provider() -> None:
    """Reset the singleton instance (primarily for testing)."""
    global _rule_provider_instance
    _rule_provider_instance = None
