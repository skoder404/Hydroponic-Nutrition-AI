from __future__ import annotations

from typing import Any, Dict, List


class RecommendationService:
    def build_recommendation(self, *, sensor_result: Dict[str, Any], image_result: Dict[str, Any], fusion_result: Dict[str, Any]) -> Dict[str, Any]:
        sensor_pred = sensor_result.get("prediction", "unknown")
        image_pred = image_result.get("predicted_class", "unknown")
        status = fusion_result.get("status", "insufficient_confidence")

        summary = "The current readings do not provide enough confidence for a corrective action."
        actions: List[str] = []
        warnings: List[str] = []
        reasons: List[str] = []

        if sensor_pred != "unknown":
            reasons.append(f"Sensor model class: {sensor_pred}.")
        if image_pred != "unknown":
            reasons.append(f"Image model class: {image_pred}.")

        if status == "sensor_only":
            summary = f"Sensor-only result: {sensor_pred}. Visual analysis is not available yet."
            actions.append("Verify the sensor values with a fresh reading and inspect the crop before changing the nutrient solution.")
            warnings.append("No nutrient dose or specific correction is inferred from this classification.")
        elif status == "normal_condition":
            summary = "The sensor model classified these readings as Normal; no visual assessment was made."
            actions.append("Continue routine monitoring and repeat the reading if plant appearance changes.")
        elif status == "multimodal_review":
            summary = "Both models returned high-confidence findings; their classes have not been mapped to a shared diagnosis."
            actions.append("Review the sensor class and image class separately, then verify against crop observations before acting.")
            warnings.append("Cross-model agreement is not inferred from confidence alone.")
        elif status == "image_only":
            summary = "The image model returned a high-confidence class while sensor confidence was lower."
            actions.append("Inspect the crop and collect a fresh sensor reading before making a correction.")
        elif status == "conflicting_evidence":
            summary = "The sensor and image evidence are mixed, so the system avoids forcing a diagnosis."
            actions.append("Take another measurement and capture a fresh image before making a correction.")
            warnings.append("Avoid a high-risk adjustment based on mixed evidence.")
        elif status == "insufficient_confidence":
            summary = "The current confidence is too low for a reliable diagnosis."
            actions.append("Take another sensor reading and review the crop directly before making any adjustment.")

        if not actions:
            actions.append("Continue routine crop monitoring.")

        return {
            "summary": summary,
            "actions": actions,
            "warnings": warnings,
            "reasons": reasons,
        }
