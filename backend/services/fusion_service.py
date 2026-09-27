from __future__ import annotations

from typing import Any, Dict

from backend.config import HIGH_CONFIDENCE, LOW_CONFIDENCE


class FusionService:
    def __init__(self, high_confidence: float = HIGH_CONFIDENCE, low_confidence: float = LOW_CONFIDENCE):
        self.high_confidence = high_confidence
        self.low_confidence = low_confidence

    def fuse(self, sensor_result: Dict[str, Any], image_result: Dict[str, Any]) -> Dict[str, Any]:
        sensor_conf = float(sensor_result.get("confidence", 0.0) or 0.0)
        image_conf = float(image_result.get("confidence", 0.0) or 0.0)
        sensor_pred = str(sensor_result.get("prediction", "unknown"))
        image_pred = str(image_result.get("predicted_class", "unknown"))
        image_available = image_result.get("status") == "ready" and image_pred != "unknown"

        if not image_available:
            if sensor_conf < self.low_confidence:
                status = "insufficient_confidence"
                diagnosis = "Sensor confidence is too low for a reliable interpretation."
                interpretation = "The CNN is not available yet, and sensor confidence is below the configured review threshold."
            elif sensor_conf < self.high_confidence:
                status = "insufficient_confidence"
                diagnosis = f"The sensor model classified the readings as {sensor_pred}, but confidence is below the high-confidence threshold."
                interpretation = "The CNN is not available yet. Verify the sensor readings and crop condition before acting."
            elif sensor_pred == "Normal":
                status = "normal_condition"
                diagnosis = "The sensor model classified the readings as Normal."
                interpretation = "Only the sensor model is available. Continue monitoring; this is not a visual assessment."
            else:
                status = "sensor_only"
                diagnosis = f"The sensor model classified the readings as {sensor_pred}."
                interpretation = "Only the sensor model is available. Verify the readings and inspect the crop before adjusting nutrients."
            return {
                "status": status,
                "final_diagnosis": diagnosis,
                "confidence": round(sensor_conf, 4),
                "interpretation": interpretation,
                "modalities_used": ["sensor"],
            }

        if sensor_conf >= self.high_confidence and image_conf >= self.high_confidence:
            return {
                "status": "multimodal_review",
                "final_diagnosis": "Both models returned high-confidence findings; review each modality separately.",
                "confidence": round(min(sensor_conf, image_conf), 4),
                "interpretation": f"Sensor model: {sensor_pred}. Image model: {image_pred}. No cross-model class mapping is assumed.",
                "modalities_used": ["sensor", "image"],
            }

        if sensor_conf >= self.high_confidence:
            return {
                "status": "sensor_only",
                "final_diagnosis": f"The sensor model classified the readings as {sensor_pred}.",
                "confidence": round(sensor_conf, 4),
                "interpretation": f"The sensor model is above the high-confidence threshold. The image model returned {image_pred}; keep the findings separate.",
                "modalities_used": ["sensor", "image"],
            }

        if image_conf >= self.high_confidence:
            return {
                "status": "image_only",
                "final_diagnosis": f"The image model classified the image as {image_pred}.",
                "confidence": round(image_conf, 4),
                "interpretation": "The image model is above the high-confidence threshold. Sensor confidence is lower; review the two findings separately.",
                "modalities_used": ["sensor", "image"],
            }

        return {
            "status": "insufficient_confidence",
            "final_diagnosis": "Neither model returned confidence above the review threshold.",
            "confidence": round(min(sensor_conf, image_conf), 4),
            "interpretation": "Review the raw sensor readings and image separately; do not make a corrective change from these results alone.",
            "modalities_used": ["sensor", "image"],
        }
