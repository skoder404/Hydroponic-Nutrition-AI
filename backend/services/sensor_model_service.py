import joblib
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from backend.config import FEATURES, SENSOR_MODEL_PATH


class SensorModelService:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else SENSOR_MODEL_PATH
        self.model = None
        self.features: List[str] = FEATURES
        self.classes: List[str] = []
        self.ready = False
        self.load_model()

    def load_model(self) -> None:
        if not self.model_path.exists():
            self.ready = False
            return
        payload = joblib.load(self.model_path)
        if isinstance(payload, dict):
            self.model = payload.get("model")
            self.features = payload.get("features") or FEATURES
            self.classes = list(payload.get("classes") or self.model.classes_)
        else:
            self.model = payload
            self.classes = list(getattr(self.model, "classes_", []))
        self.ready = self.model is not None

    def predict(self, sensor_input: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ready or self.model is None:
            raise FileNotFoundError("Sensor model is not available in backend/ml_models/hydro_sensor_model.joblib")

        payload = {feature: float(sensor_input.get(feature, 0)) for feature in self.features}
        frame = pd.DataFrame([payload], columns=self.features)
        prediction = self.model.predict(frame)[0]
        probabilities = self.model.predict_proba(frame)[0]
        class_map = dict(zip(self.model.classes_, probabilities))
        class_map = {str(key): float(value) for key, value in class_map.items()}
        confidence = max(class_map.values())
        return {
            "prediction": str(prediction),
            "confidence": round(float(confidence), 4),
            "probabilities": {k: round(float(v), 4) for k, v in class_map.items()},
        }
