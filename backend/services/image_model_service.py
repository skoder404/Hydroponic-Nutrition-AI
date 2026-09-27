from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from backend.config import IMAGE_MODEL_DIR


class ImageModelService:
    """Pluggable image interface. The actual CNN model can be added later."""

    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else IMAGE_MODEL_DIR
        self.model_path = self.model_dir / "model.h5"
        self.ready = False

    def predict(self, image_path: Optional[str] = None) -> Dict[str, Any]:
        return {
            "predicted_class": "unknown",
            "confidence": 0.0,
            "class_probabilities": {},
            "status": "pending_cnn_model",
            "message": "CNN inference is not configured. No visual prediction was made.",
        }
