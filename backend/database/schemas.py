from typing import Optional
from pydantic import BaseModel, Field


class SensorInput(BaseModel):
    ph: float = Field(..., ge=0, le=14)
    ec: float = Field(..., ge=0)
    water_temp: float = Field(..., ge=0)
    humidity: float = Field(..., ge=0, le=100)
    air_temp: float = Field(..., ge=-50, le=80)


class ESP32SensorInput(SensorInput):
    device_id: str
    timestamp: Optional[str] = None


class ImagePredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: dict
    status: str = "ready"


class CombinedPredictionResponse(BaseModel):
    sensor: dict
    image: dict
    fusion: dict
    recommendation: dict
