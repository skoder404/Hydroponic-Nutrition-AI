from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ObservationRecord:
    id: Optional[int]
    timestamp: str
    source: str
    ph: Optional[float] = None
    ec: Optional[float] = None
    water_temp: Optional[float] = None
    humidity: Optional[float] = None
    air_temp: Optional[float] = None
    image_path: Optional[str] = None


@dataclass
class PredictionRecord:
    id: Optional[int]
    observation_id: int
    sensor_prediction: Optional[str] = None
    sensor_confidence: Optional[float] = None
    sensor_probabilities: Optional[Any] = None
    image_prediction: Optional[str] = None
    image_confidence: Optional[float] = None
    image_probabilities: Optional[Any] = None
    fusion_status: Optional[str] = None
    final_diagnosis: Optional[str] = None
    final_confidence: Optional[float] = None
    recommendation: Optional[str] = None
    timestamp: Optional[str] = None
