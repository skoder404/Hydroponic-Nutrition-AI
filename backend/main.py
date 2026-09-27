from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import DB_PATH, UPLOAD_DIR
from backend.database.database import (
    get_history,
    get_latest_sensor_reading,
    init_db,
    save_analysis,
)
from backend.database.schemas import ESP32SensorInput, SensorInput
from backend.services.fusion_service import FusionService
from backend.services.image_model_service import ImageModelService
from backend.services.recommendation_service import RecommendationService
from backend.services.sensor_model_service import SensorModelService

MAX_IMAGE_BYTES = 10 * 1024 * 1024
IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

init_db()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Hydroponic Nutrition AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

sensor_service = SensorModelService()
image_service = ImageModelService()
fusion_service = FusionService()
recommendation_service = RecommendationService()


async def _save_image(image: UploadFile) -> Path:
    suffix = IMAGE_TYPES.get(image.content_type or "")
    if suffix is None:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")

    content = await image.read(MAX_IMAGE_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image files must be 10 MB or smaller.")
    valid_signature = (
        (suffix == ".jpg" and content.startswith(b"\xff\xd8\xff"))
        or (suffix == ".png" and content.startswith(b"\x89PNG\r\n\x1a\n"))
        or (suffix == ".webp" and content.startswith(b"RIFF") and content[8:12] == b"WEBP")
    )
    if not valid_signature:
        raise HTTPException(status_code=415, detail="The image content does not match its media type.")

    image_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    image_path.write_bytes(content)
    return image_path


async def _analyze(
    sensor_payload: Dict[str, float],
    *,
    source: str,
    image: UploadFile | None = None,
    timestamp: str | None = None,
) -> Dict[str, Any]:
    if not sensor_service.ready:
        raise HTTPException(status_code=503, detail="The trained sensor model is not available.")

    try:
        sensor_result = sensor_service.predict(sensor_payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    image_path = await _save_image(image) if image is not None else None
    image_result = image_service.predict(str(image_path) if image_path else None)
    fusion_result = fusion_service.fuse(sensor_result, image_result)
    recommendation = recommendation_service.build_recommendation(
        sensor_result=sensor_result,
        image_result=image_result,
        fusion_result=fusion_result,
    )

    try:
        observation_id = save_analysis(
            source=source,
            sensor_payload=sensor_payload,
            sensor_result=sensor_result,
            image_result=image_result,
            fusion_result=fusion_result,
            recommendation=recommendation,
            image_path=str(image_path) if image_path else None,
            timestamp=timestamp,
        )
    except Exception:
        if image_path is not None:
            image_path.unlink(missing_ok=True)
        raise

    return {
        "observation_id": observation_id,
        "sensor": {**sensor_result, "readings": sensor_payload},
        "image": image_result,
        "fusion": fusion_result,
        "recommendation": recommendation,
    }


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "sensor_model_ready": sensor_service.ready,
        "image_model_ready": image_service.ready,
        "image_status": "pending_cnn_model",
        "database_ready": DB_PATH.exists(),
    }


@app.post("/api/sensor/predict")
def predict_sensor(payload: SensorInput) -> Dict[str, Any]:
    if not sensor_service.ready:
        raise HTTPException(status_code=503, detail="The trained sensor model is not available.")
    return sensor_service.predict(payload.model_dump())


@app.post("/api/image/predict")
async def predict_image(image: UploadFile = File(...)) -> Dict[str, Any]:
    image_path = await _save_image(image)
    return image_service.predict(str(image_path))


@app.post("/api/predict")
async def predict_manual(
    ph: float = Form(..., ge=0),
    ec: float = Form(..., ge=0),
    water_temp: float = Form(..., ge=0),
    humidity: float = Form(..., ge=0, le=100),
    air_temp: float = Form(..., ge=-50, le=80),
    image: UploadFile | None = File(default=None),
) -> Dict[str, Any]:
    return await _analyze(
        {
            "ph": ph,
            "ec": ec,
            "water_temp": water_temp,
            "humidity": humidity,
            "air_temp": air_temp,
        },
        source="manual",
        image=image,
    )


@app.post("/api/predict/combined")
async def predict_combined(payload: SensorInput) -> Dict[str, Any]:
    return await _analyze(payload.model_dump(), source="manual")


@app.post("/api/esp32/sensor")
async def esp32_sensor(payload: ESP32SensorInput) -> Dict[str, Any]:
    result = await _analyze(
        payload.model_dump(exclude={"device_id", "timestamp"}),
        source="esp32",
        timestamp=payload.timestamp,
    )
    return {**result, "device_id": payload.device_id}


@app.get("/api/sensor/latest")
def latest_sensor() -> Dict[str, Any]:
    row = get_latest_sensor_reading()
    return row if row else {"status": "no_data"}


@app.get("/api/history")
def history(limit: int = Query(default=25, ge=1, le=100)) -> Dict[str, Any]:
    return {"items": get_history(limit=limit)}
