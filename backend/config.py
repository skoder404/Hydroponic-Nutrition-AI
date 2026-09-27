from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DEFAULT_MODEL_DIR = BASE_DIR / "ml_models"
OUTPUT_MODEL_DIR = PROJECT_ROOT / "outputs"
DEFAULT_SENSOR_MODEL_PATH = DEFAULT_MODEL_DIR / "hydro_sensor_model.joblib"
MODEL_DIR = DEFAULT_MODEL_DIR if DEFAULT_SENSOR_MODEL_PATH.exists() else OUTPUT_MODEL_DIR

SENSOR_MODEL_PATH = MODEL_DIR / "hydro_sensor_model.joblib"
IMAGE_MODEL_DIR = MODEL_DIR / "image_model"
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = BASE_DIR / "hydroponic.db"

FEATURES = ["ph", "ec", "water_temp", "humidity", "air_temp"]

HIGH_CONFIDENCE = float(os.getenv("HIGH_CONFIDENCE", "0.8"))
LOW_CONFIDENCE = float(os.getenv("LOW_CONFIDENCE", "0.5"))
