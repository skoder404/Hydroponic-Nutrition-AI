# Hydroponic Nutrition AI

Hydroponic Nutrition AI combines sensor-based lettuce condition classification with observation history and a pluggable image-analysis interface. The existing sensor model is loaded for inference only; this application does not retrain or modify it. CNN inference remains disabled until a trained image model is available.

## Project structure

- `sensor_model_training.ipynb` — sensor model training and evaluation notebook. The app does not execute it.
- `backend/` — FastAPI inference API, SQLite persistence, sensor service, and pending CNN interface.
- `frontend/` — React + Vite observation and analysis dashboard.
- `Hydroponic Lettuce Dataset/` — training and evaluation datasets.
- `outputs/` — locally generated model artifact. This folder is ignored by Git.
- `mlm/` — local Python environment. This folder is ignored by Git.

## Dataset

The dataset is stored under `Hydroponic Lettuce Dataset/` and includes:

- `hydroponic_lettuce_train.csv`
- `hydroponic_lettuce_test.csv`
- `hydroponic_lettuce_full.csv`

## Model

The existing trained model is expected at:

- `outputs/hydro_sensor_model.joblib`

The backend also checks `backend/ml_models/hydro_sensor_model.joblib` first. Place the existing artifact in either location on a fresh clone. The artifact is intentionally not included in Git because generated outputs are ignored.

The API reports sensor confidence and class probabilities. Its image service explicitly returns `pending_cnn_model`; it does not invent a visual class. Fusion does not assume sensor and image classes share a taxonomy, and recommendations do not prescribe nutrient doses.

## Run locally

Install the backend dependencies in the project environment:

```powershell
python -m venv mlm
\.\mlm\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Start the API from the repository root:

```powershell
\.\mlm\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, start the dashboard:

```powershell
cd frontend
npm install
npm run dev
```

The dashboard defaults to `http://127.0.0.1:8000` for its API. Set `VITE_API_BASE_URL` before starting Vite to use another API URL.

## API

- `GET /api/health` — sensor, CNN, and database status.
- `POST /api/sensor/predict` — JSON sensor-only prediction without saving an observation.
- `POST /api/predict/combined` — JSON sensor analysis and persisted result.
- `POST /api/predict` — multipart sensor analysis with optional image attachment.
- `POST /api/image/predict` — image upload; returns pending until CNN integration is implemented.
- `POST /api/esp32/sensor` — store and classify an ESP32 reading, preserving its timestamp.
- `GET /api/sensor/latest` — latest observation and sensor prediction.
- `GET /api/history?limit=25` — most recent stored observations and predictions.

## Tests

Install the test client and run the API regression suite:

```powershell
\.\mlm\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
\.\mlm\Scripts\python.exe -m unittest discover -s tests -v
```

## Run the notebook

Open `sensor_model_training.ipynb` in Jupyter Notebook or VS Code and run the cells in order.

## Notes

The training notebook uses grouped cross-validation by date to avoid leakage between related sensor readings. The local SQLite database, image uploads, model artifact, Python environment, frontend dependencies, and frontend build output are excluded from Git.
