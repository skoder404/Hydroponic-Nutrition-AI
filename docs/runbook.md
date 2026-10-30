# Hydroponic Nutrition AI - Demo Runbook & Operations Guide (v0.1.0)

> **Purpose:** Step-by-step instructions for team members, operators, and evaluators to start, operate, demonstrate, and verify the Hydroponic Nutrition AI application without external assistance.

---

## 1. Quick Start (Docker Compose Full-Stack)

The fastest and most reliable way to run the entire system (PostgreSQL 16, FastAPI backend, Next.js frontend) is via Docker Compose.

### Start the Stack
```bash
# 1. Copy environment template if not present
cp .env.example .env

# 2. Build and start services in detached mode
docker compose up -d --build
```

### Verify Service Health
```bash
# Verify backend API health
curl -s http://localhost:8000/api/v1/health | jq .

# Verify frontend web interface
curl -sI http://localhost:3000 | head -n 5
```

- **Web Application:** [http://localhost:3000](http://localhost:3000)
- **FastAPI Documentation (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **OpenAPI Schema:** [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)
- **Operational Metrics:** [http://localhost:8000/metrics](http://localhost:8000/metrics)

### Stop the Stack
```bash
# Stop containers while preserving database volume
docker compose down

# Stop and wipe database data (clean restart)
docker compose down -v
```

---

## 2. Local Machine Development Setup (Without Docker)

### Prerequisites
- Python 3.11+
- Node.js 20+ & npm
- PostgreSQL 16 (or local Docker container for database only: `docker compose up -d db`)

### Backend Setup
```bash
# 1. Activate virtual environment
source .venv/bin/activate       # On Linux/macOS
.\.venv\Scripts\Activate.ps1    # On Windows PowerShell

# 2. Run database migrations
alembic upgrade head

# 3. Start FastAPI server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
# Accessible at http://localhost:3000
```

---

## 3. Loading Models & Inspecting the Model Registry

The system automatically loads all promoted production models at startup into memory from `models/` pointer manifests.

### Check Loaded Models via CLI
```bash
python -c "from backend.app.services.model_service import get_model_registry_service; ms = get_model_registry_service(); ms.load_promoted_models(); print(ms.get_model_summaries())"
```

### Inspect via API
```bash
curl -s http://localhost:8000/api/v1/models | jq .
```

The registry returns active promoted models across **lettuce**, **spinach**, and **tomato** for both `image` (EfficientNetB0) and `sensor` (Decision Tree / Random Forest) modalities with Macro-F1 evaluation scores and provenance metadata.

---

## 4. Pointing the App to Campus Hardware (`.env`)

The system features a modular campus device adapter (`DEVICE_ADAPTER`) that interfaces with the stationary CPPLUS camera and polyhouse sensor telemetry.

### Configuration
In `.env`:
```ini
# Device Adapter Mode: 'campus' for live hardware, 'mock' for synthetic testing, 'manual' for operator input
DEVICE_ADAPTER=campus

# Campus Polyhouse Server URL (IP of the local polyhouse edge server)
CAMPUS_SERVER_URL=http://192.168.1.100:5000/telemetry/latest

# Campus Camera RTSP/MJPEG/HTTP Stream URL
CAMPUS_CAMERA_URL=http://192.168.1.100:5000/camera/current_frame.jpg

# Freshness thresholds
CAMPUS_MAX_AGE_S=60
CAMPUS_REQUEST_TIMEOUT_S=3.0
```

### Verification & Fallback
1. Test server connectivity:
   ```bash
   curl -I --connect-timeout 3 http://192.168.1.100:5000/telemetry/latest
   ```
2. In the UI, navigate to `/analyze` and select the **Campus Live** source tab.
3. If the server is unreachable or readings exceed `CAMPUS_MAX_AGE_S` (60 seconds), the UI clearly flags **Data Stale / Offline** and seamlessly switches to the **Manual Entry** tab so inspection can proceed without disruption.

---

## 5. Step-by-Step Live Demo Script

Use this 7-step sequence during academic, stakeholder, or technical demonstrations:

### Step 1: Subsystem Health & Diagnostics
- Navigate to `/` or open `http://localhost:8000/api/v1/health`.
- Show the top-level connectivity banner:
  - Database status (`connected`)
  - Detected compute device (`cuda:0` RTX 3050 or `cpu`)
  - Model registry health (`ok`)
  - Operational fallback counters (`gemini_to_rules: 0`, `gpu_to_cpu: 0`, `segmentation_failure: 0`)

### Step 2: Model Registry & Evaluation Evidence
- Open `/models` or `http://localhost:8000/api/v1/models`.
- Show that models are versioned and audited.
- Highlight the **Macro-F1** evaluation metrics for each crop, explaining why Macro-F1 prevents majority-class bias in unbalanced nutrient deficiencies.

### Step 3: Multi-Leaf Instance Segmentation & Classification
- Navigate to `/analyze`.
- Select crop: **Spinach** or **Tomato**.
- Upload a foliar image from `image_data/test/` or choose a sample card.
- Click **Analyze Plant**.
- Point out the **vine progression** stepper: Validating ➔ Segmenting ➔ Classifying ➔ Recommending.
- Inspect the segmented leaves: each proposal is outlined, isolated, and given its own disease/deficiency classification, confidence score, and leaf-level area ratio.
- Note any rejected proposals with clear explanations (e.g., "Area ratio < 0.003", "Non-leaf background").

### Step 4: Water Chemistry Sensor Classification
- In `/analyze`, enter or view sensor readings (pH, EC, Water Temperature, Nitrogen, Phosphorus, Potassium).
- Observe real-time boundary validation against agronomic ranges.
- Review sensor condition classification (e.g. `Healthy`, `EC_High`, `Nitrogen_Low`).

### Step 5: Explainable Recommendations (Gemini + Rules)
- View the generated **Nutritional Recommendation**:
  - Highlights *evidence* (which leaf anomalies or sensor deviations informed the advice).
  - Explicitly states source: `gemini` (AI-assisted advisory) or `rules` (deterministic rule engine).
  - Emphasizes strict safety invariant: advisory only, no automated chemical dosing.

### Step 6: Audit History & Export
- Navigate to `/history`.
- Verify the observation was persisted with UTC timestamp, crop, input modalities, and classification verdict.
- Click **Export JSON** or **Export CSV** to demonstrate clinical data archiving.

### Step 7: Resilience & Fallback Demonstration
- Demonstrate graceful degradation when an upstream fails:
  - If Gemini API key is missing or invalid, system automatically delivers deterministic rule recommendations and increments `gemini_to_rules`.
  - Check `http://localhost:8000/api/v1/metrics` to show the counter incremented to `1`.

---

## 6. Known Limitations to State During a Demo

Transparency and scientific integrity are core tenets of this system. State these points clearly:

1. **Unpaired Image & Sensor Data:** Image datasets and sensor water chemistry logs were collected independently. They are presented side-by-side to support human decision-making, never merged into a fake multimodal index.
2. **Rule-Derived Sensor Ground Truth:** Several sensor labels (such as high/low EC or nutrient limits) were derived from agronomic threshold rules rather than destructively harvested leaf tissue tests.
3. **Advisory Scope:** Recommendations provide diagnostic checklists and corrective guidance for human growers; the software never directly controls nutrient pumps or solenoid valves.
4. **Single-Plant vs. Canopy:** SAM 2.1 segmentation is calibrated for top-down or oblique single-crop photography. Extreme canopy occlusions may yield rejected proposals.

---

## 7. Operational Fallback Behaviour

| Subsystem Failure | Fallback Behavior | Telemetry Metric |
|---|---|---|
| **Gemini API Outage / Timeout** | System immediately executes deterministic, agronomist-verified rule recommendations from `ml/recommendations/`. | `gemini_to_rules_total` |
| **CUDA Out of Memory (OOM)** | Automatically flushes VRAM cache, cleans tensors, and transparently retries inference on CPU. | `gpu_to_cpu_total` |
| **Leaf Segmentation Failure** | Returns HTTP 422 with `SEGMENTATION_FAILED` (retryable=True) and clear guidance to supply a clearer leaf photograph. | `segmentation_failure_total` |
| **Campus Hardware Offline** | Flags telemetry as stale (>60s) or offline and provides instant manual data entry fallback. | UI Banner & logs |
| **PostgreSQL Outage** | `/health` reports `status: degraded` and `db: unavailable`; analysis continues in in-memory advisory mode with "Not Saved" notification. | `/api/v1/health` status |

---

## 8. Database Backup and Restore

### Automated Backup
```bash
./scripts/backup_db.sh
# Stored in backups/backup_*.sql.gz (retains last 7 days automatically)
```

### Restore Database
```bash
./scripts/restore_db.sh backups/backup_YYYYMMDD_HHMMSS.sql.gz
```

---

## 9. HTTPS / TLS Reverse Proxy Deployment

In campus LAN or staging environments, terminate TLS via Caddy:

```bash
docker compose -f docker-compose.yml -f deploy/docker-compose.tls.yml up -d
```

- Accessible via `https://localhost` (or configured `DOMAIN`).
- Plain HTTP on port 80 automatically redirects to HTTPS (308 redirect).
- HSTS headers (`max-age=31536000`) enforced on all responses.

---

## 10. Release v0.1.0 Verification Checklist

Before tagging and cutting a release, verify:

- [x] Full unit test suite passes: `pytest tests/unit`
- [x] Integration and fallback suites pass: `pytest tests/integration`
- [x] OpenAPI canonical contract synchronized: `python scripts/export_openapi.py`
- [x] Pre-commit hooks clean: `pre-commit run --all-files`
- [x] Secret scanner confirms zero credentials committed: `python scripts/secret_scan.py`
- [x] Runbook verified end-to-end: `python scripts/verify_t9_006.py`
- [x] Git release tag created: `git tag -a v0.1.0 -m "Release v0.1.0: Hydroponic Nutrition AI"`
