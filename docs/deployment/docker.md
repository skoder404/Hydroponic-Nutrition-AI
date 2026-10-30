# Docker Deployment Guide: Backend and Frontend (T9-001)

This guide documents the containerization strategy for the Hydroponic Nutrition AI system, covering multi-stage builds, non-root user execution, volume mounting conventions, and CPU/GPU build variants.

---

## 1. Container Architecture Overview

Both services are built using multi-stage Dockerfiles to minimize final image footprint, eliminate compiler tools from production layers, and maintain security boundaries:

```text
┌────────────────────────────────────────────────────────┐
│                    Backend Image                       │
├────────────────────────────┬───────────────────────────┤
│ Stage 1: builder           │ Stage 2: runner           │
│ - python:3.11-slim-bookworm│ - python:3.11-slim-bookworm│
│ - build-essential, libpq   │ - Minimal runtime libpq5  │
│ - Virtualenv dependencies  │ - Non-root user: appuser  │
│                            │ - Mount targets: /app/... │
└────────────────────────────┴───────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                    Frontend Image                      │
├────────────────────────────┬───────────────────────────┤
│ Stage 1: deps & Stage 2    │ Stage 3: runner           │
│ - node:20-alpine           │ - node:20-alpine          │
│ - npm ci & next build      │ - Non-root user: nextjs   │
│ - Static assets compiled   │ - Standalone Next.js run  │
└────────────────────────────┴───────────────────────────┘
```

---

## 2. Security & Layer Isolation Policies

### 2.1 Non-Root Container Execution
In compliance with enterprise security baselines and PRD non-functional requirements:
- **Backend:** Executes as `appuser:appgroup` (`UID 10001`, `GID 10001`).
- **Frontend:** Executes as `nextjs:nodejs` (`UID 1001`, `GID 1001`).
No container process runs as `root`.

### 2.2 Dataset & Model Volume Isolation
Per Canonical Overview §16 and Task T9-001 acceptance criteria:
- **Zero Datasets in Layers:** Raw sensor telemetry (`sensor_data/`), image data (`image_data/`), and processed split artifacts (`processed_data/`) are strictly excluded via `.dockerignore`.
- **Zero Model Weights in Layers:** Trained weights (`models/`) and uploaded inspection files (`uploads/`) are mounted dynamically as volumes at runtime.
- **Zero Secrets in Layers:** Environment files (`.env*`), private keys, and API tokens are barred via `.dockerignore`.

---

## 3. Building and Running Containers

### 3.1 Building the Backend (CPU Default)

From the project root:

```bash
docker build -f backend/Dockerfile -t hydroponics-backend:latest .
```

To run the backend standalone with mounted models and uploads:

```bash
docker run -d \
  --name hydroponics-backend \
  -p 8000:8000 \
  -v "$(pwd)/models:/app/models:ro" \
  -v "$(pwd)/uploads:/app/uploads:rw" \
  --env-file .env \
  hydroponics-backend:latest
```

### 3.2 Building the Frontend

From the project root (or inside `frontend/`):

```bash
docker build -f frontend/Dockerfile -t hydroponics-frontend:latest ./frontend
```

To run the frontend:

```bash
docker run -d \
  --name hydroponics-frontend \
  -p 3000:3000 \
  -e NEXT_PUBLIC_API_BASE_URL="http://127.0.0.1:8000" \
  hydroponics-frontend:latest
```

---

## 4. Hardware Acceleration: GPU Build Variant

By default, the backend container uses the CPU-only PyTorch build (`requirements/ml-cpu.txt`) to ensure zero-friction portability to campus lab machines without dedicated GPUs (`FORCE_CPU=true`).

### Enabling NVIDIA CUDA GPU Acceleration
For deployment on hosts with an NVIDIA GPU (e.g. RTX 3050 Laptop GPU):

1. **Install NVIDIA Container Toolkit** on the Docker host:
   ```bash
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```
2. **Switch to CUDA Wheels in Builder Stage:**
   In `backend/Dockerfile`, install using `requirements/ml.txt` (which pulls CUDA wheels):
   ```dockerfile
   RUN pip install --no-cache-dir -r /build/requirements/backend.txt && \
       pip install --no-cache-dir -r /build/requirements/ml.txt
   ```
3. **Run with `--gpus all` and `FORCE_CPU=false`:**
   ```bash
   docker run --gpus all \
     -e FORCE_CPU=false \
     -p 8000:8000 \
     -v "$(pwd)/models:/app/models:ro" \
     hydroponics-backend:latest
   ```
