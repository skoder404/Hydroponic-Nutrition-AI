# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for Hydroponic Nutrition AI - Backend
# Pinned Base Image: Python 3.11 Slim Bookworm per ADR 0002

# ==============================================================================
# Stage 1: Build Dependencies
# ==============================================================================
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build tools and PostgreSQL client dev headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy pinned requirements
COPY requirements/backend.txt /build/requirements/backend.txt
COPY requirements/ml-cpu.txt /build/requirements/ml-cpu.txt

# Install dependencies (CPU build by default per T9-001)
# ==============================================================================
# GPU BUILD VARIANT (Documentation):
# To build with NVIDIA CUDA GPU acceleration (e.g. for RTX 3050 Laptop GPU):
# 1. Base on an nvidia runtime or install torch cu124 wheels:
#    RUN pip install --no-cache-dir -r /build/requirements/backend.txt && \
#        pip install --no-cache-dir -r /build/requirements/ml.txt
# 2. In docker-compose or docker run, pass `--gpus all` and set `FORCE_CPU=false`.
# ==============================================================================
RUN pip install --no-cache-dir -r /build/requirements/backend.txt \
    && pip install --no-cache-dir -r /build/requirements/ml-cpu.txt

# ==============================================================================
# Stage 2: Runtime Image (Non-Root User)
# ==============================================================================
FROM python:3.11-slim-bookworm AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FORCE_CPU=true \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install runtime system libraries (libpq for postgres, curl for healthcheck, ca-certs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create dedicated non-root user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Create required runtime directories for mounts (models, uploads, data)
# NOTE: Datasets and model checkpoints are NEVER baked into the image layers.
# They are mounted dynamically as volumes at runtime per Overview and T9-001.
RUN mkdir -p /app/models /app/uploads /app/data && \
    chown -R appuser:appgroup /app

# Copy application source code
COPY --chown=appuser:appgroup backend /app/backend
COPY --chown=appuser:appgroup ml /app/ml
COPY --chown=appuser:appgroup alembic /app/alembic
COPY --chown=appuser:appgroup alembic.ini /app/alembic.ini

# Switch to non-root user
USER appuser:appgroup

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
