"""Unit tests for Backend and Frontend Dockerfiles and .dockerignore rules (T9-001)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def test_dockerfiles_exist():
    """Verify that both backend and frontend Dockerfiles exist."""
    backend_dockerfile = ROOT / "backend" / "Dockerfile"
    frontend_dockerfile = ROOT / "frontend" / "Dockerfile"
    dockerignore = ROOT / ".dockerignore"

    assert backend_dockerfile.is_file(), "backend/Dockerfile must exist"
    assert frontend_dockerfile.is_file(), "frontend/Dockerfile must exist"
    assert dockerignore.is_file(), ".dockerignore must exist at repository root"


def test_backend_dockerfile_multi_stage_and_pinned():
    """Verify backend Dockerfile has multi-stage build, pinned base, and non-root user."""
    content = (ROOT / "backend" / "Dockerfile").read_text(encoding="utf-8")

    # Multi-stage check
    stages = re.findall(
        r"^FROM\s+([^\s]+)\s+AS\s+([^\s]+)", content, re.MULTILINE | re.IGNORECASE
    )
    assert len(stages) >= 2, (
        f"Backend Dockerfile must have at least 2 stages, found: {stages}"
    )

    # Pinned base images (no :latest or unpinned tags)
    for image, stage_name in stages:
        assert ":" in image, (
            f"Base image '{image}' in stage '{stage_name}' must have a pinned tag"
        )
        assert not image.endswith(":latest"), (
            f"Base image '{image}' must not use ':latest'"
        )
        assert "python:3.11" in image, (
            f"Backend image should use pinned python:3.11, got: {image}"
        )

    # Non-root user check
    assert re.search(r"^USER\s+appuser", content, re.MULTILINE), (
        "Backend must run as non-root appuser"
    )

    # CPU build default and documented GPU variant
    assert "FORCE_CPU=true" in content or "FORCE_CPU" in content
    assert "ml-cpu.txt" in content, (
        "Backend Dockerfile must install ml-cpu.txt by default"
    )
    assert "GPU" in content, "Backend Dockerfile must document GPU build variant"

    # Healthcheck instruction
    assert re.search(r"^HEALTHCHECK\s+", content, re.MULTILINE), (
        "Backend Dockerfile must define HEALTHCHECK"
    )


def test_frontend_dockerfile_multi_stage_and_pinned():
    """Verify frontend Dockerfile has multi-stage build, pinned base, and non-root user."""
    content = (ROOT / "frontend" / "Dockerfile").read_text(encoding="utf-8")

    # Multi-stage check
    stages = re.findall(
        r"^FROM\s+([^\s]+)\s+AS\s+([^\s]+)", content, re.MULTILINE | re.IGNORECASE
    )
    assert len(stages) >= 2, (
        f"Frontend Dockerfile must have at least 2 stages, found: {stages}"
    )

    # Pinned base images
    for image, stage_name in stages:
        assert ":" in image, (
            f"Base image '{image}' in stage '{stage_name}' must have a pinned tag"
        )
        assert not image.endswith(":latest"), (
            f"Base image '{image}' must not use ':latest'"
        )
        assert "node:" in image, (
            f"Frontend image should use pinned node base image, got: {image}"
        )

    # Non-root user check
    assert re.search(r"^USER\s+nextjs", content, re.MULTILINE), (
        "Frontend must run as non-root nextjs"
    )

    # Healthcheck instruction
    assert re.search(r"^HEALTHCHECK\s+", content, re.MULTILINE), (
        "Frontend Dockerfile must define HEALTHCHECK"
    )


def test_dockerignore_excludes_secrets_and_datasets():
    """Verify .dockerignore excludes datasets, models, caches, and secrets."""
    content = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    required_exclusions = [
        "image_data/",
        "sensor_data/",
        "processed_data/",
        "uploads/",
        "models/",
        ".env",
        ".venv/",
        "node_modules",
    ]

    for req in required_exclusions:
        matched = any(
            req in line or line.rstrip("/") == req.rstrip("/") for line in lines
        )
        assert matched, (
            f".dockerignore must exclude '{req}' to prevent dataset or secret leakage"
        )
