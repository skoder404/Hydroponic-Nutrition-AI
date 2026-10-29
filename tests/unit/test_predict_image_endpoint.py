"""Unit tests for POST /predict/image endpoint (T6-007).

Acceptance Criteria:
- [x] Missing `input_mode` is rejected with 422
- [x] Plant mode returns per-leaf results and a summary
- [x] Unavailable family is reported clearly
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.api.v1.predict import get_image_service_dep
from backend.app.core.rate_limit import get_rate_limiter
from backend.app.main import create_app
from backend.app.services.image_service import ImageService, reset_image_service
from backend.app.services.model_service import reset_model_registry_service
from tests.unit.test_image_service import create_mock_segmenter_with_results


def _create_synthetic_plant_image_bytes(format_name: str = "JPEG") -> bytes:
    """Generate in-memory byte stream of a synthetic plant image with 2 vegetation patches."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Green circular spots
    cv2.circle(img, (60, 60), 30, (25, 200, 25), -1)
    cv2.circle(img, (140, 140), 30, (25, 200, 25), -1)

    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format=format_name)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def clean_environment():
    """Reset singletons and rate limiter before and after each test."""
    reset_model_registry_service()
    reset_image_service()
    get_rate_limiter().reset()
    yield
    reset_model_registry_service()
    reset_image_service()
    get_rate_limiter().reset()


@pytest.fixture
def client():
    """Create a FastAPI test client with loaded promoted models and mock segmenter."""
    app = create_app(configure_logging=False)
    mock_seg = create_mock_segmenter_with_results(valid_count=2, rejected_count=0)
    mock_service = ImageService(segmenter=mock_seg)
    app.dependency_overrides[get_image_service_dep] = lambda: mock_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# -----------------------------------------------------------------------------
# Acceptance Criterion 1: Missing input_mode is rejected with 422
# -----------------------------------------------------------------------------


def test_missing_input_mode_rejected_with_422(client: TestClient):
    """AC 1: Missing input_mode returns HTTP 422 in unified error envelope."""
    img_bytes = _create_synthetic_plant_image_bytes()
    files = {"file": ("plant.jpg", img_bytes, "image/jpeg")}
    data = {"crop": "lettuce"}  # input_mode omitted

    response = client.post("/api/v1/predict/image", data=data, files=files)
    assert response.status_code == 422
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] in ("MISSING_FIELD", "INVALID_INPUT")
    assert "input_mode" in str(body).lower()


def test_empty_or_invalid_input_mode_rejected_with_422(client: TestClient):
    """AC 1: Empty or invalid input_mode returns HTTP 422."""
    img_bytes = _create_synthetic_plant_image_bytes()
    files = {"file": ("plant.jpg", img_bytes, "image/jpeg")}

    # Empty string
    response = client.post(
        "/api/v1/predict/image",
        data={"crop": "lettuce", "input_mode": ""},
        files=files,
    )
    assert response.status_code == 422
    assert "input_mode" in response.text.lower()

    # Invalid value
    response = client.post(
        "/api/v1/predict/image",
        data={"crop": "lettuce", "input_mode": "unsupported_mode"},
        files=files,
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"


# -----------------------------------------------------------------------------
# Acceptance Criterion 2: Plant mode returns per-leaf results and a summary
# -----------------------------------------------------------------------------


def test_plant_mode_returns_per_leaf_results_and_summary(client: TestClient):
    """AC 2: Plant mode returns per-leaf predictions, uncertainty, and discrete summary."""
    img_bytes = _create_synthetic_plant_image_bytes()
    files = {"file": ("plant.jpg", img_bytes, "image/jpeg")}
    data = {
        "crop": "lettuce",
        "input_mode": "plant",
        "provenance": "manual",
    }

    response = client.post("/api/v1/predict/image", data=data, files=files)
    assert response.status_code == 200, response.text
    result = response.json()

    # Core response envelope
    assert result["crop"] == "lettuce"
    assert result["modality"] == "image"
    assert result["input_mode"] == "plant"
    assert result["model_id"] is not None
    assert result["model_version"] is not None
    assert len(result["model_version"]) > 0

    # AC 2: per-leaf results
    assert "leaves" in result
    assert isinstance(result["leaves"], list)
    assert len(result["leaves"]) > 0

    for leaf in result["leaves"]:
        assert "leaf_id" in leaf
        assert "bbox" in leaf
        assert len(leaf["bbox"]) == 4
        assert "predicted_class" in leaf
        assert "display_label" in leaf
        assert "confidence" in leaf
        assert "confidence_semantics" in leaf
        assert "uncertain" in leaf
        assert isinstance(leaf["uncertain"], bool)

    # AC 2: Discrete summary (no averaged confidence)
    assert "summary" in result
    summary = result["summary"]
    assert summary is not None
    assert "leaf_count" in summary
    assert summary["leaf_count"] == len(result["leaves"])
    assert "class_counts" in summary
    assert isinstance(summary["class_counts"], dict)
    assert sum(summary["class_counts"].values()) == summary["leaf_count"]
    # Verify no averaged score
    assert "average_confidence" not in summary
    assert "mean_confidence" not in summary


def test_leaf_mode_returns_single_leaf_prediction(client: TestClient):
    """Verify single-leaf mode returns per-leaf prediction."""
    img_bytes = _create_synthetic_plant_image_bytes(format_name="PNG")
    files = {"image": ("leaf.png", img_bytes, "image/png")}
    data = {
        "crop": "spinach",
        "input_mode": "leaf",
    }

    response = client.post("/api/v1/predict/image", data=data, files=files)
    assert response.status_code == 200, response.text
    result = response.json()

    assert result["crop"] == "spinach"
    assert result["modality"] == "image"
    assert len(result["leaves"]) >= 1
    assert result["confidence"] is not None
    assert result["display_label"] is not None


# -----------------------------------------------------------------------------
# Acceptance Criterion 3: Unavailable family is reported clearly
# -----------------------------------------------------------------------------


def test_unavailable_family_reported_clearly(client: TestClient):
    """AC 3: Requesting an unpromoted or nonexistent model family returns MODEL_UNAVAILABLE."""
    img_bytes = _create_synthetic_plant_image_bytes()
    files = {"file": ("plant.jpg", img_bytes, "image/jpeg")}
    data = {
        "crop": "lettuce",
        "input_mode": "plant",
        "model_family": "random_forest",
    }

    response = client.post("/api/v1/predict/image", data=data, files=files)
    assert response.status_code == 503
    body = response.json()
    assert "error" in body
    err = body["error"]
    assert err["code"] == "MODEL_UNAVAILABLE"
    # AC 3: Naming what is available
    assert "EfficientNetB0" in err["message"] or "EfficientNetB0" in str(
        err.get("details", {})
    )
    assert "random_forest" in err["message"]


def test_missing_image_file_rejected(client: TestClient):
    """Verify omitting image file returns 422."""
    response = client.post(
        "/api/v1/predict/image",
        data={"crop": "lettuce", "input_mode": "plant"},
        files={},
    )
    assert response.status_code == 422
    assert "file" in response.text.lower() or "image" in response.text.lower()


def test_unsupported_media_type_rejected(client: TestClient):
    """Verify text file disguised as .jpg returns 415 UNSUPPORTED_MEDIA."""
    fake_img = b"echo 'not an image file'"
    files = {"file": ("photo.jpg", fake_img, "image/jpeg")}
    data = {"crop": "lettuce", "input_mode": "plant"}

    response = client.post("/api/v1/predict/image", data=data, files=files)
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA"
