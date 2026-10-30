"""Integration tests for image and sensor prediction paths (T8-005).

Acceptance Criteria:
- [x] Each path has a success and a failure test
- [x] Stub models are clearly marked as test fixtures
- [x] No real datasets required in CI

Paths Tested:
1. Image Upload -> Instance Segmentation -> Prediction (Plant Mode)
   - Success: Multi-leaf plant image returns individual leaves and summary.
   - Failure: Hardware/routine segmentation failure returns retryable SEGMENTATION_FAILED error.
2. Leaf Crop -> Features / Direct Classification -> Prediction (Leaf Mode)
   - Success: Single leaf image returns top-level predicted class and confidence.
   - Success: Leaf crop image to feature extraction to stub classifier pipeline.
   - Failure: Missing required input_mode is rejected with HTTP 422.
   - Failure: Corrupt/empty mask raises typed extraction error without silent fallback.
   - Failure: Corrupt image file returns HTTP 415 UNSUPPORTED_MEDIA.
3. Sensor Telemetry Payload -> Sensor Model Prediction
   - Success: Valid sensor payload returns display label, confidence, limitations, and provenance.
   - Failure: Invalid sensor unit (e.g. EC in volts) rejected with HTTP 422 INVALID_INPUT.
   - Failure: Missing sensor readings rejected with HTTP 422 MISSING_FIELD / INVALID_INPUT.
"""

from __future__ import annotations

import io
from collections.abc import Generator
from typing import Any

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.api.v1.predict import get_image_service_dep
from backend.app.core.rate_limit import get_rate_limiter
from backend.app.devices.adapter import reset_device_adapter
from backend.app.main import create_app
from backend.app.services.image_service import ImageService, reset_image_service
from backend.app.services.model_service import reset_model_registry_service
from backend.app.services.sensor_service import reset_sensor_service
from ml.features.common import EmptyMaskError
from ml.features.lettuce import extract_lettuce_features
from ml.segmentation.sam2_segmenter import (
    SAM2Segmenter,
    SegmentationError,
    SegmentationResult,
)


# =============================================================================
# TEST FIXTURES: Small In-Memory Stub Models & Generators (No Real Datasets in CI)
# =============================================================================


class StubImageClassifierFixture(nn.Module):
    """TEST FIXTURE: Tiny stub PyTorch image classification model.

    Marked as a test fixture. Operates deterministically without downloading
    or reading pretrained weights from disk.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        h = self.pool(h).flatten(1)
        return self.fc(h)


class StubSensorClassifierFixture:
    """TEST FIXTURE: Lightweight scikit-learn compatible stub classifier.

    Marked as a test fixture. Implements predict() and predict_proba()
    to test sensor feature pipelines without requiring full dataset training.
    """

    def __init__(self, classes: list[str]) -> None:
        self.classes_ = np.array(classes)

    def predict(self, X: np.ndarray | list[Any]) -> np.ndarray:
        return np.array([self.classes_[0]] * len(X))

    def predict_proba(self, X: np.ndarray | list[Any]) -> np.ndarray:
        probs = np.zeros((len(X), len(self.classes_)))
        probs[:, 0] = 0.95
        probs[:, 1:] = 0.05 / (len(self.classes_) - 1)
        return probs


def create_stub_segmenter_fixture(
    valid_leaf_count: int = 2,
) -> SAM2Segmenter:
    """TEST FIXTURE: In-memory SAM2 segmenter returning synthetic leaf masks."""
    mock_seg = SAM2Segmenter.__new__(SAM2Segmenter)
    mock_seg._model = True
    mock_seg._predictor = True
    mock_seg._loaded_device = "cpu"
    mock_seg.checkpoint_path = "stub_fixtures/stub_sam2.pt"
    mock_seg.config_path = "stub_fixtures/stub_sam2.yaml"
    mock_seg.fallback_on_oom = False

    def mock_segment(
        image: np.ndarray,
        point_coords: Any = None,
        point_labels: Any = None,
        box: Any = None,
        simulate_oom: bool = False,
    ) -> SegmentationResult:
        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=bool)
        if point_coords and point_coords[0][0] < w // 2:
            mask[20:60, 20:60] = True
            score = 0.96
        else:
            mask[70:120, 70:120] = True
            score = 0.91

        return SegmentationResult(
            mask=mask,
            config_version="stub_sam2:v1",
            device_used="cpu",
            score=score,
            fallback_triggered=False,
        )

    mock_seg.segment = mock_segment  # type: ignore[assignment]
    return mock_seg


def create_failing_segmenter_fixture() -> SAM2Segmenter:
    """TEST FIXTURE: Segmenter stub that simulates hardware/runtime failure."""
    failing_seg = SAM2Segmenter.__new__(SAM2Segmenter)
    failing_seg._model = True
    failing_seg._predictor = True
    failing_seg._loaded_device = "cpu"

    def fail_segment(*args: Any, **kwargs: Any) -> SegmentationResult:
        raise SegmentationError(
            "Simulated camera bus hardware timeout during SAM2 inference"
        )

    failing_seg.segment = fail_segment  # type: ignore[assignment]
    return failing_seg


def _create_synthetic_plant_image_bytes(format_name: str = "PNG") -> bytes:
    """TEST FIXTURE: Generate in-memory byte stream of synthetic multi-leaf plant image."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (30, 210, 30), -1)
    cv2.circle(img, (140, 140), 35, (40, 200, 40), -1)

    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format=format_name)
    return buf.getvalue()


def _create_synthetic_leaf_crop_bytes(format_name: str = "PNG") -> bytes:
    """TEST FIXTURE: Generate in-memory byte stream of a single isolated leaf image."""
    img = np.zeros((150, 150, 3), dtype=np.uint8)
    cv2.ellipse(img, (75, 75), (40, 25), 30, 0, 360, (35, 220, 35), -1)

    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format=format_name)
    return buf.getvalue()


# =============================================================================
# PYTEST SETUP & CLIENT FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def clean_environment() -> Generator[None, None, None]:
    """Reset singletons, rate limiters, and mock adapters between tests."""
    reset_model_registry_service()
    reset_image_service()
    reset_sensor_service()
    reset_device_adapter()
    get_rate_limiter().reset()
    yield
    reset_model_registry_service()
    reset_image_service()
    reset_sensor_service()
    reset_device_adapter()
    get_rate_limiter().reset()


@pytest.fixture
def client_with_stub_segmenter() -> Generator[TestClient, None, None]:
    """FastAPI TestClient wired with stub SAM2 segmenter fixture."""
    app = create_app(configure_logging=False)
    stub_seg = create_stub_segmenter_fixture(valid_leaf_count=2)
    mock_service = ImageService(segmenter=stub_seg)
    app.dependency_overrides[get_image_service_dep] = lambda: mock_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def client_with_failing_segmenter() -> Generator[TestClient, None, None]:
    """FastAPI TestClient wired with failing segmenter stub."""
    app = create_app(configure_logging=False)
    failing_seg = create_failing_segmenter_fixture()
    mock_service = ImageService(segmenter=failing_seg)
    app.dependency_overrides[get_image_service_dep] = lambda: mock_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# =============================================================================
# PATH 1: Plant Image -> Instance Segmentation -> Prediction
# =============================================================================


def test_path1_plant_mode_segmentation_to_prediction_success(
    client_with_stub_segmenter: TestClient,
) -> None:
    """Path 1 Success: Upload plant photo -> segment leaves -> per-leaf predictions & summary."""
    img_bytes = _create_synthetic_plant_image_bytes(format_name="PNG")
    response = client_with_stub_segmenter.post(
        "/api/v1/predict/image",
        data={"crop": "lettuce", "input_mode": "plant"},
        files={"file": ("synthetic_plant.png", img_bytes, "image/png")},
    )

    assert response.status_code == 200, response.text
    data = response.json()

    # Invariants for plant mode
    assert data["crop"] == "lettuce"
    assert data["input_mode"] == "plant"
    assert data["leaves"] is not None
    assert len(data["leaves"]) >= 1, "Should segment and classify at least one leaf"

    # Per-leaf validation
    for leaf in data["leaves"]:
        assert "leaf_id" in leaf
        assert "predicted_class" in leaf
        assert "confidence" in leaf
        assert 0.0 <= leaf["confidence"] <= 1.0
        assert "bbox" in leaf
        assert len(leaf["bbox"]) == 4
        assert "contour_area" in leaf
        assert "uncertain" in leaf

    # Summary verification
    assert data["summary"] is not None
    summary = data["summary"]
    leaf_count = summary.get("leaf_count", summary.get("total_leaves"))
    assert leaf_count == len(data["leaves"])
    assert "class_counts" in summary or "leaf_counts" in summary


def test_path1_plant_mode_segmentation_failure_returns_retryable_error(
    client_with_failing_segmenter: TestClient,
) -> None:
    """Path 1 Failure: Segmentation failure returns retryable SEGMENTATION_FAILED error."""
    img_bytes = _create_synthetic_plant_image_bytes(format_name="PNG")
    response = client_with_failing_segmenter.post(
        "/api/v1/predict/image",
        data={"crop": "lettuce", "input_mode": "plant"},
        files={"file": ("synthetic_plant.png", img_bytes, "image/png")},
    )

    assert response.status_code == 422, (
        "Segmentation failure must map to HTTP 422 envelope"
    )
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "SEGMENTATION_FAILED"
    assert data["error"]["retryable"] is True
    assert (
        "hardware timeout" in data["error"]["message"].lower()
        or "failed" in data["error"]["message"].lower()
    )


# =============================================================================
# PATH 2: Leaf Crop -> Features / Direct Classification -> Prediction
# =============================================================================


def test_path2_leaf_mode_crop_to_prediction_endpoint_success(
    client_with_stub_segmenter: TestClient,
) -> None:
    """Path 2 Success: Single leaf image returns top-level class, confidence, and leaf prediction."""
    leaf_bytes = _create_synthetic_leaf_crop_bytes(format_name="PNG")
    response = client_with_stub_segmenter.post(
        "/api/v1/predict/image",
        data={"crop": "lettuce", "input_mode": "leaf"},
        files={"file": ("single_leaf.png", leaf_bytes, "image/png")},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["crop"] == "lettuce"
    assert data["input_mode"] in ("leaf", "single_leaf")
    assert data["predicted_class"] is not None
    assert data["confidence"] is not None
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["leaves"]) >= 1, (
        "Single leaf prediction should be present in leaves list"
    )


def test_path2_leaf_crop_features_to_stub_model_pipeline_success_and_failure() -> None:
    """Path 2 Success & Failure: Unit/Pipeline test of leaf crop -> feature extraction -> stub classifier."""
    # 1. Success pipeline: Generate synthetic leaf and mask
    img_rgb = np.full((120, 120, 3), 100, dtype=np.uint8)
    cv2.circle(img_rgb, (60, 60), 30, (20, 200, 20), -1)

    mask = np.zeros((120, 120), dtype=bool)
    mask[35:85, 35:85] = True

    # Extract 7-dimensional standardized lettuce features
    feature_vector = extract_lettuce_features(img_rgb, mask)
    features_np = feature_vector.to_numpy().reshape(1, -1)
    assert features_np.shape == (1, 7)

    # Classify with Stub Sensor/Feature Classifier Fixture
    classes = [
        "Healthier",
        "NitrogenDeficiency",
        "PhosphorusDeficiency",
        "PotassiumDeficiency",
    ]
    stub_classifier = StubSensorClassifierFixture(classes=classes)
    predicted_label = stub_classifier.predict(features_np)[0]
    probabilities = stub_classifier.predict_proba(features_np)[0]

    assert predicted_label in classes
    assert len(probabilities) == len(classes)
    assert np.isclose(np.sum(probabilities), 1.0)

    # 2. Failure mode: Empty mask raises typed EmptyMaskError
    empty_mask = np.zeros((120, 120), dtype=bool)
    with pytest.raises(EmptyMaskError):
        extract_lettuce_features(img_rgb, empty_mask)


def test_path2_leaf_mode_missing_input_mode_rejected(
    client_with_stub_segmenter: TestClient,
) -> None:
    """Path 2 Failure: Missing input_mode parameter rejected with HTTP 422 MISSING_FIELD."""
    leaf_bytes = _create_synthetic_leaf_crop_bytes(format_name="PNG")
    response = client_with_stub_segmenter.post(
        "/api/v1/predict/image",
        data={"crop": "lettuce"},  # Missing input_mode!
        files={"file": ("single_leaf.png", leaf_bytes, "image/png")},
    )

    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] in ("MISSING_FIELD", "INVALID_INPUT")
    assert (
        "input_mode" in str(data["error"]["details"]).lower()
        or "input_mode" in data["error"]["message"].lower()
    )


def test_path2_leaf_mode_corrupt_image_rejected(
    client_with_stub_segmenter: TestClient,
) -> None:
    """Path 2 Failure: Corrupt non-image bytes rejected with HTTP 415 UNSUPPORTED_MEDIA."""
    corrupt_bytes = b"This is not a real image file, just arbitrary ASCII text."
    response = client_with_stub_segmenter.post(
        "/api/v1/predict/image",
        data={"crop": "lettuce", "input_mode": "leaf"},
        files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
    )

    assert response.status_code == 415
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "UNSUPPORTED_MEDIA"


# =============================================================================
# PATH 3: Sensor Telemetry Payload -> Prediction
# =============================================================================


def test_path3_sensor_payload_to_prediction_success() -> None:
    """Path 3 Success: Valid sensor payload returns prediction, limitations, and metadata."""
    app = create_app(configure_logging=False)
    with TestClient(app) as client:
        payload = {
            "crop": "lettuce",
            "readings": {
                "ph": 6.2,
                "ec": 1.9,
                "water_temp": 21.5,
                "humidity": 65.0,
                "air_temp": 24.0,
            },
            "units": {
                "ph": "pH",
                "ec": "mS/cm",
                "water_temp": "degC",
                "humidity": "%",
                "air_temp": "degC",
            },
            "provenance": "manual",
        }
        response = client.post("/api/v1/predict/sensor", json=payload)
        assert response.status_code == 200, response.text
        data = response.json()

        assert data["crop"] == "lettuce"
        assert data["modality"] == "sensor"
        assert data["predicted_class"] is not None
        assert data["confidence"] is not None
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["model_version"] is not None
        assert data["timestamp"] is not None
        assert isinstance(data["limitations"], list)


def test_path3_sensor_payload_to_prediction_invalid_unit_failure() -> None:
    """Path 3 Failure: Invalid unit for physical telemetry rejected with HTTP 422 INVALID_INPUT."""
    app = create_app(configure_logging=False)
    with TestClient(app) as client:
        payload = {
            "crop": "lettuce",
            "readings": {"ph": 6.0, "ec": 1.8},
            "units": {
                "ph": "pH",
                "ec": "volts",
            },  # Invalid unit for EC (expected mS/cm)
            "provenance": "manual",
        }
        response = client.post("/api/v1/predict/sensor", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "INVALID_INPUT"
        assert (
            "unit" in str(data["error"]).lower()
            or "volts" in str(data["error"]).lower()
        )


def test_path3_sensor_payload_to_prediction_missing_readings_failure() -> None:
    """Path 3 Failure: Empty/missing sensor readings rejected with HTTP 422."""
    app = create_app(configure_logging=False)
    with TestClient(app) as client:
        payload = {
            "crop": "lettuce",
            "readings": {},  # Empty readings
            "provenance": "manual",
        }
        response = client.post("/api/v1/predict/sensor", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] in ("INVALID_INPUT", "MISSING_FIELD")
