"""Verification script for T6-007: Add POST /predict/image endpoint.

Acceptance Criteria:
- Missing input_mode is rejected with 422
- Plant mode returns per-leaf results and a summary
- Unavailable family is reported clearly
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.api.v1.predict import get_image_service_dep
from backend.app.core.rate_limit import get_rate_limiter
from backend.app.main import create_app
from backend.app.services.image_service import ImageService, reset_image_service
from backend.app.services.model_service import reset_model_registry_service
from tests.unit.test_image_service import create_mock_segmenter_with_results


def create_sample_image(format_name: str = "JPEG") -> bytes:
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (60, 60), 30, (25, 200, 25), -1)
    cv2.circle(img, (140, 140), 30, (25, 200, 25), -1)
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format=format_name)
    return buf.getvalue()


def main() -> int:
    print("Verifying T6-007: Add POST /predict/image endpoint...")

    reset_model_registry_service()
    reset_image_service()
    get_rate_limiter().reset()

    app = create_app(configure_logging=False)
    mock_seg = create_mock_segmenter_with_results(valid_count=2, rejected_count=0)
    mock_service = ImageService(segmenter=mock_seg)
    app.dependency_overrides[get_image_service_dep] = lambda: mock_service

    client = TestClient(app)
    img_bytes = create_sample_image("JPEG")

    # AC 1: Missing input_mode is rejected with 422
    resp_missing = client.post(
        "/api/v1/predict/image",
        data={"crop": "lettuce"},
        files={"file": ("plant.jpg", img_bytes, "image/jpeg")},
    )
    assert resp_missing.status_code == 422, (
        f"Expected 422 for missing input_mode, got {resp_missing.status_code}"
    )
    body_missing = resp_missing.json()
    assert "error" in body_missing
    assert body_missing["error"]["code"] in ("MISSING_FIELD", "INVALID_INPUT")
    print("  [PASS] AC 1: Missing input_mode is rejected with 422")

    # AC 2: Plant mode returns per-leaf results and a summary
    resp_plant = client.post(
        "/api/v1/predict/image",
        data={"crop": "lettuce", "input_mode": "plant"},
        files={"file": ("plant.jpg", img_bytes, "image/jpeg")},
    )
    assert resp_plant.status_code == 200, (
        f"Expected 200 for plant mode, got {resp_plant.status_code}: {resp_plant.text}"
    )
    body_plant = resp_plant.json()
    assert "leaves" in body_plant and len(body_plant["leaves"]) > 0, (
        "Expected leaves in plant mode"
    )
    assert "summary" in body_plant and body_plant["summary"] is not None, (
        "Expected summary in plant mode"
    )
    assert body_plant["summary"]["leaf_count"] == len(body_plant["leaves"])
    for leaf in body_plant["leaves"]:
        assert "predicted_class" in leaf
        assert "uncertain" in leaf
        assert "bbox" in leaf
    print("  [PASS] AC 2: Plant mode returns per-leaf results and a discrete summary")

    # AC 3: Unavailable family is reported clearly
    resp_unavail = client.post(
        "/api/v1/predict/image",
        data={
            "crop": "lettuce",
            "input_mode": "plant",
            "model_family": "nonexistent_arch",
        },
        files={"file": ("plant.jpg", img_bytes, "image/jpeg")},
    )
    assert resp_unavail.status_code == 503, (
        f"Expected 503 for unavailable model family, got {resp_unavail.status_code}"
    )
    body_unavail = resp_unavail.json()
    assert body_unavail["error"]["code"] == "MODEL_UNAVAILABLE"
    assert "nonexistent_arch" in body_unavail["error"]["message"]
    print(
        "  [PASS] AC 3: Unavailable family is reported clearly with 503 MODEL_UNAVAILABLE"
    )

    print("ALL ACs for T6-007 PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
