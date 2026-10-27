"""Unit tests for server-side upload validation utility (T3-002).

Acceptance Criteria:
1. A renamed text file with a .jpg extension is rejected (UNSUPPORTED_MEDIA, HTTP 415).
2. Path traversal in filename has no effect (UUID storage strictly in uploads/).
3. Oversized file returns 413 code PAYLOAD_TOO_LARGE.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi import FastAPI, File, UploadFile
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.core.errors import (
    InvalidInputError,
    PayloadTooLargeError,
    UnsupportedMediaError,
    register_error_handlers,
)
from backend.app.core.uploads import (
    ValidatedUpload,
    validate_and_save_image,
    validate_upload_file,
)


def _create_test_image(
    format_name: str = "JPEG",
    size: tuple[int, int] = (64, 64),
    color: tuple[int, int, int] = (34, 139, 34),
) -> bytes:
    """Helper creating raw bytes of a valid synthetic image."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format_name)
    return buf.getvalue()


# ------------------------------------------------------------------------------
# Criterion 1: Renamed Text File with .jpg Extension is Rejected
# ------------------------------------------------------------------------------


def test_renamed_text_file_with_jpg_extension_is_rejected(tmp_path: Path) -> None:
    """Acceptance Criterion 1: Magic bytes detect disguised text file and reject it."""
    fake_jpg_content = b"echo 'This is not an image file'; rm -rf /"

    with pytest.raises(UnsupportedMediaError) as exc_info:
        validate_and_save_image(
            file_bytes=fake_jpg_content,
            original_filename="payload.jpg",
            content_type="image/jpeg",
            uploads_dir=tmp_path,
        )

    err = exc_info.value
    assert err.code == "UNSUPPORTED_MEDIA"
    assert err.status_code == 415
    assert err.retryable is False


def test_corrupt_bytes_with_partial_jpeg_header_rejected(tmp_path: Path) -> None:
    """Rejects corrupted or truncated files even if initial 3 bytes mimic JPEG."""
    corrupt_jpeg = b"\xff\xd8\xff" + b"Corrupted garbage data not a real JPEG"

    with pytest.raises(UnsupportedMediaError) as exc_info:
        validate_and_save_image(
            file_bytes=corrupt_jpeg,
            original_filename="corrupt.jpg",
            content_type="image/jpeg",
            uploads_dir=tmp_path,
        )

    assert exc_info.value.code == "UNSUPPORTED_MEDIA"
    assert exc_info.value.status_code == 415


def test_empty_file_rejected(tmp_path: Path) -> None:
    """Zero-byte files are rejected with INVALID_INPUT (HTTP 422)."""
    with pytest.raises(InvalidInputError) as exc_info:
        validate_and_save_image(
            file_bytes=b"",
            original_filename="empty.jpg",
            content_type="image/jpeg",
            uploads_dir=tmp_path,
        )

    assert exc_info.value.code == "INVALID_INPUT"
    assert exc_info.value.status_code == 422


# ------------------------------------------------------------------------------
# Criterion 2: Path Traversal in Filename Has No Effect
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "malicious_filename",
    [
        "../../../../etc/passwd",
        "..\\..\\Windows\\System32\\cmd.exe",
        "/var/log/syslog.jpg",
        "nested/deep/directory/traversal.jpg",
        "....//....//shell.php",
    ],
)
def test_path_traversal_in_filename_has_no_effect(
    tmp_path: Path, malicious_filename: str
) -> None:
    """Acceptance Criterion 2: Output file is always named via UUID and placed strictly in uploads_dir."""
    valid_bytes = _create_test_image("JPEG", size=(100, 100))
    upload_dir = tmp_path / "safe_uploads"

    result = validate_and_save_image(
        file_bytes=valid_bytes,
        original_filename=malicious_filename,
        content_type="image/jpeg",
        uploads_dir=upload_dir,
    )

    assert isinstance(result, ValidatedUpload)
    # The saved file must reside strictly inside upload_dir
    assert result.saved_path.parent == upload_dir.resolve()
    assert result.saved_path.is_file()

    # Client-supplied traversal path components must NOT appear anywhere in the filename
    assert "passwd" not in result.saved_path.name
    assert "cmd.exe" not in result.saved_path.name
    assert ".." not in result.saved_path.name

    # Filename must match generated UUID pattern with canonical extension (.jpg)
    assert result.saved_path.name.endswith(".jpg")
    stem = result.saved_path.stem
    # UUID has 36 characters with 4 dashes
    assert len(stem) == 36
    assert stem.count("-") == 4


# ------------------------------------------------------------------------------
# Criterion 3: Oversized File Returns 413 Code PAYLOAD_TOO_LARGE
# ------------------------------------------------------------------------------


def test_oversized_file_returns_413_payload_too_large(tmp_path: Path) -> None:
    """Acceptance Criterion 3: Files exceeding MAX_UPLOAD_MB raise PAYLOAD_TOO_LARGE with 413."""
    # 2 MB buffer tested against 1 MB limit
    two_mb_payload = b"\xff\xd8\xff\xe0" + b"\x00" * (2 * 1024 * 1024)

    with pytest.raises(PayloadTooLargeError) as exc_info:
        validate_and_save_image(
            file_bytes=two_mb_payload,
            original_filename="large.jpg",
            content_type="image/jpeg",
            max_upload_mb=1,  # 1 MB threshold
            uploads_dir=tmp_path,
        )

    err = exc_info.value
    assert err.code == "PAYLOAD_TOO_LARGE"
    assert err.status_code == 413
    assert err.retryable is False
    assert "exceeds maximum allowed limit" in err.message
    assert err.details["max_allowed_mb"] == 1


# ------------------------------------------------------------------------------
# End-to-End FastAPI Integration Tests (HTTP Status & Error Envelope)
# ------------------------------------------------------------------------------


@pytest.fixture
def test_app(tmp_path: Path) -> FastAPI:
    """Create a minimal FastAPI test app mounting an upload endpoint."""
    app = FastAPI()
    register_error_handlers(app)

    @app.post("/upload")
    async def upload_endpoint(file: UploadFile = File(...)) -> dict[str, str]:  # noqa: B008
        validated = await validate_upload_file(
            file, max_upload_mb=1, uploads_dir=tmp_path
        )
        return {"file_id": validated.file_id, "path": str(validated.saved_path)}

    return app


def test_api_renamed_text_file_returns_415(test_app: FastAPI) -> None:
    """FastAPI endpoint returns HTTP 415 UNSUPPORTED_MEDIA on disguised text file."""
    client = TestClient(test_app)
    fake_file = io.BytesIO(b"Plain text disguised as image")

    response = client.post(
        "/upload",
        files={"file": ("fake_plant.jpg", fake_file, "image/jpeg")},
    )

    assert response.status_code == 415
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "UNSUPPORTED_MEDIA"
    assert data["error"]["retryable"] is False


def test_api_oversized_file_returns_413(test_app: FastAPI) -> None:
    """FastAPI endpoint returns HTTP 413 PAYLOAD_TOO_LARGE on oversized upload."""
    client = TestClient(test_app)
    # Exceeds 1 MB cap
    oversized = io.BytesIO(b"\x00" * (2 * 1024 * 1024))

    response = client.post(
        "/upload",
        files={"file": ("oversized.jpg", oversized, "image/jpeg")},
    )

    assert response.status_code == 413
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "PAYLOAD_TOO_LARGE"
    assert data["error"]["retryable"] is False


def test_api_valid_png_and_jpeg_uploads_succeed(test_app: FastAPI) -> None:
    """Valid image files upload successfully and are stored under UUIDs."""
    client = TestClient(test_app)

    # 1. Valid JPEG
    jpeg_bytes = _create_test_image("JPEG", size=(64, 64))
    res_jpeg = client.post(
        "/upload",
        files={"file": ("lettuce.jpeg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert res_jpeg.status_code == 200
    data_jpeg = res_jpeg.json()
    assert "file_id" in data_jpeg

    # 2. Valid PNG
    png_bytes = _create_test_image("PNG", size=(64, 64))
    res_png = client.post(
        "/upload",
        files={"file": ("spinach.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert res_png.status_code == 200
    data_png = res_png.json()
    assert "file_id" in data_png
