#!/usr/bin/env python3
"""Verification script for server-side upload validation utility (T3-002).

Acceptance Criteria:
1. A renamed text file with a .jpg extension is rejected.
2. Path traversal in filename has no effect.
3. Oversized file returns 413 code PAYLOAD_TOO_LARGE.
"""
# ruff: noqa: E402

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, S110
        pass

from fastapi import FastAPI, File, UploadFile
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.core.errors import (
    PayloadTooLargeError,
    UnsupportedMediaError,
    register_error_handlers,
)
from backend.app.core.uploads import (
    validate_and_save_image,
    validate_upload_file,
)


def _make_sample_image(
    format_name: str = "JPEG", size: tuple[int, int] = (64, 64)
) -> bytes:
    """Generate in-memory valid image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=(34, 139, 34))
    img.save(buf, format=format_name)
    return buf.getvalue()


def verify_renamed_text_file_rejected() -> tuple[bool, str]:
    """Criterion 1: A renamed text file with a .jpg extension is rejected."""
    fake_jpg = b"This is a shell script disguised as an image.\necho 'pwned'\n"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Direct function test
        try:
            validate_and_save_image(
                file_bytes=fake_jpg,
                original_filename="malicious.jpg",
                content_type="image/jpeg",
                uploads_dir=tmp_path,
            )
            return (
                False,
                "Renamed text file was unexpectedly accepted by validate_and_save_image.",
            )
        except UnsupportedMediaError as err:
            if err.status_code != 415 or err.code != "UNSUPPORTED_MEDIA":
                return (
                    False,
                    f"Unexpected error attributes: code={err.code}, status={err.status_code}",
                )
        except Exception as exc:  # noqa: BLE001
            return False, f"Unexpected exception type: {type(exc).__name__}: {exc}"

        # 2. FastAPI endpoint test
        app = FastAPI()
        register_error_handlers(app)

        @app.post("/upload")
        async def upload_route(file: UploadFile = File(...)) -> dict[str, str]:  # noqa: B008
            res = await validate_upload_file(file, uploads_dir=tmp_path)
            return {"file_id": res.file_id}

        client = TestClient(app)
        resp = client.post(
            "/upload",
            files={"file": ("malicious.jpg", io.BytesIO(fake_jpg), "image/jpeg")},
        )
        if resp.status_code != 415:
            return (
                False,
                f"FastAPI endpoint returned status {resp.status_code}, expected 415.",
            )

        data = resp.json()
        if data.get("error", {}).get("code") != "UNSUPPORTED_MEDIA":
            return (
                False,
                f"Error code was {data.get('error', {}).get('code')}, expected 'UNSUPPORTED_MEDIA'.",
            )

    return (
        True,
        "Criterion 1 satisfied: Magic bytes guard correctly rejects renamed text files with HTTP 415 UNSUPPORTED_MEDIA.",
    )


def verify_path_traversal_has_no_effect() -> tuple[bool, str]:
    """Criterion 2: Path traversal in filename has no effect."""
    traversal_filenames = [
        "../../../../etc/passwd",
        "..\\..\\Windows\\System32\\cmd.exe",
        "/var/run/secret.jpg",
        "nested/path/traversal.jpg",
    ]
    valid_bytes = _make_sample_image("JPEG")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for malicious_name in traversal_filenames:
            upload_res = validate_and_save_image(
                file_bytes=valid_bytes,
                original_filename=malicious_name,
                content_type="image/jpeg",
                uploads_dir=tmp_path,
            )

            # Saved path must reside directly inside tmp_path
            if upload_res.saved_path.parent != tmp_path.resolve():
                return (
                    False,
                    f"Path traversal succeeded! File escaped to: {upload_res.saved_path}",
                )

            # Saved filename must be a pure UUID name and not contain client path tokens
            saved_name = upload_res.saved_path.name
            if ".." in saved_name or "passwd" in saved_name or "cmd.exe" in saved_name:
                return False, f"Client filename leaked into stored name: {saved_name}"

            if not saved_name.endswith(".jpg"):
                return False, f"Stored filename missing proper extension: {saved_name}"

            # Confirm file was written and is valid
            if not upload_res.saved_path.is_file():
                return False, f"Saved file does not exist at {upload_res.saved_path}"

    return (
        True,
        "Criterion 2 satisfied: Filename path traversal is neutralized; all files stored strictly under generated UUIDs in uploads directory.",
    )


def verify_oversized_file_returns_413() -> tuple[bool, str]:
    """Criterion 3: Oversized file returns 413 code PAYLOAD_TOO_LARGE."""
    one_mb_limit = 1
    oversized_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * (2 * 1024 * 1024)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Direct function test
        try:
            validate_and_save_image(
                file_bytes=oversized_bytes,
                original_filename="large.jpg",
                content_type="image/jpeg",
                max_upload_mb=one_mb_limit,
                uploads_dir=tmp_path,
            )
            return False, "Oversized file was unexpectedly accepted without error."
        except PayloadTooLargeError as err:
            if err.status_code != 413:
                return False, f"Expected status_code 413, got {err.status_code}"
            if err.code != "PAYLOAD_TOO_LARGE":
                return False, f"Expected code 'PAYLOAD_TOO_LARGE', got '{err.code}'"
        except Exception as exc:  # noqa: BLE001
            return False, f"Unexpected exception: {type(exc).__name__}: {exc}"

        # 2. FastAPI endpoint test
        app = FastAPI()
        register_error_handlers(app)

        @app.post("/upload")
        async def upload_route(file: UploadFile = File(...)) -> dict[str, str]:  # noqa: B008
            res = await validate_upload_file(
                file, max_upload_mb=one_mb_limit, uploads_dir=tmp_path
            )
            return {"file_id": res.file_id}

        client = TestClient(app)
        resp = client.post(
            "/upload",
            files={"file": ("large.jpg", io.BytesIO(oversized_bytes), "image/jpeg")},
        )
        if resp.status_code != 413:
            return (
                False,
                f"FastAPI endpoint returned status {resp.status_code}, expected 413.",
            )

        data = resp.json()
        if data.get("error", {}).get("code") != "PAYLOAD_TOO_LARGE":
            return (
                False,
                f"Error code was '{data.get('error', {}).get('code')}', expected 'PAYLOAD_TOO_LARGE'.",
            )

    return (
        True,
        "Criterion 3 satisfied: Uploads exceeding MAX_UPLOAD_MB strictly return HTTP 413 PAYLOAD_TOO_LARGE.",
    )


def main() -> int:
    """Run all verification checks for T3-002."""
    print("=" * 70)
    print("T3-002 Verification: Server-Side Upload Validation Utility")
    print("=" * 70)

    verifiers = [
        ("Criterion 1: Renamed text file rejected", verify_renamed_text_file_rejected),
        (
            "Criterion 2: Path traversal has no effect",
            verify_path_traversal_has_no_effect,
        ),
        ("Criterion 3: Oversized file returns 413", verify_oversized_file_returns_413),
    ]

    all_passed = True
    for label, fn in verifiers:
        ok, msg = fn()
        if ok:
            print(f"[PASS] {label}")
            print(f"       {msg}\n")
        else:
            print(f"[FAIL] {label}")
            print(f"       {msg}\n")
            all_passed = False

    if all_passed:
        print("ALL CRITERIA PASSED! T3-002 is fully verified.")
        return 0
    else:
        print("VERIFICATION FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
