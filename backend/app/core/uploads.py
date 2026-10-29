"""Server-side upload validation utility and secure file persistence (T3-002).

Enforces strict defense-in-depth security on uploaded plant photographs:
1. Content-Type and Magic Bytes Verification (JPEG, PNG, WEBP).
2. Rejection of renamed non-image files (e.g. text/script files disguised as .jpg).
3. Hard file-size limits (PAYLOAD_TOO_LARGE, HTTP 413) via MAX_UPLOAD_MB.
4. PIL decoding, decompression bomb guards, and dimension bounds.
5. Safe EXIF orientation normalization (ImageOps.exif_transpose) and metadata stripping.
6. Path traversal immunity: client filenames are never used; files are persisted under
   cryptographically unique UUIDs in the trusted uploads directory.
"""

from __future__ import annotations

import io
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.datastructures import UploadFile as StarletteUploadFile

from backend.app.core.config import get_settings
from backend.app.core.errors import (
    InvalidInputError,
    PayloadTooLargeError,
    UnsupportedMediaError,
)

logger = logging.getLogger(__name__)

# Allowed MIME types and mapping to standard extensions
ALLOWED_MIME_TYPES: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

# Magic bytes signatures for supported image formats
# JPEG: starts with FF D8 FF
# PNG: starts with 89 50 4E 47 0D 0A 1A 0A
# WEBP: RIFF....WEBP (RIFF at 0..4, WEBP at 8..12)
MAGIC_SIGNATURES: list[tuple[bytes, str, str]] = [
    (b"\xff\xd8\xff", "image/jpeg", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png", "png"),
]

# Dimension and pixel bounds to defend against decompression bombs and corrupt frames
DEFAULT_MIN_DIMENSION: int = 16
DEFAULT_MAX_DIMENSION: int = 10000
DEFAULT_MAX_PIXELS: int = 50_000_000  # 50 Megapixels max


@dataclass(frozen=True)
class ValidatedUpload:
    """Immutable metadata and path descriptor for a securely validated image upload."""

    file_id: str
    saved_path: Path
    original_filename: str | None
    content_type: str
    format: str
    extension: str
    size_bytes: int
    width: int
    height: int


def detect_image_format(data: bytes) -> tuple[str, str]:
    """Detect image format and MIME type using leading magic bytes.

    Args:
        data: Raw file byte prefix (at least 16 bytes).

    Returns:
        Tuple of (mime_type, format_name) e.g. ('image/jpeg', 'jpeg').

    Raises:
        UnsupportedMediaError: If bytes do not match JPEG, PNG, or WEBP signatures.
    """
    if len(data) < 12:
        raise UnsupportedMediaError(
            message="Uploaded file is too small or truncated to be a valid image.",
            details={"received_bytes": len(data)},
        )

    # Check JPEG and PNG exact prefixes
    for signature, mime, fmt in MAGIC_SIGNATURES:
        if data.startswith(signature):
            return mime, fmt

    # Check WEBP signature: 'RIFF' at 0..4 and 'WEBP' at 8..12
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp", "webp"

    # Magic bytes check failed - renamed non-image file or unsupported format
    raise UnsupportedMediaError(
        message="Unsupported media type or corrupt image signature. Allowed formats: JPEG, PNG, WEBP.",
        details={"header_hex": data[:12].hex()},
    )


def validate_content_type(claimed_content_type: str | None, detected_mime: str) -> None:
    """Verify that client-provided Content-Type matches the detected binary format.

    Args:
        claimed_content_type: Value from HTTP Content-Type header or UploadFile.
        detected_mime: Format detected from file magic bytes.

    Raises:
        UnsupportedMediaError: If content-type is disallowed or conflicts with magic bytes.
    """
    if claimed_content_type is None:
        return

    clean_content_type = claimed_content_type.split(";")[0].strip().lower()

    if clean_content_type not in ALLOWED_MIME_TYPES:
        raise UnsupportedMediaError(
            message=f"Unsupported Content-Type '{clean_content_type}'. Allowed: image/jpeg, image/png, image/webp.",
            details={
                "claimed_content_type": claimed_content_type,
                "allowed_types": list(ALLOWED_MIME_TYPES.keys()),
            },
        )

    # Normalize jpg vs jpeg
    norm_claimed = (
        "image/jpeg"
        if clean_content_type in ("image/jpg", "image/jpeg")
        else clean_content_type
    )
    norm_detected = (
        "image/jpeg" if detected_mime in ("image/jpg", "image/jpeg") else detected_mime
    )

    if norm_claimed != norm_detected:
        raise UnsupportedMediaError(
            message=f"Content-Type mismatch: Header states '{clean_content_type}', but magic bytes indicate '{detected_mime}'.",
            details={
                "claimed_content_type": clean_content_type,
                "detected_mime": detected_mime,
            },
        )


def validate_and_save_image(
    file_bytes: bytes,
    original_filename: str | None = None,
    content_type: str | None = None,
    max_upload_mb: int | None = None,
    uploads_dir: Path | str | None = None,
    min_dimension: int = DEFAULT_MIN_DIMENSION,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> ValidatedUpload:
    """Validate image bytes, normalize EXIF orientation, and save to a secure UUID path.

    Args:
        file_bytes: Complete raw file byte buffer.
        original_filename: Client filename for provenance logging (never used for path).
        content_type: Client-provided HTTP Content-Type header.
        max_upload_mb: Size threshold in megabytes. Defaults to Settings.max_upload_mb (15 MB).
        uploads_dir: Target directory on disk. Defaults to 'uploads/'.
        min_dimension: Minimum allowed width and height in pixels.
        max_dimension: Maximum allowed width and height in pixels.
        max_pixels: Maximum allowed total pixels (width * height).

    Returns:
        ValidatedUpload dataclass with path, ID, and physical image dimensions.

    Raises:
        PayloadTooLargeError: If file_bytes exceeds max_upload_mb (HTTP 413).
        UnsupportedMediaError: If magic bytes or MIME type is not JPEG, PNG, WEBP (HTTP 415).
        InvalidInputError: If file is zero bytes, corrupted, or dimensions exceed bounds (HTTP 422).
    """
    settings = get_settings()
    effective_max_mb = (
        max_upload_mb if max_upload_mb is not None else settings.max_upload_mb
    )
    max_bytes = effective_max_mb * 1024 * 1024

    # 1. Size Enforcement (Payload Too Large - HTTP 413)
    file_size = len(file_bytes)
    if file_size > max_bytes:
        size_mb = round(file_size / (1024 * 1024), 2)
        raise PayloadTooLargeError(
            message=f"File size ({size_mb} MB) exceeds maximum allowed limit of {effective_max_mb} MB.",
            details={
                "received_bytes": file_size,
                "max_allowed_bytes": max_bytes,
                "max_allowed_mb": effective_max_mb,
            },
        )

    # 2. Non-empty check
    if file_size == 0:
        raise InvalidInputError(
            message="Uploaded file is empty (0 bytes).",
            details={"original_filename": original_filename},
        )

    # 3. Magic Bytes Detection (Unsupported Media - HTTP 415)
    detected_mime, format_name = detect_image_format(file_bytes)

    # 4. Content-Type Header Verification
    validate_content_type(content_type, detected_mime)

    # 5. Decode Image with PIL (Decompression & Corruption Guard - HTTP 422)
    try:
        # Prevent PIL decompression bombs
        Image.MAX_IMAGE_PIXELS = max_pixels

        stream = io.BytesIO(file_bytes)
        with Image.open(stream) as img:
            # Force decoding of raster data to verify integrity
            img.load()

            # 6. Dimension Bounds Verification
            width, height = img.size
            total_pixels = width * height

            if width < min_dimension or height < min_dimension:
                raise InvalidInputError(
                    message=f"Image dimensions ({width}x{height}) are smaller than minimum allowed ({min_dimension}x{min_dimension}).",
                    details={
                        "width": width,
                        "height": height,
                        "min_dimension": min_dimension,
                    },
                )

            if width > max_dimension or height > max_dimension:
                raise InvalidInputError(
                    message=f"Image dimensions ({width}x{height}) exceed maximum allowed dimension ({max_dimension}x{max_dimension}).",
                    details={
                        "width": width,
                        "height": height,
                        "max_dimension": max_dimension,
                    },
                )

            if total_pixels > max_pixels:
                raise InvalidInputError(
                    message=f"Image total pixels ({total_pixels}) exceed maximum allowed limit ({max_pixels}).",
                    details={"total_pixels": total_pixels, "max_pixels": max_pixels},
                )

            # 7. Safe EXIF Orientation Transpose & Metadata Stripping
            # Rotates the pixel matrix according to EXIF orientation tag (if present)
            transposed_img = ImageOps.exif_transpose(img)
            if transposed_img is None:
                transposed_img = img

            norm_width, norm_height = transposed_img.size

            # Ensure image mode is RGB for JPEG/WEBP or RGBA for PNG
            if format_name in ("jpeg", "jpg") and transposed_img.mode != "RGB":
                transposed_img = transposed_img.convert("RGB")
            elif format_name == "png" and transposed_img.mode not in (
                "RGB",
                "RGBA",
                "L",
            ):
                transposed_img = transposed_img.convert("RGBA")

            # 8. Path Traversal Immunity: Generate Cryptographic UUID Filename
            # Never use original_filename in target path
            file_id = str(uuid.uuid4())
            extension = ALLOWED_MIME_TYPES[detected_mime]
            safe_filename = f"{file_id}{extension}"

            target_dir = (
                Path(uploads_dir) if uploads_dir is not None else Path("uploads")
            ).resolve()
            target_dir.mkdir(parents=True, exist_ok=True)

            target_path = (target_dir / safe_filename).resolve()

            # Verify target path lies strictly inside target_dir (guarantees traversal immunity)
            try:
                target_path.relative_to(target_dir)
            except ValueError as err:
                raise InvalidInputError(
                    message="Target storage path escapes designated upload directory.",
                    details={"target_path": str(target_path)},
                ) from err

            # 9. Save Sanitized Image (EXIF stripped)
            save_format = "JPEG" if format_name == "jpeg" else format_name.upper()
            transposed_img.save(target_path, format=save_format, quality=95)

            logger.info(
                f"Successfully validated and persisted upload {file_id}{extension} ({norm_width}x{norm_height})",
                extra={
                    "event": "upload_validated",
                    "file_id": file_id,
                    "format": format_name,
                    "width": norm_width,
                    "height": norm_height,
                    "size_bytes": file_size,
                },
            )

            return ValidatedUpload(
                file_id=file_id,
                saved_path=target_path,
                original_filename=original_filename,
                content_type=detected_mime,
                format=format_name,
                extension=extension,
                size_bytes=file_size,
                width=norm_width,
                height=norm_height,
            )

    except UnidentifiedImageError as err:
        raise UnsupportedMediaError(
            message="File could not be identified as a supported image format.",
            details={"original_filename": original_filename},
        ) from err
    except (OSError, SyntaxError) as err:
        raise InvalidInputError(
            message="Corrupted or invalid image data stream.",
            details={"original_filename": original_filename, "error": str(err)},
        ) from err


async def validate_upload_file(
    upload_file: UploadFile | StarletteUploadFile,
    max_upload_mb: int | None = None,
    uploads_dir: Path | str | None = None,
) -> ValidatedUpload:
    """Asynchronously stream and validate a FastAPI UploadFile with chunked size limits.

    Aborts reading as soon as chunked byte accumulation exceeds max_upload_mb,
    preventing denial-of-service memory exhaustion attacks.

    Args:
        upload_file: Incoming FastAPI UploadFile instance.
        max_upload_mb: Optional size cap in MB.
        uploads_dir: Optional destination directory override.

    Returns:
        ValidatedUpload dataclass with saved file details.
    """
    settings = get_settings()
    effective_max_mb = (
        max_upload_mb if max_upload_mb is not None else settings.max_upload_mb
    )
    max_bytes = effective_max_mb * 1024 * 1024

    chunk_size = 64 * 1024  # 64 KB chunks
    accumulated = bytearray()

    while True:
        chunk = await upload_file.read(chunk_size)
        if not chunk:
            break
        accumulated.extend(chunk)
        if len(accumulated) > max_bytes:
            # Early abort before loading full gigabytes into memory
            raise PayloadTooLargeError(
                message=f"Upload exceeded maximum size limit of {effective_max_mb} MB.",
                details={
                    "max_allowed_mb": effective_max_mb,
                    "accumulated_bytes": len(accumulated),
                },
            )

    return validate_and_save_image(
        file_bytes=bytes(accumulated),
        original_filename=upload_file.filename,
        content_type=upload_file.content_type,
        max_upload_mb=effective_max_mb,
        uploads_dir=uploads_dir,
    )
