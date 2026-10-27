"""Core configuration, device management, and logging utilities."""

from backend.app.core.config import Settings, get_settings
from backend.app.core.device import DeviceManager, get_device_manager
from backend.app.core.errors import (
    AppException,
    InvalidInputError,
    PayloadTooLargeError,
    UnsupportedMediaError,
)
from backend.app.core.logging import (
    JSONFormatter,
    redact_secrets,
    request_id_ctx_var,
    setup_logging,
)
from backend.app.core.paths import (
    UntrustedModelPathError,
    get_trusted_model_root,
    safe_load_model_artifact,
    validate_model_path,
)
from backend.app.core.uploads import (
    ValidatedUpload,
    detect_image_format,
    validate_and_save_image,
    validate_upload_file,
)

__all__ = [
    "AppException",
    "DeviceManager",
    "InvalidInputError",
    "JSONFormatter",
    "PayloadTooLargeError",
    "Settings",
    "UnsupportedMediaError",
    "UntrustedModelPathError",
    "ValidatedUpload",
    "detect_image_format",
    "get_device_manager",
    "get_settings",
    "get_trusted_model_root",
    "redact_secrets",
    "request_id_ctx_var",
    "safe_load_model_artifact",
    "setup_logging",
    "validate_and_save_image",
    "validate_model_path",
    "validate_upload_file",
]
