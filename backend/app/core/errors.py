"""Shared error envelope, typed exceptions, and FastAPI exception handlers (T3-004, T3-002).

Implements the single error envelope specified in TECH_RULES:
{
    "error": {
        "code": "PAYLOAD_TOO_LARGE",
        "message": "...",
        "details": {...},
        "retryable": false
    }
}
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class AppException(Exception):
    """Base application exception conforming to the unified error envelope."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.retryable = retryable

    def to_envelope(self) -> dict[str, Any]:
        """Format the exception into the project standard error envelope."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "retryable": self.retryable,
            }
        }


# ------------------------------------------------------------------------------
# Concrete Typed Exceptions matching TECH_RULES Status Code Table
# ------------------------------------------------------------------------------


class UnsupportedMediaError(AppException):
    """Raised when an uploaded file has an unsupported format or invalid magic bytes (HTTP 415)."""

    def __init__(
        self,
        message: str = "Unsupported media type. Allowed formats: JPEG, PNG, WEBP.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="UNSUPPORTED_MEDIA",
            message=message,
            status_code=415,
            details=details,
            retryable=False,
        )


class PayloadTooLargeError(AppException):
    """Raised when an upload exceeds the maximum allowed file size (HTTP 413)."""

    def __init__(
        self,
        message: str = "Payload exceeds maximum allowed upload size.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="PAYLOAD_TOO_LARGE",
            message=message,
            status_code=413,
            details=details,
            retryable=False,
        )


class InvalidInputError(AppException):
    """Raised when request payload or image content fails validation or decoding (HTTP 422)."""

    def __init__(
        self,
        message: str = "Invalid input data.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="INVALID_INPUT",
            message=message,
            status_code=422,
            details=details,
            retryable=False,
        )


class MissingFieldError(AppException):
    """Raised when a required sensor or payload field is missing (HTTP 422)."""

    def __init__(
        self,
        message: str = "Required field is missing.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="MISSING_FIELD",
            message=message,
            status_code=422,
            details=details,
            retryable=False,
        )


class SegmentationFailedError(AppException):
    """Raised when SAM 2.1 or classical leaf segmentation fails (HTTP 422, retryable=True)."""

    def __init__(
        self,
        message: str = "Leaf segmentation failed. Please provide a clearer plant photograph.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="SEGMENTATION_FAILED",
            message=message,
            status_code=422,
            details=details,
            retryable=True,
        )


class SchemaMismatchError(AppException):
    """Raised when input feature vector or model schema does not align with runtime (HTTP 409)."""

    def __init__(
        self,
        message: str = "Feature schema mismatch between request and active model.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="SCHEMA_MISMATCH",
            message=message,
            status_code=409,
            details=details,
            retryable=False,
        )


class ModelUnavailableError(AppException):
    """Raised when no promoted model is registered or weights are corrupt (HTTP 503)."""

    def __init__(
        self,
        message: str = "Requested model artifact is unavailable.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="MODEL_UNAVAILABLE",
            message=message,
            status_code=503,
            details=details,
            retryable=False,
        )


class DeviceUnavailableError(AppException):
    """Raised when campus hardware is unreachable or sensor telemetry is stale (HTTP 503, retryable=True)."""

    def __init__(
        self,
        message: str = "Campus device server is disconnected or telemetry is stale.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="DEVICE_UNAVAILABLE",
            message=message,
            status_code=503,
            details=details,
            retryable=True,
        )


class PersistenceFailedError(AppException):
    """Raised when database storage fails (HTTP 503, response states not saved)."""

    def __init__(
        self,
        message: str = "Database persistence failed. Result was not saved.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="PERSISTENCE_FAILED",
            message=message,
            status_code=503,
            details=details,
            retryable=False,
        )


# ------------------------------------------------------------------------------
# FastAPI Exception Handlers
# ------------------------------------------------------------------------------


async def app_exception_handler(_request: Request, exc: Any) -> JSONResponse:
    """Handle custom application exceptions and format into standard error envelope."""
    if isinstance(exc, AppException):
        return JSONResponse(status_code=exc.status_code, content=exc.to_envelope())
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(exc),
                "details": {},
                "retryable": False,
            }
        },
    )


async def starlette_http_exception_handler(_request: Request, exc: Any) -> JSONResponse:
    """Translate standard Starlette / FastAPI HTTPExceptions to the unified envelope."""
    # Map common status codes to standard error codes
    code_map = {
        400: "INVALID_INPUT",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        413: "PAYLOAD_TOO_LARGE",
        415: "UNSUPPORTED_MEDIA",
        422: "INVALID_INPUT",
        500: "INTERNAL_ERROR",
        503: "SERVICE_UNAVAILABLE",
    }
    status = getattr(exc, "status_code", 500)
    detail = getattr(exc, "detail", str(exc))
    code = code_map.get(status, "HTTP_ERROR")
    envelope = {
        "error": {
            "code": code,
            "message": str(detail),
            "details": {},
            "retryable": False,
        }
    }
    return JSONResponse(status_code=status, content=envelope)


async def validation_exception_handler(_request: Request, exc: Any) -> JSONResponse:
    """Format Pydantic / FastAPI request validation errors into unified error envelope."""
    errors = exc.errors() if hasattr(exc, "errors") else []
    # Check if any error is a missing field error
    has_missing = any(err.get("type") == "missing" for err in errors)
    code = "MISSING_FIELD" if has_missing else "INVALID_INPUT"

    # Simplify details to avoid leaking internal Pydantic internals
    details: dict[str, Any] = {
        "validation_errors": [
            {
                "field": ".".join(str(loc) for loc in err.get("loc", [])),
                "message": err.get("msg", ""),
                "type": err.get("type", ""),
            }
            for err in errors
        ]
    }

    envelope = {
        "error": {
            "code": code,
            "message": "Validation failed for request parameters.",
            "details": details,
            "retryable": False,
        }
    }
    return JSONResponse(status_code=422, content=envelope)


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unhandled exceptions: log request_id, never leak paths or stack traces."""
    req_id = getattr(request.state, "request_id", "unknown")
    logger.exception(
        "Unhandled exception processing request",
        extra={"request_id": req_id, "path": request.url.path, "error": str(exc)},
    )
    envelope = {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "An unexpected internal server error occurred.",
            "details": {"request_id": req_id},
            "retryable": False,
        }
    }
    return JSONResponse(status_code=500, content=envelope)


def register_error_handlers(app: FastAPI) -> None:
    """Register all unified exception handlers on a FastAPI application instance."""
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(StarletteHTTPException, starlette_http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
