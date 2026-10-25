"""FastAPI application factory, middleware, and router setup.

Features:
- create_app() factory
- Structured JSON logging setup
- Request-ID tracking middleware (header X-Request-ID + contextvar + log injection)
- /api/v1 router mounting
"""
from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.api.v1 import api_v1_router
from backend.app.core.config import get_settings
from backend.app.core.logging import JSONFormatter, request_id_ctx_var, setup_logging

logger = logging.getLogger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware to inject, track, and log request_id across all HTTP requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract existing X-Request-ID or generate new UUID
        req_id = (
            request.headers.get("X-Request-ID")
            or request.headers.get("X-Request-Id")
            or str(uuid.uuid4())
        )

        token = request_id_ctx_var.set(req_id)
        request.state.request_id = req_id

        t0 = time.perf_counter()
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={"method": request.method, "path": request.url.path, "request_id": req_id},
        )

        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                f"Unhandled error during request: {request.method} {request.url.path}",
                extra={"method": request.method, "path": request.url.path, "request_id": req_id},
            )
            raise
        finally:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            request_id_ctx_var.reset(token)

        # Inject X-Request-ID into response headers
        response.headers["X-Request-ID"] = req_id

        logger.info(
            f"Request completed: {request.method} {request.url.path} status={response.status_code} latency={latency_ms}ms",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "request_id": req_id,
            },
        )

        return response


def create_app(configure_logging: bool = True) -> FastAPI:
    """Factory creating and configuring the FastAPI application instance."""
    settings = get_settings()

    # Configure structured JSON logging if not already configured with JSONFormatter
    if configure_logging:
        root_logger = logging.getLogger()
        has_json_formatter = any(
            isinstance(getattr(h, "formatter", None), JSONFormatter)
            for h in root_logger.handlers
        )
        if not has_json_formatter:
            setup_logging(log_level="INFO")

    app = FastAPI(
        title="Hydroponic Nutrition AI API",
        version="0.1.0",
        description="Hydroponic diagnostic and nutrition recommendation system API.",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Middleware registration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIdMiddleware)

    # Route mounting
    app.include_router(api_v1_router, prefix="/api/v1")

    logger.info("FastAPI application initialized successfully", extra={"event": "app_startup"})
    return app


# Module-level instance for uvicorn run
app = create_app()
