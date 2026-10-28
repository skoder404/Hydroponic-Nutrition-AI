"""Per-IP sliding window rate limiting middleware and manager (T3-006).

Implements per-IP rate limits with configurable thresholds via environment variables,
stricter limits on inference endpoints (/predict/image, /analyze), exemption for health checks,
and returns HTTP 429 in the unified error envelope upon exceeding quotas.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Callable
from typing import Any, ClassVar

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

# Unit multipliers for parsing rate limit expressions
TIME_MULTIPLIERS = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "m": 60.0,
    "min": 60.0,
    "minute": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
}


def parse_rate_limit(expr: str | int) -> tuple[int, float]:
    """Parse a rate limit expression like '60/minute', '10/second', or integer RPM.

    Returns:
        tuple of (limit_count, window_seconds).
    """
    if isinstance(expr, int):
        return expr, 60.0

    expr_clean = expr.strip().lower()
    if "/" not in expr_clean:
        try:
            return int(expr_clean), 60.0
        except ValueError as exc:
            raise ValueError(f"Invalid rate limit expression: {expr}") from exc

    parts = expr_clean.split("/", 1)
    try:
        limit = int(parts[0].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid rate limit number: {parts[0]}") from exc

    unit_raw = parts[1].strip()
    match = re.match(r"^(\d+)?\s*([a-z]+)$", unit_raw)
    if not match:
        raise ValueError(f"Unrecognized time unit in rate limit expression: {unit_raw}")

    scale = int(match.group(1)) if match.group(1) else 1
    unit_str = match.group(2)

    if unit_str not in TIME_MULTIPLIERS:
        raise ValueError(
            f"Unsupported time unit '{unit_str}'. Choose from: second, minute, hour, day"
        )

    window_s = scale * TIME_MULTIPLIERS[unit_str]
    return limit, window_s


class RateLimiter:
    """Thread-safe in-memory sliding-window log rate limiter."""

    EXEMPT_PATHS: ClassVar[set[str]] = {
        "/health",
        "/api/v1/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    STRICT_PATTERNS = (
        "/predict/image",
        "/predict/sensor",
        "/analyze",
    )

    def __init__(self) -> None:
        self._history: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_exempt(self, path: str) -> bool:
        """Check if request path is exempt from rate limiting."""
        normalized = path.rstrip("/")
        return normalized in self.EXEMPT_PATHS or any(
            normalized.endswith(suffix)
            for suffix in ("/health", "/docs", "/redoc", "/openapi.json")
        )

    def is_strict(self, path: str) -> bool:
        """Check if request path requires stricter ML inference limits."""
        normalized = path.rstrip("/")
        return any(
            pattern in normalized or normalized.endswith(pattern)
            for pattern in self.STRICT_PATTERNS
        )

    def check(self, ip: str, path: str) -> tuple[bool, int, float, float]:
        """Check if the request from given IP to path should be allowed.

        Returns:
            (allowed: bool, limit: int, window_s: float, retry_after_s: float)
        """
        settings = get_settings()

        if not settings.rate_limit_enabled or self.is_exempt(path):
            return True, 0, 0.0, 0.0

        if self.is_strict(path):
            tier = "strict"
            expr = settings.rate_limit_strict
        else:
            tier = "default"
            expr = settings.rate_limit_default

        limit, window_s = parse_rate_limit(expr)
        key = f"{ip}:{tier}"
        now = time.monotonic()
        cutoff = now - window_s

        with self._lock:
            timestamps = self._history.get(key, [])
            # Evict timestamps outside sliding window
            timestamps = [ts for ts in timestamps if ts > cutoff]

            if len(timestamps) >= limit:
                retry_after_s = max(1.0, timestamps[0] + window_s - now)
                self._history[key] = timestamps
                return False, limit, window_s, retry_after_s

            timestamps.append(now)
            self._history[key] = timestamps
            return True, limit, window_s, 0.0

    def reset(self) -> None:
        """Clear all recorded rate limit state."""
        with self._lock:
            self._history.clear()


# Global module singleton
_limiter_instance = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Return the global RateLimiter instance."""
    return _limiter_instance


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI / Starlette middleware enforcing per-IP sliding window rate limits."""

    def __init__(self, app: Any, limiter: RateLimiter | None = None) -> None:
        super().__init__(app)
        self.limiter = limiter or get_rate_limiter()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        settings = get_settings()

        if not settings.rate_limit_enabled:
            return await call_next(request)

        # Resolve client IP (handle reverse-proxy X-Forwarded-For)
        client_ip = "127.0.0.1"
        x_forwarded_for = request.headers.get("X-Forwarded-For")
        if x_forwarded_for:
            client_ip = x_forwarded_for.split(",")[0].strip()
        elif request.client and request.client.host:
            client_ip = request.client.host

        allowed, limit, window_s, retry_after_s = self.limiter.check(
            client_ip, request.url.path
        )

        if not allowed:
            retry_sec = math.ceil(retry_after_s)
            logger.warning(
                f"Rate limit exceeded: IP {client_ip} on {request.url.path} (limit={limit}/{int(window_s)}s)",
                extra={
                    "client_ip": client_ip,
                    "path": request.url.path,
                    "limit": limit,
                    "window_s": window_s,
                    "retry_after_s": retry_sec,
                },
            )

            envelope = {
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded. Maximum {limit} requests per {int(window_s)} seconds allowed. Please retry after {retry_sec} seconds.",
                    "details": {
                        "limit": limit,
                        "window_seconds": int(window_s),
                        "retry_after_seconds": retry_sec,
                        "path": request.url.path,
                        "client_ip": client_ip,
                    },
                    "retryable": False,
                }
            }
            return JSONResponse(
                status_code=429,
                content=envelope,
                headers={
                    "Retry-After": str(retry_sec),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(retry_sec),
                },
            )

        response = await call_next(request)
        if limit > 0:
            response.headers["X-RateLimit-Limit"] = str(limit)
        return response
