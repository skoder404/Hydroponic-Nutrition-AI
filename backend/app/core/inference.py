"""Inference runner providing off-threadpool execution, timeouts, and concurrency limit (T6-015).

Invariants:
- Heavy ML and vision tasks run in dedicated worker threads to keep FastAPI event loop responsive.
- Concurrency is bounded by an asyncio Semaphore (configured via `settings.max_concurrent_inference`).
- Execution is strictly bounded by request timeout (configured via `settings.inference_timeout_s`),
  returning a structured HTTP 504 / 503 error envelope upon timeout.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from starlette.concurrency import run_in_threadpool

from backend.app.core.config import get_settings
from backend.app.core.errors import InferenceTimeoutError, ServiceUnavailableError

logger = logging.getLogger(__name__)

T = TypeVar("T")

_semaphore: asyncio.Semaphore | None = None
_semaphore_limit: int | None = None


def get_inference_semaphore() -> asyncio.Semaphore:
    """Return singleton asyncio.Semaphore for bounded concurrency."""
    global _semaphore, _semaphore_limit
    settings = get_settings()
    current_limit = settings.max_concurrent_inference

    if _semaphore is None or _semaphore_limit != current_limit:
        _semaphore = asyncio.Semaphore(current_limit)
        _semaphore_limit = current_limit

    return _semaphore


async def run_blocking_inference(
    func: Callable[..., T],
    *args: Any,
    timeout_s: float | None = None,
    **kwargs: Any,
) -> T:
    """Execute a blocking inference callable off the event loop with concurrency limit and timeout.

    Args:
        func: Synchronous callable (e.g. ImageService.predict or SensorService.predict).
        *args: Positional arguments to forward to func.
        timeout_s: Optional timeout in seconds. If None, defaults to settings.inference_timeout_s.
        **kwargs: Keyword arguments to forward to func.

    Returns:
        The return value of func(*args, **kwargs).

    Raises:
        InferenceTimeoutError: If execution exceeds deadline (HTTP 504).
        ServiceUnavailableError: If concurrency queue is exhausted or system is saturated.
    """
    settings = get_settings()
    effective_timeout = timeout_s if timeout_s is not None else settings.inference_timeout_s
    sem = get_inference_semaphore()

    try:
        # Acquire slot with timeout
        await asyncio.wait_for(sem.acquire(), timeout=effective_timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "Inference queue saturated: failed to acquire worker slot within %s seconds",
            effective_timeout,
        )
        raise ServiceUnavailableError(
            message=f"Inference service capacity exceeded. Max {settings.max_concurrent_inference} concurrent tasks allowed.",
            details={"max_concurrent": settings.max_concurrent_inference},
        )

    try:
        coro = run_in_threadpool(func, *args, **kwargs)
        result = await asyncio.wait_for(coro, timeout=effective_timeout)
        return result
    except asyncio.TimeoutError as exc:
        logger.warning(
            "Inference execution exceeded deadline of %s seconds: %s",
            effective_timeout,
            func.__name__ if hasattr(func, "__name__") else str(func),
        )
        raise InferenceTimeoutError(
            message=f"Inference execution timed out after {effective_timeout:.1f} seconds.",
            details={"timeout_seconds": effective_timeout},
        ) from exc
    finally:
        sem.release()

