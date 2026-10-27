"""Device module forwarding to device_manager for backwards compatibility (T6-003)."""

from __future__ import annotations

from backend.app.core.device_manager import (
    DeviceManager,
    SimulatedCudaOOMError,
    get_device_manager,
    is_oom_exception,
)

__all__ = [
    "DeviceManager",
    "SimulatedCudaOOMError",
    "get_device_manager",
    "is_oom_exception",
]
