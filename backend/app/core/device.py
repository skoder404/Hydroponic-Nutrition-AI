"""Device manager for runtime device detection and device honesty.

Adheres to TECH_RULES:
- Inference is CPU/GPU-agnostic via runtime detection (never assumed).
- Respects FORCE_CPU environment variable or config setting.
- Never claims GPU without runtime verification.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from backend.app.core.config import get_settings


class DeviceManager:
    """Manages execution device resolution with device honesty."""

    def __init__(self, force_cpu: bool | None = None) -> None:
        settings = get_settings()
        env_force = os.getenv("FORCE_CPU", "").strip().lower() in ("1", "true", "yes", "on")
        self.force_cpu: bool = force_cpu if force_cpu is not None else (settings.force_cpu or env_force)

    def is_cuda_available(self) -> bool:
        """Check if CUDA is truly available at runtime."""
        try:
            import torch
            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def get_device_str(self) -> str:
        """Return canonical device string: 'cuda:0' or 'cpu'."""
        if not self.force_cpu and self.is_cuda_available():
            return "cuda:0"
        return "cpu"

    def get_device_name(self) -> str:
        """Return human-readable device name."""
        if not self.force_cpu and self.is_cuda_available():
            try:
                import torch
                return str(torch.cuda.get_device_name(0))
            except Exception:  # noqa: BLE001
                return "CUDA GPU"
        return "CPU"

    def get_status(self) -> dict[str, Any]:
        """Return diagnostic status dictionary."""
        dev_str = self.get_device_str()
        cuda_avail = self.is_cuda_available()
        return {
            "target_device": dev_str,
            "device_name": self.get_device_name(),
            "cuda_available": cuda_avail,
            "force_cpu": self.force_cpu,
        }


@lru_cache(maxsize=1)
def get_device_manager() -> DeviceManager:
    """Return singleton DeviceManager instance."""
    return DeviceManager()
