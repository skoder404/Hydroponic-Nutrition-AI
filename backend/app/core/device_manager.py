"""DeviceManager for runtime GPU/CPU detection, device honesty, and OOM retry (T6-003).

Features:
- Runtime CUDA detection without assuming hardware availability.
- Honors FORCE_CPU environment variable and config override.
- Exposes device name and VRAM / system memory metrics.
- Enforces Device Honesty: never reports GPU use unless a tensor was verified on GPU.
- Context helper / execution runner that intercepts CUDA OOM, runs VRAM cleanup,
  and executes the documented retry ladder: smaller batch -> CPU fallback.
"""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, TypeVar

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SimulatedCudaOOMError(RuntimeError):
    """Simulated CUDA Out-Of-Memory exception for deterministic testing."""


def is_oom_exception(exc: BaseException) -> bool:
    """Check if an exception represents a CUDA Out-Of-Memory condition."""
    if isinstance(exc, SimulatedCudaOOMError):
        return True

    # Check for PyTorch CUDA OutOfMemoryError if torch is installed
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass

    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


class DeviceManager:
    """Manages compute device detection, memory introspection, and OOM fallback."""

    def __init__(self, force_cpu: bool | None = None) -> None:
        settings = get_settings()
        env_val = os.getenv("FORCE_CPU", "").strip().lower()
        env_force = env_val in ("1", "true", "yes", "on")

        if force_cpu is not None:
            self.force_cpu: bool = force_cpu
        elif env_force:
            self.force_cpu = True
        else:
            self.force_cpu = bool(getattr(settings, "force_cpu", False))

        # Device honesty: track device where computation actually executed
        self._device_actually_used: str = "cpu"
        self._device_verified_on_gpu: bool = False

    def is_cuda_available(self) -> bool:
        """Check if CUDA is available at runtime."""
        if self.force_cpu:
            return False

        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def get_target_device(self) -> str:
        """Return canonical target device string: 'cuda:0' or 'cpu'."""
        if not self.force_cpu and self.is_cuda_available():
            return "cuda:0"
        return "cpu"

    def get_device_str(self) -> str:
        """Alias for get_target_device for backwards compatibility."""
        return self.get_target_device()

    def get_device_name(self) -> str:
        """Return human-readable device name."""
        if not self.force_cpu and self.is_cuda_available():
            try:
                import torch

                return str(torch.cuda.get_device_name(0))
            except Exception:  # noqa: BLE001
                return "CUDA GPU"
        return "CPU"

    def get_memory_info(self) -> dict[str, Any]:
        """Expose device memory metrics (VRAM if CUDA, system RAM if CPU)."""
        if not self.force_cpu and self.is_cuda_available():
            try:
                import torch

                dev = 0
                props = torch.cuda.get_device_properties(dev)
                allocated = torch.cuda.memory_allocated(dev)
                reserved = torch.cuda.memory_reserved(dev)
                total = props.total_memory

                return {
                    "device": f"cuda:{dev}",
                    "total_mb": round(total / (1024 * 1024), 2),
                    "allocated_mb": round(allocated / (1024 * 1024), 2),
                    "reserved_mb": round(reserved / (1024 * 1024), 2),
                    "free_mb": round((total - reserved) / (1024 * 1024), 2),
                }
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to inspect CUDA memory: %s", e)

        # CPU fallback metrics
        return {
            "device": "cpu",
            "total_mb": None,
            "allocated_mb": None,
            "reserved_mb": None,
            "free_mb": None,
        }

    # ==========================================================================
    # Device Honesty Tracking
    # ==========================================================================

    def record_device_use(self, device: str) -> None:
        """Record verified device placement.

        Strict Device Honesty invariant: Only marks GPU as used if an operation
        or tensor was explicitly confirmed on the GPU device.
        """
        device_lower = device.lower()
        if "cuda" in device_lower:
            self._device_actually_used = "cuda:0"
            self._device_verified_on_gpu = True
        else:
            self._device_actually_used = "cpu"

    def get_device_actually_used(self) -> str:
        """Return the device where tensors were actually verified to run."""
        return self._device_actually_used

    def reset_device_tracking(self) -> None:
        """Reset verified device tracking back to default CPU state."""
        self._device_actually_used = "cpu"
        self._device_verified_on_gpu = False

    def get_status(self) -> dict[str, Any]:
        """Return comprehensive diagnostics dictionary."""
        return {
            "target_device": self.get_target_device(),
            "device_actually_used": self.get_device_actually_used(),
            "device_name": self.get_device_name(),
            "cuda_available": self.is_cuda_available(),
            "force_cpu": self.force_cpu,
            "memory": self.get_memory_info(),
        }

    # ==========================================================================
    # OOM Retry Ladder & Context Helper
    # ==========================================================================

    def cleanup_memory(self) -> None:
        """Perform garbage collection and flush CUDA cache."""
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @contextmanager
    def retry_on_oom(
        self,
        on_fallback: Callable[[str], None] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Context manager monitoring execution for CUDA OOM.

        Yields a control dict that can be inspected by the caller. If a CUDA OOM
        is intercepted, memory is cleaned up and fallback is signaled.
        """
        state = {
            "device": self.get_target_device(),
            "fallback_triggered": False,
            "oom_caught": False,
        }
        try:
            yield state
        except BaseException as exc:
            if is_oom_exception(exc):
                state["oom_caught"] = True
                state["fallback_triggered"] = True
                logger.warning(
                    "CUDA OOM caught in retry context. Running memory cleanup..."
                )
                self.cleanup_memory()
                if on_fallback:
                    on_fallback("cpu")
                self.record_device_use("cpu")
                try:
                    from backend.app.core.telemetry import get_fallback_tracker

                    get_fallback_tracker().increment_gpu_to_cpu(
                        reason="CUDA OOM in retry_on_oom"
                    )
                except Exception:
                    pass
            raise

    def execute_with_retry(
        self,
        fn: Callable[[str, int], T],
        initial_batch_size: int = 1,
    ) -> tuple[T, dict[str, Any]]:
        """Execute a workload with the documented OOM retry ladder.

        Ladder:
        1. Attempt on target device (e.g. cuda:0) with initial_batch_size.
        2. On CUDA OOM: clean memory, retry on GPU with halved batch size (if initial_batch_size > 1).
        3. On second CUDA OOM (or initial_batch_size == 1): clean memory, fallback to CPU.
        4. Record device actually used upon successful execution.

        Args:
            fn: Callable receiving (device_str, batch_size) returning result.
            initial_batch_size: Starting batch size (defaults to 1).

        Returns:
            Tuple of (result, execution_metadata_dict).
        """
        target_device = self.get_target_device()
        current_device = target_device
        current_batch = initial_batch_size
        retried_smaller_batch = False
        fallback_to_cpu = False

        while True:
            try:
                result = fn(current_device, current_batch)
                # Successful execution -> record verified device
                self.record_device_use(current_device)
                meta = {
                    "device_used": current_device,
                    "batch_size_used": current_batch,
                    "fallback_to_cpu": fallback_to_cpu,
                    "retried_smaller_batch": retried_smaller_batch,
                }
                return result, meta

            except BaseException as exc:
                if not is_oom_exception(exc):
                    raise

                logger.warning(
                    "CUDA OOM on device=%s, batch_size=%d. Error: %s",
                    current_device,
                    current_batch,
                    exc,
                )
                self.cleanup_memory()

                # Step 2: Retry with smaller batch on GPU if batch_size > 1 and not yet retried
                if (
                    current_device != "cpu"
                    and current_batch > 1
                    and not retried_smaller_batch
                ):
                    current_batch = max(1, current_batch // 2)
                    retried_smaller_batch = True
                    logger.info(
                        "Retrying on %s with smaller batch_size=%d",
                        current_device,
                        current_batch,
                    )
                    continue

                # Step 3: Fallback to CPU with batch_size 1
                if current_device != "cpu":
                    current_device = "cpu"
                    current_batch = max(1, current_batch)
                    fallback_to_cpu = True
                    try:
                        from backend.app.core.telemetry import get_fallback_tracker

                        get_fallback_tracker().increment_gpu_to_cpu(
                            reason="CUDA OOM in execute_with_retry"
                        )
                    except Exception:
                        pass
                    logger.info("Retrying on CPU fallback...")
                    continue

                # If CPU also fails with OOM, raise error
                raise


@lru_cache(maxsize=1)
def get_device_manager() -> DeviceManager:
    """Return the application singleton DeviceManager instance."""
    return DeviceManager()
