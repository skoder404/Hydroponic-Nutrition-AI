"""Verification script for T6-003: DeviceManager runtime detection and OOM retry.

Verifies acceptance criteria:
1. FORCE_CPU=1 yields CPU.
2. Simulated OOM retries as documented.
3. /health shows device actually used.
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.core.device_manager import (
    DeviceManager,
    SimulatedCudaOOMError,
    get_device_manager,
)
from backend.app.main import create_app


def verify_force_cpu() -> None:
    # Test parameter
    dm_explicit = DeviceManager(force_cpu=True)
    assert dm_explicit.get_target_device() == "cpu", (
        "Explicit force_cpu must yield 'cpu'"
    )
    assert dm_explicit.is_cuda_available() is False

    # Test environment variable
    with patch.dict(os.environ, {"FORCE_CPU": "1"}):
        dm_env = DeviceManager()
        assert dm_env.get_target_device() == "cpu", (
            "FORCE_CPU=1 environment variable must yield 'cpu'"
        )
        assert dm_env.force_cpu is True

    print("FORCE_CPU=1 yielding CPU: VERIFIED.")


def verify_oom_retry_ladder() -> None:
    dm = DeviceManager(force_cpu=False)

    with patch.object(dm, "is_cuda_available", return_value=True):
        attempts = []

        def failing_gpu_work(device: str, batch: int) -> str:
            attempts.append((device, batch))
            if "cuda" in device:
                raise SimulatedCudaOOMError("Simulated GPU Out of Memory")
            return f"Processed on {device} with batch {batch}"

        _result, meta = dm.execute_with_retry(failing_gpu_work, initial_batch_size=4)

        assert attempts == [
            ("cuda:0", 4),
            ("cuda:0", 2),
            ("cpu", 2),
        ], f"Expected retry sequence on GPU smaller batch then CPU, got {attempts}"

        assert meta["fallback_to_cpu"] is True, "Fallback to CPU flag must be True"
        assert meta["retried_smaller_batch"] is True, (
            "Retried smaller batch flag must be True"
        )
        assert meta["device_used"] == "cpu", "Device used must be 'cpu'"
        assert dm.get_device_actually_used() == "cpu"

    print("Simulated OOM retry ladder (halved batch -> CPU fallback): VERIFIED.")


def verify_health_endpoint() -> None:
    dm = get_device_manager()
    dm.reset_device_tracking()

    client = TestClient(create_app())

    # 1. Before GPU computation: reports 'cpu'
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["device"] == "cpu", (
        f"Expected default device 'cpu', got {data.get('device')}"
    )

    # 2. After verified GPU tensor placement: reports 'cuda:0'
    dm.record_device_use("cuda:0")
    resp_gpu = client.get("/api/v1/health")
    assert resp_gpu.status_code == 200
    data_gpu = resp_gpu.json()
    assert data_gpu["device"] == "cuda:0", (
        f"Expected 'cuda:0', got {data_gpu.get('device')}"
    )

    # Cleanup
    dm.reset_device_tracking()
    print("/health endpoint reporting device actually used: VERIFIED.")


def main() -> int:
    print("=== Verifying T6-003: DeviceManager ===")
    verify_force_cpu()
    verify_oom_retry_ladder()
    verify_health_endpoint()
    print("=== All T6-003 Acceptance Criteria PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
