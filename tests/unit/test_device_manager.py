"""Unit tests for DeviceManager runtime detection, device honesty, and OOM fallback (T6-003)."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.core.device_manager import (
    DeviceManager,
    SimulatedCudaOOMError,
    get_device_manager,
)
from backend.app.main import create_app


def test_force_cpu_explicit_flag():
    dm = DeviceManager(force_cpu=True)
    assert dm.force_cpu is True
    assert dm.get_target_device() == "cpu"
    assert dm.get_device_str() == "cpu"
    assert dm.is_cuda_available() is False
    assert dm.get_device_name() == "CPU"


def test_force_cpu_env_variable():
    with patch.dict(os.environ, {"FORCE_CPU": "1"}):
        dm = DeviceManager()
        assert dm.force_cpu is True
        assert dm.get_target_device() == "cpu"

    with patch.dict(os.environ, {"FORCE_CPU": "true"}):
        dm = DeviceManager()
        assert dm.force_cpu is True
        assert dm.get_target_device() == "cpu"


def test_device_honesty_tracking():
    dm = DeviceManager(force_cpu=True)
    # Default state before any tensor verification is strictly 'cpu'
    assert dm.get_device_actually_used() == "cpu"

    # Recording GPU placement updates state
    dm.record_device_use("cuda:0")
    assert dm.get_device_actually_used() == "cuda:0"

    # Reset tracking
    dm.reset_device_tracking()
    assert dm.get_device_actually_used() == "cpu"


def test_memory_info_structure():
    dm = DeviceManager(force_cpu=True)
    mem = dm.get_memory_info()
    assert isinstance(mem, dict)
    assert "device" in mem
    assert mem["device"] == "cpu"


def test_execute_with_retry_success_without_oom():
    dm = DeviceManager(force_cpu=True)

    def work(device: str, batch: int):
        return f"processed on {device} with batch {batch}"

    result, meta = dm.execute_with_retry(work, initial_batch_size=4)
    assert result == "processed on cpu with batch 4"
    assert meta["device_used"] == "cpu"
    assert meta["batch_size_used"] == 4
    assert meta["fallback_to_cpu"] is False
    assert meta["retried_smaller_batch"] is False


def test_execute_with_retry_simulated_oom_ladder():
    """Test OOM retry ladder: GPU OOM -> retry smaller batch on GPU -> if still OOM -> CPU fallback."""
    dm = DeviceManager(force_cpu=False)

    # Mock is_cuda_available to simulate a GPU environment
    with patch.object(dm, "is_cuda_available", return_value=True):
        attempts = []

        def failing_gpu_work(device: str, batch: int):
            attempts.append((device, batch))
            if "cuda" in device:
                raise SimulatedCudaOOMError("Simulated CUDA OOM")
            return f"success on {device} batch {batch}"

        # Starting batch size 4 on simulated GPU
        result, meta = dm.execute_with_retry(failing_gpu_work, initial_batch_size=4)

        # Sequence of attempts:
        # 1. cuda:0 with batch 4 (OOM)
        # 2. cuda:0 with batch 2 (OOM)
        # 3. cpu with batch 2 (Success)
        assert attempts == [
            ("cuda:0", 4),
            ("cuda:0", 2),
            ("cpu", 2),
        ]
        assert result == "success on cpu batch 2"
        assert meta["fallback_to_cpu"] is True
        assert meta["retried_smaller_batch"] is True
        assert dm.get_device_actually_used() == "cpu"


def test_execute_with_retry_gpu_success_on_halved_batch():
    dm = DeviceManager(force_cpu=False)

    with patch.object(dm, "is_cuda_available", return_value=True):
        attempts = []

        def work_succeeds_on_smaller(device: str, batch: int):
            attempts.append((device, batch))
            if "cuda" in device and batch > 2:
                raise SimulatedCudaOOMError("OOM on large batch")
            return f"success on {device} batch {batch}"

        result, meta = dm.execute_with_retry(
            work_succeeds_on_smaller, initial_batch_size=4
        )

        assert attempts == [
            ("cuda:0", 4),
            ("cuda:0", 2),
        ]
        assert result == "success on cuda:0 batch 2"
        assert meta["fallback_to_cpu"] is False
        assert meta["retried_smaller_batch"] is True
        assert dm.get_device_actually_used() == "cuda:0"


def test_retry_on_oom_context_manager():
    dm = DeviceManager(force_cpu=True)
    fallback_called = []

    def on_fallback(dev):
        fallback_called.append(dev)

    with (
        pytest.raises(SimulatedCudaOOMError),
        dm.retry_on_oom(on_fallback=on_fallback) as ctx,
    ):
        assert ctx["device"] == "cpu"
        raise SimulatedCudaOOMError("CUDA out of memory")

    assert fallback_called == ["cpu"]
    assert dm.get_device_actually_used() == "cpu"


def test_health_endpoint_reports_device_actually_used():
    dm = get_device_manager()
    dm.reset_device_tracking()

    client = TestClient(create_app())

    # Case 1: Default state (no verified GPU tensor) -> reports 'cpu'
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["device"] == "cpu"

    # Case 2: After verified GPU computation
    dm.record_device_use("cuda:0")
    resp2 = client.get("/api/v1/health")
    assert resp2.status_code == 200
    assert resp2.json()["device"] == "cuda:0"

    # Reset
    dm.reset_device_tracking()
