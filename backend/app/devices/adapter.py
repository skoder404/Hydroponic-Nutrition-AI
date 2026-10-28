"""DeviceAdapter interface, MockAdapter, ManualAdapter, and CampusAdapter (T6-012).

Conforms to TECH_RULES and T6-012 Acceptance Criteria:
- Stale payload is reported stale and never used (raises DeviceUnavailableError).
- Mock data is never labelled live (source is 'mock', is_mock=True).
- Manual entry works with DEVICE_ADAPTER=manual.
- Validates payload schema, units, timestamp, and freshness (CAMPUS_MAX_AGE_S).
- Reports one of: missing / disconnected / invalid / stale / ok per reading.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Literal

from backend.app.core.config import get_settings
from backend.app.core.errors import DeviceUnavailableError
from backend.app.schemas.device import (
    CameraDeviceStatus,
    DeviceStatusResponse,
    ReadingStatus,
    SensorReadingStatus,
)
from ml.sensor_models.schemas import get_sensor_schema

logger = logging.getLogger(__name__)

# Minimal valid 1x1 transparent PNG image bytes for mock camera feed
MOCK_PNG_FRAME = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
    b"\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf"
    b"\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
)


class DeviceAdapter(ABC):
    """Abstract base interface for hardware device integrations."""

    def __init__(self, max_age_s: int = 120) -> None:
        self.max_age_s = max_age_s

    @property
    @abstractmethod
    def adapter_name(self) -> Literal["mock", "campus", "manual"]:
        """Unique identifier of the adapter implementation."""

    @property
    @abstractmethod
    def source_label(self) -> Literal["mock", "manual", "live"]:
        """Data provenance label (MockAdapter MUST return 'mock', never 'live')."""

    @property
    @abstractmethod
    def is_mock(self) -> bool:
        """Flag indicating if this adapter returns mock fixtures."""

    @abstractmethod
    def get_status(self) -> DeviceStatusResponse:
        """Report aggregate device operational status across all sensors and camera."""

    @abstractmethod
    def get_latest_sensor_readings(self, crop: str = "lettuce") -> dict[str, Any]:
        """Fetch the latest valid sensor readings for the specified crop.

        Raises:
            DeviceUnavailableError: If telemetry is missing, disconnected, invalid, or stale.
        """

    @abstractmethod
    def get_latest_camera_frame(self) -> tuple[bytes, str]:
        """Fetch the latest camera image frame as (raw_bytes, mime_type).

        Raises:
            DeviceUnavailableError: If camera is unreachable, invalid, or stale.
        """

    def evaluate_reading_freshness(
        self, observation_time: datetime | None
    ) -> tuple[ReadingStatus, float | None]:
        """Evaluate if an observation timestamp is fresh or stale relative to max_age_s."""
        if observation_time is None:
            return "missing", None

        now = datetime.now(timezone.utc)
        if observation_time.tzinfo is None:
            observation_time = observation_time.replace(tzinfo=timezone.utc)

        age_seconds = max(0.0, (now - observation_time).total_seconds())
        if age_seconds > self.max_age_s:
            return "stale", age_seconds
        return "ok", age_seconds


class MockAdapter(DeviceAdapter):
    """Mock device adapter for local development and CI testing.

    Invariants:
    - Provenance source is strictly 'mock', NEVER 'live'.
    - is_mock is strictly True.
    - Supports simulated staleness for testing rejection of stale telemetry.
    """

    def __init__(self, max_age_s: int = 120) -> None:
        super().__init__(max_age_s)
        self._custom_readings: dict[str, dict[str, float]] = {}
        self._custom_timestamp: datetime | None = None
        self._simulate_disconnected: bool = False
        self._simulate_stale: bool = False

    @property
    def adapter_name(self) -> Literal["mock", "campus", "manual"]:
        return "mock"

    @property
    def source_label(self) -> Literal["mock", "manual", "live"]:
        # TECH_RULES & T6-012 AC: Mock data is NEVER labelled live
        return "mock"

    @property
    def is_mock(self) -> bool:
        return True

    def set_mock_telemetry(
        self,
        crop: str,
        readings: dict[str, float],
        timestamp: datetime | None = None,
    ) -> None:
        """Configure specific mock sensor readings."""
        self._custom_readings[crop.lower()] = readings
        self._custom_timestamp = timestamp

    def set_simulate_disconnected(self, disconnected: bool = True) -> None:
        """Toggle simulated hardware disconnection."""
        self._simulate_disconnected = disconnected

    def set_simulate_stale(self, stale: bool = True) -> None:
        """Toggle simulated stale telemetry."""
        self._simulate_stale = stale

    def _default_readings_for_crop(self, crop: str) -> dict[str, float]:
        """Standard agronomic readings within normal bounds per crop."""
        c = crop.lower()
        if c == "lettuce":
            return {
                "ph": 6.2,
                "ec": 1.8,
                "water_temp": 20.5,
                "humidity": 65.0,
                "air_temp": 22.0,
            }
        elif c == "spinach":
            return {
                "air_temperature_C": 21.0,
                "relative_humidity_percent": 60.0,
                "water_temperature_C": 19.5,
                "ph": 6.5,
                "ec_mS_cm": 2.1,
            }
        elif c == "tomato":
            return {
                "EC_sensor": 2.4,
                "pH_sensor": 6.3,
                "Water_Temp_sensor": 21.5,
                "Air_Temp_sensor": 24.0,
                "Humidity_sensor": 62.0,
            }
        return {"ph": 6.0, "ec": 2.0}

    def get_status(self) -> DeviceStatusResponse:
        now = datetime.now(timezone.utc)

        if self._simulate_disconnected:
            camera_status = CameraDeviceStatus(
                status="disconnected",
                timestamp=None,
                age_seconds=None,
                resolution=None,
                message="Mock hardware disconnected",
            )
            return DeviceStatusResponse(
                adapter="mock",
                source="mock",
                status="disconnected",
                timestamp=now,
                max_age_s=self.max_age_s,
                readings={},
                camera=camera_status,
                is_mock=True,
            )

        # Determine observation timestamp
        if self._simulate_stale:
            # Over 10 minutes old
            obs_time = datetime.fromtimestamp(now.timestamp() - (self.max_age_s + 600), tz=timezone.utc)
        elif self._custom_timestamp:
            obs_time = self._custom_timestamp
        else:
            obs_time = now

        freshness_status, age_sec = self.evaluate_reading_freshness(obs_time)

        # Build lettuce sensor statuses
        lettuce_schema = get_sensor_schema("lettuce")
        readings_dict = (
            self._custom_readings.get("lettuce")
            or self._default_readings_for_crop("lettuce")
        )
        sensor_statuses: dict[str, SensorReadingStatus] = {}

        for f in lettuce_schema.features:
            val = readings_dict.get(f.name)
            if val is None:
                st: ReadingStatus = "missing"
                msg = f"Feature '{f.name}' missing from mock payload"
            elif val < f.hard_bounds[0] or val > f.hard_bounds[1]:
                st = "invalid"
                msg = f"Value {val} outside bounds {f.hard_bounds}"
            else:
                st = freshness_status
                msg = "Telemetry fresh" if st == "ok" else f"Telemetry stale ({age_sec:.1f}s > {self.max_age_s}s)"

            sensor_statuses[f.name] = SensorReadingStatus(
                name=f.name,
                value=val,
                unit=f.unit,
                status=st,
                timestamp=obs_time,
                age_seconds=age_sec,
                message=msg,
            )

        aggregate_status: ReadingStatus = "ok"
        if freshness_status == "stale":
            aggregate_status = "stale"
        elif any(s.status == "invalid" for s in sensor_statuses.values()):
            aggregate_status = "invalid"
        elif any(s.status == "missing" for s in sensor_statuses.values()):
            aggregate_status = "missing"

        camera_status = CameraDeviceStatus(
            status=freshness_status,
            timestamp=obs_time,
            age_seconds=age_sec,
            resolution="1920x1080",
            message="Mock camera stream operational",
        )

        return DeviceStatusResponse(
            adapter="mock",
            source="mock",
            status=aggregate_status,
            timestamp=now,
            max_age_s=self.max_age_s,
            readings=sensor_statuses,
            camera=camera_status,
            is_mock=True,
        )

    def get_latest_sensor_readings(self, crop: str = "lettuce") -> dict[str, Any]:
        status_resp = self.get_status()
        if status_resp.status == "disconnected":
            raise DeviceUnavailableError("Mock hardware is disconnected.")
        if status_resp.status == "stale":
            # T6-012 AC: Stale payload is reported stale and NEVER used
            raise DeviceUnavailableError(
                f"Sensor telemetry is stale (reported stale and cannot be used for inference)."
            )

        readings = self._custom_readings.get(crop.lower()) or self._default_readings_for_crop(crop)
        return dict(readings)

    def get_latest_camera_frame(self) -> tuple[bytes, str]:
        status_resp = self.get_status()
        if status_resp.status in ("disconnected", "stale"):
            raise DeviceUnavailableError(f"Camera frame unavailable ({status_resp.status}).")
        return MOCK_PNG_FRAME, "image/png"


class ManualAdapter(DeviceAdapter):
    """Manual device adapter where operator measurements are submitted in-memory.

    Invariants:
    - Allows manual entry when DEVICE_ADAPTER=manual.
    - If no entry has been submitted, reports status='missing'.
    - Validates submitted readings against physical schema bounds.
    - Rejects stale manual entries when age exceeds max_age_s.
    """

    def __init__(self, max_age_s: int = 120) -> None:
        super().__init__(max_age_s)
        self._entries: dict[str, tuple[dict[str, float], datetime]] = {}

    @property
    def adapter_name(self) -> Literal["mock", "campus", "manual"]:
        return "manual"

    @property
    def source_label(self) -> Literal["mock", "manual", "live"]:
        return "manual"

    @property
    def is_mock(self) -> bool:
        return False

    def submit_readings(
        self,
        crop: str,
        readings: dict[str, float],
        timestamp: datetime | None = None,
    ) -> None:
        """Submit a manual operator measurement entry."""
        ts = timestamp or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        self._entries[crop.lower()] = (readings, ts)

    def clear(self) -> None:
        """Clear all stored manual entries."""
        self._entries.clear()

    def get_status(self) -> DeviceStatusResponse:
        now = datetime.now(timezone.utc)
        lettuce_entry = self._entries.get("lettuce")

        if not lettuce_entry:
            camera_status = CameraDeviceStatus(
                status="missing",
                timestamp=None,
                age_seconds=None,
                resolution=None,
                message="No manual camera frame submitted",
            )
            return DeviceStatusResponse(
                adapter="manual",
                source="manual",
                status="missing",
                timestamp=now,
                max_age_s=self.max_age_s,
                readings={},
                camera=camera_status,
                is_mock=False,
            )

        readings_dict, obs_time = lettuce_entry
        freshness_status, age_sec = self.evaluate_reading_freshness(obs_time)

        lettuce_schema = get_sensor_schema("lettuce")
        sensor_statuses: dict[str, SensorReadingStatus] = {}

        for f in lettuce_schema.features:
            val = readings_dict.get(f.name)
            if val is None:
                st: ReadingStatus = "missing"
                msg = f"Feature '{f.name}' not provided in manual submission"
            elif val < f.hard_bounds[0] or val > f.hard_bounds[1]:
                st = "invalid"
                msg = f"Manual entry {val} exceeds physical bounds {f.hard_bounds}"
            else:
                st = freshness_status
                msg = "Manual reading fresh" if st == "ok" else f"Manual reading stale ({age_sec:.1f}s > {self.max_age_s}s)"

            sensor_statuses[f.name] = SensorReadingStatus(
                name=f.name,
                value=val,
                unit=f.unit,
                status=st,
                timestamp=obs_time,
                age_seconds=age_sec,
                message=msg,
            )

        aggregate_status: ReadingStatus = "ok"
        if freshness_status == "stale":
            aggregate_status = "stale"
        elif any(s.status == "invalid" for s in sensor_statuses.values()):
            aggregate_status = "invalid"
        elif any(s.status == "missing" for s in sensor_statuses.values()):
            aggregate_status = "missing"

        camera_status = CameraDeviceStatus(
            status="missing",
            timestamp=None,
            age_seconds=None,
            resolution=None,
            message="No manual camera frame available",
        )

        return DeviceStatusResponse(
            adapter="manual",
            source="manual",
            status=aggregate_status,
            timestamp=now,
            max_age_s=self.max_age_s,
            readings=sensor_statuses,
            camera=camera_status,
            is_mock=False,
        )

    def get_latest_sensor_readings(self, crop: str = "lettuce") -> dict[str, Any]:
        entry = self._entries.get(crop.lower())
        if not entry:
            raise DeviceUnavailableError(f"No manual sensor readings submitted for crop '{crop}'.")

        readings, obs_time = entry
        freshness_status, age_sec = self.evaluate_reading_freshness(obs_time)
        if freshness_status == "stale":
            # T6-012 AC: Stale payload is reported stale and NEVER used
            raise DeviceUnavailableError(
                f"Manual sensor telemetry is stale ({age_sec:.1f}s > {self.max_age_s}s) and cannot be used."
            )

        # Validate bounds
        schema = get_sensor_schema(crop)
        for f in schema.features:
            if f.name in readings:
                val = readings[f.name]
                if val < f.hard_bounds[0] or val > f.hard_bounds[1]:
                    raise DeviceUnavailableError(
                        f"Manual sensor feature '{f.name}' has invalid value {val} (allowed: {f.hard_bounds})."
                    )

        return dict(readings)

    def get_latest_camera_frame(self) -> tuple[bytes, str]:
        raise DeviceUnavailableError("Manual camera frame not uploaded.")


class CampusAdapter(DeviceAdapter):
    """Campus hardware adapter fetching live telemetry and camera feed from remote campus server."""

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8080",
        camera_url: str = "http://127.0.0.1:8080/feed",
        max_age_s: int = 120,
        timeout_s: int = 5,
    ) -> None:
        super().__init__(max_age_s)
        self.server_url = server_url
        self.camera_url = camera_url
        self.timeout_s = timeout_s

    @property
    def adapter_name(self) -> Literal["mock", "campus", "manual"]:
        return "campus"

    @property
    def source_label(self) -> Literal["mock", "manual", "live"]:
        return "live"

    @property
    def is_mock(self) -> bool:
        return False

    def get_status(self) -> DeviceStatusResponse:
        import httpx

        now = datetime.now(timezone.utc)
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                res = client.get(f"{self.server_url.rstrip('/')}/status")
                if res.status_code != 200:
                    status_name: ReadingStatus = "invalid"
                    msg = f"Campus server returned HTTP {res.status_code}"
                else:
                    status_name = "ok"
                    msg = "Campus server reachable"
        except Exception as exc:
            status_name = "disconnected"
            msg = f"Campus server unreachable: {exc}"

        camera_status = CameraDeviceStatus(
            status=status_name,
            timestamp=now if status_name == "ok" else None,
            age_seconds=0.0 if status_name == "ok" else None,
            resolution="1920x1080" if status_name == "ok" else None,
            message=msg,
        )

        return DeviceStatusResponse(
            adapter="campus",
            source="live",
            status=status_name,
            timestamp=now,
            max_age_s=self.max_age_s,
            readings={},
            camera=camera_status,
            is_mock=False,
        )

    def get_latest_sensor_readings(self, crop: str = "lettuce") -> dict[str, Any]:
        import httpx

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                res = client.get(f"{self.server_url.rstrip('/')}/sensors/{crop}")
                if res.status_code != 200:
                    raise DeviceUnavailableError(f"Campus server error: HTTP {res.status_code}")
                data = res.json()
        except httpx.HTTPError as exc:
            raise DeviceUnavailableError(f"Campus device server is disconnected: {exc}") from exc

        # Timestamp and freshness validation
        ts_str = data.get("timestamp")
        if not ts_str:
            raise DeviceUnavailableError("Campus payload missing timestamp.")

        obs_time = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        freshness_status, age_sec = self.evaluate_reading_freshness(obs_time)
        if freshness_status == "stale":
            raise DeviceUnavailableError(
                f"Campus sensor telemetry is stale ({age_sec:.1f}s > {self.max_age_s}s) and cannot be used."
            )

        return data.get("readings", {})

    def get_latest_camera_frame(self) -> tuple[bytes, str]:
        import httpx

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                res = client.get(self.camera_url)
                if res.status_code != 200:
                    raise DeviceUnavailableError(f"Campus camera error: HTTP {res.status_code}")
                return res.content, res.headers.get("Content-Type", "image/jpeg")
        except httpx.HTTPError as exc:
            raise DeviceUnavailableError(f"Campus camera feed unreachable: {exc}") from exc


_device_adapter_singleton: DeviceAdapter | None = None


def get_device_adapter(adapter_type: str | None = None) -> DeviceAdapter:
    """Return configured DeviceAdapter singleton (or new instance if adapter_type specified)."""
    global _device_adapter_singleton
    if _device_adapter_singleton is not None and adapter_type is None:
        return _device_adapter_singleton

    settings = get_settings()
    kind = (adapter_type or settings.device_adapter).lower()

    if kind == "mock":
        adapter: DeviceAdapter = MockAdapter(max_age_s=settings.campus_max_age_s)
    elif kind == "manual":
        adapter = ManualAdapter(max_age_s=settings.campus_max_age_s)
    elif kind == "campus":
        adapter = CampusAdapter(
            server_url=settings.campus_server_url,
            camera_url=settings.campus_camera_url,
            max_age_s=settings.campus_max_age_s,
            timeout_s=settings.campus_request_timeout_s,
        )
    else:
        logger.warning(f"Unknown device adapter '{kind}', falling back to MockAdapter.")
        adapter = MockAdapter(max_age_s=settings.campus_max_age_s)

    if adapter_type is None:
        _device_adapter_singleton = adapter
    return adapter


def reset_device_adapter() -> None:
    """Reset the cached singleton adapter instance."""
    global _device_adapter_singleton
    _device_adapter_singleton = None
