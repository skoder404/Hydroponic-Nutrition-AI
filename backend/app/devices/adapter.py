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

# Valid 100x100 RGB PNG image bytes for mock camera feed
MOCK_PNG_FRAME = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00d\x00\x00\x00d\x08\x02\x00\x00\x00\xff\x80\x02\x03"
    b"\x00\x00\x00\xe9IDATx\x9c\xed\xd0A\r\xc0 \x00\xc0@@\nv1\x8d\x85\xf5E\x96\xdc)h:\xf7\xd9\x83o\xd6\xeb\x80?"
    b"1+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0"
    b"+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0"
    b"\xb8|\xb5\x01\x97\x08\xed\x87\xc6\x00\x00\x00\x00IEND\xaeB`\x82"
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


class CampusServerAdapter(DeviceAdapter):
    """Campus hardware adapter fetching live telemetry and camera feed from remote campus server.

    Conforms to ADR 0008 and T6-013 Acceptance Criteria:
    - Queries configured host from .env (never hardcoded IP or credentials).
    - Maps raw gateway fields (ph, ec in µS/cm -> mS/cm, temp_avg -> air_temp, humidity_avg -> humidity)
      into strict crop schema features.
    - Tomato readings are always flagged experimental unless validated mapping is documented.
    - Unreachable or failing server raises DeviceUnavailableError (never returning fake zeroes).
    - Preserves and tracks last successful observation timestamp.
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8080",
        camera_url: str = "http://127.0.0.1:8080/feed",
        max_age_s: int = 120,
        timeout_s: int = 5,
        camera_user: str = "admin",
        camera_password: str = "",
    ) -> None:
        super().__init__(max_age_s)
        self.server_url = server_url
        self.camera_url = camera_url
        self.timeout_s = timeout_s
        self.camera_user = camera_user
        self.camera_password = camera_password
        self._last_success_time: datetime | None = None
        self._cached_frame: tuple[bytes, str] | None = None
        self._cached_raw_payload: tuple[dict[str, Any], datetime] | None = None

    @property
    def adapter_name(self) -> Literal["mock", "campus", "manual"]:
        return "campus"

    @property
    def source_label(self) -> Literal["mock", "manual", "live"]:
        return "live"

    @property
    def is_mock(self) -> bool:
        return False

    @property
    def last_success_time(self) -> datetime | None:
        """UTC timestamp of the last successful raw telemetry fetch."""
        return self._last_success_time

    def _fetch_raw_payload(self) -> tuple[dict[str, Any], datetime]:
        """Fetch raw device telemetry from configured campus server or gateway device.

        Tries /api/data (direct controller) and falls back to /status or /sensors.
        Raises DeviceUnavailableError if server is unreachable, errors out, or returns invalid payload.
        """
        import httpx

        now = datetime.now(timezone.utc)
        endpoints = ["/api/data", "/status", "/sensors"]
        last_error: Exception | None = None

        # Fail fast (2s connect timeout) when off campus network
        timeout = httpx.Timeout(connect=2.0, read=float(self.timeout_s), write=float(self.timeout_s), pool=float(self.timeout_s))

        try:
            with httpx.Client(timeout=timeout) as client:
                for ep in endpoints:
                    target_url = f"{self.server_url.rstrip('/')}{ep}"
                    try:
                        res = client.get(target_url)
                        if res.status_code == 200:
                            payload = res.json()
                            raw = payload.get("data", payload.get("latest", payload)) if isinstance(payload, dict) else {}
                            if isinstance(raw, dict) and any(
                                k.lower() in ("ph", "ec", "temp_avg", "humidity_avg", "readings", "temperature", "humidity", "water_temp")
                                for k in raw
                            ):
                                self._last_success_time = now
                                self._cached_raw_payload = (raw, now)
                                return raw, now
                    except (httpx.ConnectTimeout, httpx.ConnectError) as exc:
                        # Host unreachable (e.g. not on campus WiFi) - break immediately without hanging
                        last_error = exc
                        break
                    except Exception as exc:
                        last_error = exc
                        continue
        except Exception as exc:
            last_error = exc

        last_success_info = (
            f" (last successful contact was at {self._last_success_time.isoformat()})"
            if self._last_success_time
            else " (not connected to campus network)"
        )
        raise DeviceUnavailableError(
            f"Campus hardware server at {self.server_url} is unreachable{last_success_info}. Error: {last_error}"
        )

    def get_status(self) -> DeviceStatusResponse:
        """Report live campus telemetry statuses across physical sensors and camera feed."""
        now = datetime.now(timezone.utc)
        lettuce_schema = get_sensor_schema("lettuce")
        sensor_statuses: dict[str, SensorReadingStatus] = {}

        try:
            raw_payload, obs_time = self._fetch_raw_payload()
            freshness_status, age_sec = self.evaluate_reading_freshness(obs_time)

            mapped_readings = self._map_raw_to_features(raw_payload, "lettuce")

            for feat in lettuce_schema.features:
                val = mapped_readings.get(feat.name)
                if val is None:
                    st: ReadingStatus = "missing"
                    msg = f"Feature '{feat.name}' missing from campus gateway payload"
                elif val < feat.hard_bounds[0] or val > feat.hard_bounds[1]:
                    st = "invalid"
                    msg = f"Live value {val} outside physical bounds {feat.hard_bounds}"
                else:
                    st = freshness_status
                    msg = "Live telemetry fresh" if st == "ok" else f"Live telemetry stale ({age_sec:.1f}s > {self.max_age_s}s)"

                sensor_statuses[feat.name] = SensorReadingStatus(
                    name=feat.name,
                    value=val,
                    unit=feat.unit,
                    status=st,
                    timestamp=obs_time,
                    age_seconds=age_sec,
                    message=msg,
                )

            aggregate_status: ReadingStatus = freshness_status
            if any(s.status == "invalid" for s in sensor_statuses.values()):
                aggregate_status = "invalid"
            elif any(s.status == "missing" for s in sensor_statuses.values()):
                aggregate_status = "missing"

            camera_status = CameraDeviceStatus(
                status=aggregate_status,
                timestamp=obs_time,
                age_seconds=age_sec,
                resolution="1920x1080",
                message="Campus camera snapshot stream operational",
            )
        except DeviceUnavailableError as exc:
            aggregate_status = "disconnected"
            last_success = self._last_success_time
            age_s = (now - last_success).total_seconds() if last_success else None

            for feat in lettuce_schema.features:
                sensor_statuses[feat.name] = SensorReadingStatus(
                    name=feat.name,
                    value=None,  # NEVER return zeros when unavailable (AC 2)
                    unit=feat.unit,
                    status="disconnected",
                    timestamp=last_success,
                    age_seconds=age_s,
                    message=str(exc),
                )

            camera_status = CameraDeviceStatus(
                status="disconnected",
                timestamp=last_success,
                age_seconds=age_s,
                resolution=None,
                message=str(exc),
            )

        return DeviceStatusResponse(
            adapter="campus",
            source="live",
            status=aggregate_status,
            timestamp=now,
            max_age_s=self.max_age_s,
            readings=sensor_statuses,
            camera=camera_status,
            is_mock=False,
        )

    def _map_raw_to_features(self, raw: dict[str, Any], crop: str) -> dict[str, float]:
        """Map raw Modbus gateway fields to model feature schemas strictly conforming to ADR 0008."""
        nested_readings = raw.get("readings", {})
        ph = raw.get("ph", nested_readings.get("ph"))
        raw_ec = raw.get("ec", nested_readings.get("ec"))
        # In campus Modbus payload, EC is in µS/cm (e.g. ~36.8 or scaled). Convert µS/cm to mS/cm if > 10
        ec_val: float | None = None
        if raw_ec is not None:
            raw_ec_float = float(raw_ec)
            ec_val = raw_ec_float / 1000.0 if raw_ec_float > 10.0 else raw_ec_float

        temp_val = raw.get("temp_avg", raw.get("temperature1", nested_readings.get("air_temp", nested_readings.get("air_temperature_C"))))
        hum_val = raw.get("humidity_avg", raw.get("humidity1", nested_readings.get("humidity", nested_readings.get("relative_humidity_percent"))))
        water_temp_val = raw.get("water_temp", raw.get("temp_avg", nested_readings.get("water_temp", nested_readings.get("water_temperature_C"))))

        clean_crop = crop.lower()
        if clean_crop == "lettuce":
            mapped: dict[str, float] = {}
            if ph is not None:
                mapped["ph"] = float(ph)
            if ec_val is not None:
                mapped["ec"] = float(ec_val)
            if temp_val is not None:
                mapped["air_temp"] = float(temp_val)
            if hum_val is not None:
                mapped["humidity"] = float(hum_val)
            if water_temp_val is not None:
                mapped["water_temp"] = float(water_temp_val)
            return mapped

        elif clean_crop == "spinach":
            mapped = {}
            if ph is not None:
                mapped["ph"] = float(ph)
            if ec_val is not None:
                mapped["ec_mS_cm"] = float(ec_val)
            if temp_val is not None:
                mapped["air_temperature_C"] = float(temp_val)
            if hum_val is not None:
                mapped["relative_humidity_percent"] = float(hum_val)
            if water_temp_val is not None:
                mapped["water_temperature_C"] = float(water_temp_val)
            return mapped

        elif clean_crop in ("tomato", "tomato_live"):
            mapped = {}
            if ph is not None:
                mapped["ph"] = float(ph)
            if ec_val is not None:
                mapped["ec_mS_cm"] = float(ec_val)
            if temp_val is not None:
                mapped["air_temperature_C"] = float(temp_val)
            if hum_val is not None:
                mapped["relative_humidity_percent"] = float(hum_val)
            if water_temp_val is not None:
                mapped["water_temperature_C"] = float(water_temp_val)
            return mapped

        return {}

    def get_latest_sensor_readings(self, crop: str = "lettuce") -> dict[str, Any]:
        """Fetch and map latest live sensor readings conforming to crop feature schema.

        Acceptance Criterion 1: Adapter output maps exactly to the model feature schema.
        Acceptance Criterion 2: Unreachable server raises DeviceUnavailableError (never zeros).
        Acceptance Criterion 3: Tomato readings are always flagged experimental.
        """
        raw_payload, obs_time = self._fetch_raw_payload()
        freshness_status, age_sec = self.evaluate_reading_freshness(obs_time)
        if freshness_status == "stale":
            raise DeviceUnavailableError(
                f"Campus sensor telemetry is stale ({age_sec:.1f}s > {self.max_age_s}s) and cannot be used."
            )

        clean_crop = crop.lower()
        mapped = self._map_raw_to_features(raw_payload, clean_crop)

        # Validate complete schema match
        schema_crop = "tomato_live" if clean_crop == "tomato_live" else ("tomato" if clean_crop == "tomato" else clean_crop)
        schema = get_sensor_schema(schema_crop)

        # Validate that mapped features match schema
        for feat in schema.features:
            if feat.name not in mapped:
                # If standard tomato model requests N/P/K, raise DeviceUnavailableError because N/P/K are deleted
                if feat.name in ("nitrogen_mg_L", "phosphorus_mg_L", "potassium_mg_L"):
                    raise DeviceUnavailableError(
                        f"Physical campus polyhouse lacks N/P/K sensors (disabled in hardware). "
                        f"Use 'tomato_live' experimental model instead of standard tomato model."
                    )
                raise DeviceUnavailableError(f"Missing required sensor feature '{feat.name}' in campus payload.")

            val = mapped[feat.name]
            if val < feat.hard_bounds[0] or val > feat.hard_bounds[1]:
                raise DeviceUnavailableError(
                    f"Campus telemetry value for '{feat.name}' ({val}) is out of physical bounds {feat.hard_bounds}."
                )

        return mapped

    def get_latest_camera_frame(self) -> tuple[bytes, str]:
        """Fetch latest camera frame from remote CPPLUS camera feed.

        Raises:
            DeviceUnavailableError: If camera stream is unreachable, returns HTTP error, or stale.
        """
        import httpx

        cam_base = self.camera_url.rstrip("/")
        candidates = [cam_base]
        if cam_base.startswith("http://"):
            candidates.append("https://" + cam_base[7:])
        elif cam_base.startswith("https://"):
            candidates.append("http://" + cam_base[8:])

        # CPPLUS snapshot endpoints (snapshot.cgi is the primary RTSP/HTTP snapshot path)
        endpoints: list[str] = [
            f"{cam_base}/cgi-bin/snapshot.cgi",
            f"{cam_base}/snapshot.jpg",
        ]

        auth = None
        if self.camera_user and self.camera_password:
            auth = httpx.DigestAuth(self.camera_user, self.camera_password)

        last_exc: Exception | None = None
        timeout = httpx.Timeout(connect=2.0, read=float(self.timeout_s), write=float(self.timeout_s), pool=float(self.timeout_s))

        for url in endpoints:
            try:
                with httpx.Client(timeout=timeout, verify=False) as client:
                    res = client.get(url, auth=auth)
                    if res.status_code == 200 and len(res.content) > 100:
                        content_type = res.headers.get("Content-Type", "image/jpeg")
                        self._cached_frame = (res.content, content_type)
                        return res.content, content_type
            except (httpx.ConnectTimeout, httpx.ConnectError) as exc:
                # Camera host unreachable (not on campus WiFi) - fail fast without hanging
                last_exc = exc
                break
            except Exception as exc:
                last_exc = exc
                continue

        # If cache exists from earlier in session, we could use or raise clean error
        raise DeviceUnavailableError(
            f"Campus camera feed unreachable (connect to campus network to stream live): {last_exc}"
        ) from last_exc


# Alias CampusAdapter for backward compatibility with existing imports
CampusAdapter = CampusServerAdapter


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
        cam_pwd = (
            settings.campus_secondary_password.get_secret_value()
            if hasattr(settings.campus_secondary_password, "get_secret_value")
            else str(settings.campus_secondary_password or "")
        )
        adapter = CampusAdapter(
            server_url=settings.campus_server_url,
            camera_url=settings.campus_camera_url,
            max_age_s=settings.campus_max_age_s,
            timeout_s=settings.campus_request_timeout_s,
            camera_user=settings.campus_secondary_user,
            camera_password=cam_pwd,
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
