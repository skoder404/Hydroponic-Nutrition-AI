# ADR 0008 — Campus Server Interface

| Field | Value |
|---|---|
| Field | Value |
|---|---|
| **Status** | Live Verified on Campus Network |
| **Date** | 2026-10-27 |
| **Driver** | Yashwant |
| **Navigator** | Sushil |
| **Sources** | `cloud_server/server.py`, `cloud_server/schema.sql`, `cloud_server/read.txt`, `cloud_server/mysql_config.json`, Live Probe via `scripts/probe_campus_server.py` |

## Context

The campus polyhouse system uses a two-tier architecture:

1. **Gateway device** (`:5000`) — a controller running a Flask/HTTP server that directly reads Modbus sensors, manages pulse timers, and controls relays. It exposes `/api/data` with live telemetry.
2. **Cloud server** (`server.py`, `:50503`) — a Python `http.server.ThreadingHTTPServer` that runs on a campus laptop. It receives sensor uploads via HTTP, stores them in MySQL, and serves a relay-control web panel.
3. **Camera NVR** — a CPPLUS NVR on the local network serving camera streams and HTTP Digest snapshot captures.

Our application reads sensor telemetry and camera frames from this infrastructure.

## Decision

### Protocol & Endpoints

- **Gateway Device (Port 5000):**
  - `GET /api/data`: Returns live sensor telemetry and relay statuses without authentication (session login also supported at `POST /api/login`).
- **Cloud Server (Port 50503):**
  - `GET /api/status`: Latest sensor payload stored in MySQL + pending relay commands.
  - `POST /api/data`: Gateway device upload endpoint.
- **Camera (CPPLUS NVR):**
  - **Live Video Streaming (RTSP):**
    - Protocol: RTSP on port `554`.
    - Main Stream (4K 3200x1800, high-res): `rtsp://<user>:<password>@<nvr-ip>:554/cam/realmonitor?channel=1&subtype=0`
    - Sub-Stream (D1 704x576, low-latency live preview): `rtsp://<user>:<password>@<nvr-ip>:554/cam/realmonitor?channel=1&subtype=1`
    - Verified with OpenCV `cv2.VideoCapture` (both streams decode cleanly in RGB).
  - **Snapshot Capture (HTTPS):**
    - Protocol: HTTPS on port `443` with self-signed certificate.
    - Authentication: HTTP Digest Authentication.
    - Endpoint: `GET /cgi-bin/snapshot.cgi?channel=1` (returns on-demand JPEG frame `Content-Type: image/jpeg`, 3200x1800).


### Verified Live Sensor Fields

Captured directly from the live gateway (`live_device_payload.json`):

| Sensor Field | Modbus Name | Unit | Live Value | Model Feature Mapping |
|---|---|---|---|---|
| `ph` | pH | pH | ~7.66 | `ph` |
| `ec` | EC | µS/cm | ~36.8 | `ec` (scaled to mS/cm) |
| `temp_avg` | Average Air Temp | °C | ~41.5 | `air_temp` / `air_temperature_C` |
| `temperature1` | Ambient Temp 1 | °C | ~42.6 | Optional auxiliary |
| `temperature2` | Ambient Temp 2 | °C | ~40.4 | Optional auxiliary |
| `humidity_avg` | Average Humidity | % | ~34.7 | `humidity` / `relative_humidity_percent` |
| `humidity1` | Relative Humidity 1 | % | ~33.2 | Optional auxiliary |
| `humidity2` | Relative Humidity 2 | % | ~36.1 | Optional auxiliary |
| `salinity` | Salinity | PPT | ~0.02 | Auxiliary |
| `tds` | Total Dissolved Solids | ppm | ~18.3 | Excluded (redundant with EC) |
| `ldr1`, `ldr2` | Light Intensity | Lux/raw | ~14800 | Auxiliary |
| `devices` | Relay States (fan1, pump1..4) | bool | active | Relays & actuators |

### Physical Actuator & Relay Mapping (from `cloud_server/server.py`)

The campus polyhouse hardware uses two RS-485 Modbus relay slaves:
- **Slave 1 (Coils 0–7):**
  - `0`: `FAN1`
  - `1`: `FAN2`
  - `2`: `FOGGER`
  - `3`: `HONEYCOMB`
  - `4–7`: `OUT5` – `OUT8`
- **Slave 2 (Coils 0–3):**
  - `0`: `PUMP1`
  - `1`: `PUMP2`
  - `2`: `PUMP3`
  - `3`: `PUMP4`

Relay control commands can be dispatched via `POST /api/relay/control` on the cloud server (`{"slave_id": 2, "address": 0, "state": true}` or `{"name": "PUMP1", "state": true}`) or posted to `/api/commands` for gateway polling.


### Critical Finding: N/P/K Availability

The gateway device configuration explicitly lists in `_deleted_keys`:
```json
"_deleted_keys": ["nitrogen", "phosphorus", "potassium", "relay_2"]
```
**Conclusion:** Nitrogen (N), Phosphorus (P), and Potassium (K) hardware sensors are **disabled/deleted** in the physical campus setup.
**Consequence for Models:** Per PRD D11 and T4-020, the live-compatible tomato sensor model must not rely on N/P/K and must be trained exclusively on the live-available features (`ph`, `ec`, `temp_avg`, `humidity_avg`).

---

## Resolved Unknowns (Live Verified)

| # | Unknown | Live Finding / Resolution | Impact |
|---|---|---|---|
| 1 | Exact sensor fields, names, units | Verified via `/api/data`: `ph`, `ec` (µS/cm), `temp_avg` (°C), `humidity_avg` (%), `salinity` (PPT), `tds` (ppm). | Feature mapping in `CampusServerAdapter` confirmed. |
| 2 | N/P/K availability | **CONFIRMED NOT AVAILABLE**: `_deleted_keys` explicitly disables N/P/K. | Validates T4-020 live-compatible tomato model requirement. |
| 3 | Camera feed location and format | **CONFIRMED**: CPPLUS NVR serves live RTSP streams on port 554 (4K `subtype=0`, D1 `subtype=1`) and on-demand JPEG snapshots at `/cgi-bin/snapshot.cgi?channel=1` via HTTP Digest Auth. | Frame capture & live streaming pipeline confirmed. |
| 4 | Device upload frequency | Sensor readings refresh at ~1-5s intervals on the gateway. | `CAMPUS_POLL_INTERVAL_S=30` is well within sensor update rate. |
| 5 | Dashboard API at `:5000` | Gateway dashboard exposes `/api/data` directly. | Dual source: adapter can read from `:5000/api/data` or `:50503/api/status`. |
| 6 | Off-campus reachability | Verified campus network only (`192.168.100.x` subnet). | Development uses MockAdapter off-campus. |

## Captured Live Samples

Redacted real artifacts captured from the campus network are in [`docs/decisions/0008-campus-server/`](file:///d:/Hydroponic-Nutrition-AI/docs/decisions/0008-campus-server/):
- `live_status_response.json` — Gateway status response
- `live_device_payload.json` — Complete live telemetry payload with Modbus configs
- `sample_frame.jpg` — Real 656 KB JPEG snapshot captured from CPPLUS camera

## Consequences

- **CampusServerAdapter (T6-013):**
  - Will query `GET /api/data` (or `/api/status`), map `ph`, `ec`, `temp_avg`, `humidity_avg`, and map relay statuses.
  - Will use Digest Authentication with unverified SSL context to fetch camera frames from `/cgi-bin/snapshot.cgi?channel=1`.
- **Live Tomato Model (T4-020):**
  - Sits on solid ground: N/P/K exclusion is empirically verified.
