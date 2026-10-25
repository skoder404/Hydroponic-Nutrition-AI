# ADR 0008 — Campus Server Interface

| Field | Value |
|---|---|
| **Status** | Draft — pending live verification on campus network |
| **Date** | 2026-10-25 |
| **Driver** | Yashwant |
| **Sources** | `cloud_server/server.py`, `cloud_server/schema.sql`, `cloud_server/read.txt`, `cloud_server/mysql_config.json`, campus setup documentation |

## Context

The campus polyhouse system uses a two-tier architecture:

1. **Gateway device** — a Modbus-connected controller (possibly Raspberry Pi or similar) that reads sensors and controls relays. It periodically POSTs sensor data to the cloud server and polls for relay commands.
2. **Cloud server** (`server.py`) — a Python `http.server.ThreadingHTTPServer` that runs on a laptop on the campus LAN. It receives sensor uploads via HTTP, stores them in a MySQL database, and serves a relay-control web panel.

Our application needs to read sensor telemetry and (eventually) camera frames from this infrastructure.

## Decision

### Protocol

- **HTTP** (plain, no TLS) — the cloud server uses `http.server.ThreadingHTTPServer`.
- **No WebSocket, MQTT, or RTSP** endpoints are present in the codebase.
- The server binds to `0.0.0.0` on port **50503** by default (configurable via `mysql_config.json` → `http_port`).

### Authentication

- **None.** The cloud server has no authentication, no API keys, no session tokens. All endpoints are open to any client on the network.
- The campus setup documentation mentions a gateway device dashboard at `<campus-server-ip>:5000` with authentication credentials, but this is a **separate device dashboard**, not the cloud server itself.
- ⚠️ **UNKNOWN**: Whether the device dashboard at port 5000 exposes any additional sensor or camera API. Must be probed live.

### Endpoints (from `server.py`)

| Method | Path | Purpose | Request Body | Response |
|---|---|---|---|---|
| `GET` | `/` | Relay control web panel (HTML) | — | HTML page |
| `GET` | `/api/status` | Latest sensor payload + pending commands | — | `{ "status": "ok", "latest": <payload>, "current_commands": [...] }` |
| `GET` | `/api/commands` | Device polls for pending commands | — | `{ "commands": [...] }` |
| `POST` | `/api/data` | Device uploads sensor snapshot | Full JSON payload | `{ "status": "ok", "message": "data stored" }` |
| `POST` | `/api/relay/control` | Queue a relay ON/OFF command | `{ "name": "PUMP1", "state": true }` or `{ "slave_id": 2, "address": 0, "state": true }` | `{ "status": "ok", "message": "..." }` |
| `POST` | `/api/relay/clear` | Clear all pending commands | — | `{ "status": "ok" }` |
| `POST` | `/api/ack` | Device reports command execution results | Ack payload with `request_id` | `{ "status": "ok" }` |

**Our application reads from:** `GET /api/status` — this returns the most recent device payload from MySQL.

### Payload Structure

The device payload is stored as a JSON blob in MySQL (`device_data.payload`). The structure is:

```json
{
  "device_id": "...",
  "customer_id": "...",
  "timestamp": "...",
  "relay_status": [
    { "name": "FAN1", "state": true, "control_mode": "auto" },
    ...
  ]
  // + sensor fields (⚠️ UNKNOWN — see below)
}
```

**Known fields** (from `server.py` code):
- `device_id` (string) — inserted into `device_data.device_id`
- `customer_id` (string) — inserted into `device_data.customer_id`
- `timestamp` (string) — inserted into `device_data.ts`
- `relay_status` (array) — confirmed from the web panel JavaScript: each entry has `name`, `state` (boolean), `control_mode` ("auto" | "manual")

### ⚠️ UNKNOWN — Sensor Fields

The **exact sensor fields, their names, units, and ranges** are **not known from the code alone**. The `_save_data()` method stores the entire JSON blob without parsing individual sensor fields. The sensor fields can only be determined by capturing a live payload.

**Expected fields** (based on polyhouse context — must be verified):
- Temperature (°C)
- Humidity (%)
- pH
- EC / TDS (µS/cm or ppm)
- Possibly: dissolved oxygen, water temperature, light intensity, CO₂

**Critical question for our models:** Does the payload include **nitrogen (N), phosphorus (P), potassium (K)**? If not, the live-compatible tomato model variant must be trained on only the features the campus server provides (per PRD D11).

### Relay Map

From `server.py` RELAYS constant:

| Slave | Address | Name | Purpose (inferred) |
|---|---|---|---|
| 1 | 0 | FAN1 | Ventilation fan |
| 1 | 1 | FAN2 | Ventilation fan |
| 1 | 2 | FOGGER | Humidity control |
| 1 | 3 | HONEYCOMB | Evaporative cooling pad |
| 1 | 4–7 | OUT5–OUT8 | General-purpose outputs |
| 2 | 0–3 | PUMP1–PUMP4 | Nutrient/water pumps |

### Database Schema

MySQL database `polyhouse` with two tables:
- `device_data` — `id`, `device_id`, `customer_id`, `ts`, `payload` (JSON), `created_at`
- `commands_log` — `id`, `device_id`, `ts`, `payload` (JSON), `created_at`

### Update Frequency

- ⚠️ **UNKNOWN** — the gateway's upload interval is configured on the device side (not in the cloud server code). The web panel polls `/api/status` every **5 seconds** for display. The device polls `/api/commands` at a configurable interval (documentation says ~10 seconds).

### Camera

- ⚠️ **UNKNOWN** — no camera endpoint exists in `server.py`. The secondary campus IP (`<campus-secondary-ip>`) may serve camera feeds (RTSP, MJPEG, or HTTP snapshots), but this cannot be determined without a live probe.
- The device dashboard at `<campus-server-ip>:5000` may also embed a camera view.

### Network Reachability

- The cloud server is on the **campus LAN only** (`192.168.x.x` subnet).
- ⚠️ **UNKNOWN**: Whether it is reachable via VPN or only from physically connected campus Wi-Fi.

## Environment Variables

All IPs and credentials stay in `.env` (gitignored). The `.env.example` provides safe placeholder values:

```env
CAMPUS_SERVER_URL=http://127.0.0.1:50503
CAMPUS_CAMERA_URL=http://127.0.0.1:50503/feed
CAMPUS_POLL_INTERVAL_S=30
CAMPUS_MAX_AGE_S=120
CAMPUS_REQUEST_TIMEOUT_S=5
```

When connected to campus Wi-Fi, `.env` should use the real campus IP and port.

## Unknowns to Resolve on Campus Network

| # | Unknown | How to resolve | Impact |
|---|---|---|---|
| 1 | Exact sensor fields, names, units | `GET /api/status` on live server; save redacted payload | Feature mapping for models |
| 2 | N/P/K availability | Same as #1 | Determines tomato model variant |
| 3 | Camera feed location and format | Probe `<campus-secondary-ip>`, probe `<campus-server-ip>:5000` | `CAMPUS_CAMERA_URL` and frame capture |
| 4 | Device upload frequency | Observe two consecutive `GET /api/status` timestamps | `CAMPUS_POLL_INTERVAL_S` default |
| 5 | Dashboard API at `:5000` | Browse `<campus-server-ip>:5000`, inspect endpoints | Additional data sources |
| 6 | Off-campus reachability | Test from non-campus network | Development workflow |

## Sample Payloads

Redacted samples (inferred from code) are in [`docs/decisions/0008-campus-server/`](file:///d:/Hydroponic-Nutrition-AI/docs/decisions/0008-campus-server/):
- `sample_payload_1.json` — device upload structure (relay_status confirmed, sensor fields unknown)
- `sample_api_status_response.json` — GET /api/status response structure
- `sample_relay_control_request.json` — POST /api/relay/control request forms

**These will be replaced with real captured payloads after the live probe.**

## Consequences

- **CampusServerAdapter** will use `httpx` to call `GET {CAMPUS_SERVER_URL}/api/status`, parse `latest` for sensor readings, and map field names to each crop's model schema.
- Until the live probe confirms sensor fields, the adapter will use a best-effort field mapping and raise clear warnings for any missing/unexpected fields.
- The mock adapter and manual entry remain the default; campus mode is opt-in via `DEVICE_ADAPTER=campus`.
- Camera integration is deferred until the camera endpoint is discovered.
