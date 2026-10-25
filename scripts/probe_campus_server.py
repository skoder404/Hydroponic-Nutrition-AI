#!/usr/bin/env python3
"""Probe the campus cloud server and capture redacted samples.

Usage (from campus Wi-Fi or when testing):
    python scripts/probe_campus_server.py

Reads configuration from .env or environment variables:
  - CAMPUS_SERVER_URL
  - CAMPUS_SERVER_USER / CAMPUS_SERVER_PASSWORD
  - CAMPUS_SECONDARY_URL
  - CAMPUS_CAMERA_URL
  - CAMPUS_REQUEST_TIMEOUT_S

Never prints or logs real IPs or credentials. Redacts all sensitive fields
before writing sample payloads to docs/decisions/0008-campus-server/.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

# Ensure stdout handles UTF-8 on Windows consoles without charmap crash
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001, S110
        pass

# ---------------------------------------------------------------------------
# Load .env (manual fallback if python-dotenv is not installed)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k not in os.environ:
                os.environ[k] = v

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "docs" / "decisions" / "0008-campus-server"

# Keys to redact in sample payloads
_REDACT_KEYS = {"device_id", "customer_id", "ip", "mac", "serial", "ssid", "password", "user", "auth"}
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _redact(obj: object) -> object:
    """Recursively redact sensitive keys and IP addresses."""
    if isinstance(obj, dict):
        out: dict[str, object] = {}
        for k, v in obj.items():
            if k.lower() in _REDACT_KEYS:
                out[k] = "<REDACTED>"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(item) for item in obj]
    if isinstance(obj, str):
        return _IP_RE.sub("<REDACTED_IP>", obj)
    return obj


def _save_json(data: object, filename: str) -> Path:
    """Save redacted JSON to the samples directory."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLES_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def _make_request(url: str, timeout: float, user: str = "", password: str = "") -> tuple[int | None, bytes | None, dict]:
    """Execute a GET request with optional Basic Auth. Returns (status, body, headers)."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Hydroponic-Nutrition-AI/Probe")

    if user and password:
        creds = f"{user}:{password}".encode()
        b64_creds = base64.b64encode(creds).decode("ascii")
        req.add_header("Authorization", f"Basic {b64_creds}")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            body = resp.read()
            headers = dict(resp.headers)
            return status, body, headers
    except urllib.error.HTTPError as exc:
        body = exc.read() if exc.fp else b""
        return exc.code, body, dict(exc.headers or {})
    except urllib.error.URLError as exc:
        return None, None, {"error": str(exc.reason)}
    except Exception as exc:  # noqa: BLE001
        return None, None, {"error": str(exc)}


def _probe_status(base_url: str, timeout: float, user: str = "", password: str = "") -> dict | None:
    """GET /api/status or / on the given base URL."""
    status_url = f"{base_url.rstrip('/')}/api/status"
    print(f"  -> Probing {urlparse(status_url).path} ...")

    code, body, headers = _make_request(status_url, timeout, user, password)
    if code == 200 and body:
        try:
            data = json.loads(body.decode("utf-8"))
            print(f"  [OK] Status 200, parsed {len(body)} bytes JSON")
            return data
        except Exception:  # noqa: BLE001
            print(f"  [WARN] Status 200 but not JSON: {body[:100]!r}")
            return None
    elif code is not None:
        print(f"  [WARN] HTTP {code}: {headers.get('error', '')}")
        # Also try root /
        root_url = base_url.rstrip("/") + "/"
        c2, b2, _h2 = _make_request(root_url, timeout, user, password)
        if c2 == 200 and b2:
            print(f"  [OK] Root endpoint '/' returned HTTP 200 ({len(b2)} bytes)")
        return None
    else:
        print(f"  [FAIL] Unreachable: {headers.get('error', 'connection failed')}")
        return None


def _probe_camera(camera_url: str, timeout: float, user: str = "", password: str = "") -> bool:
    """Probe camera URL for image or stream frame."""
    if not camera_url:
        print("  - Camera URL not configured, skipping")
        return False

    print("  -> Probing camera URL ...")
    code, body, headers = _make_request(camera_url, timeout, user, password)
    if code == 200 and body:
        ctype = headers.get("Content-Type", "")
        print(f"  [OK] Camera responded: Content-Type={ctype}, {len(body)} bytes")
        if "image" in ctype or body[:2] == b"\xff\xd8":  # JPEG magic bytes
            frame_path = SAMPLES_DIR / "sample_frame.jpg"
            frame_path.write_bytes(body)
            print(f"  Saved sample frame -> {frame_path.relative_to(Path.cwd())}")
            return True
        return True
    elif code is not None:
        print(f"  [WARN] Camera returned HTTP {code}")
        return False
    else:
        print(f"  [FAIL] Camera unreachable: {headers.get('error', '')}")
        return False


def _analyze_payload(payload: dict) -> None:
    """Print an analysis of sensor fields and specifically N/P/K."""
    print("\n-- Payload Analysis --")
    skip_keys = {"device_id", "customer_id", "timestamp", "relay_status"}

    numeric_fields = []
    for k, v in payload.items():
        if k in skip_keys:
            continue
        if isinstance(v, (int, float)):
            numeric_fields.append((k, v, type(v).__name__))
        elif isinstance(v, str):
            try:
                float(v)
                numeric_fields.append((k, v, "numeric_string"))
            except ValueError:
                pass

    if numeric_fields:
        print(f"  Found {len(numeric_fields)} potential sensor field(s):")
        for name, val, dtype in numeric_fields:
            print(f"    - {name}: {val} ({dtype})")
    else:
        print("  Notice: No top-level numeric sensor fields detected.")
        print(f"  Available payload keys: {list(payload.keys())}")

    # Check for N/P/K
    npk_keys = {"nitrogen", "phosphorus", "potassium", "n", "p", "k", "n_ppm", "p_ppm", "k_ppm"}
    found_npk = [k for k in payload if k.lower() in npk_keys]
    if found_npk:
        print(f"  [OK] N/P/K fields detected: {found_npk}")
    else:
        print("  Notice: No N/P/K fields found (tomato model may need live-compatible variant).")


def main() -> int:
    campus_url = os.environ.get("CAMPUS_SERVER_URL", "").strip()
    user = os.environ.get("CAMPUS_SERVER_USER", "").strip()
    password = os.environ.get("CAMPUS_SERVER_PASSWORD", "").strip()

    sec_url = os.environ.get("CAMPUS_SECONDARY_URL", "").strip()
    sec_user = os.environ.get("CAMPUS_SECONDARY_USER", "").strip()
    sec_password = os.environ.get("CAMPUS_SECONDARY_PASSWORD", "").strip()

    camera_url = os.environ.get("CAMPUS_CAMERA_URL", "").strip()
    timeout = float(os.environ.get("CAMPUS_REQUEST_TIMEOUT_S", "5"))

    if not campus_url:
        print("------------------------------------------------------------")
        print("CAMPUS_SERVER_URL is not set.")
        print("Set it in .env (e.g. CAMPUS_SERVER_URL=http://<host>:<port>)")
        print("This script is meant to run when connected to campus Wi-Fi.")
        print("------------------------------------------------------------")
        return 0

    parsed = urlparse(campus_url)
    print("============================================================")
    print("  Hydroponic Nutrition AI - Campus Server Probe")
    print(f"  Target: {parsed.scheme}://<redacted_host>:{parsed.port or 'default'}")
    print(f"  Auth: {'configured' if user else 'none'}")
    print(f"  Timeout: {timeout}s")
    print("============================================================")

    # Probe 1: Primary server /api/status
    print("\n[Probe 1] Testing primary server...")
    status_data = _probe_status(campus_url, timeout, user, password)

    # Also test default cloud_server port (50503) if different from configured URL
    port = parsed.port
    if port != 50503 and parsed.hostname:
        alt_url = f"{parsed.scheme}://{parsed.hostname}:50503"
        print("\n[Probe 1b] Testing cloud_server port 50503...")
        alt_status = _probe_status(alt_url, timeout, user, password)
        if alt_status and not status_data:
            status_data = alt_status

    if status_data is not None:
        redacted = _redact(status_data)
        out_path = _save_json(redacted, "live_status_response.json")
        print(f"  Saved redacted status -> {out_path.name}")

        latest = status_data.get("latest")
        if isinstance(latest, dict):
            _analyze_payload(latest)
            out_latest = _save_json(_redact(latest), "live_device_payload.json")
            print(f"  Saved redacted payload -> {out_latest.name}")

        # Check update frequency with second read after short delay
        print("\n[Probe 2] Checking update frequency (waiting 10s)...")
        time.sleep(10)
        status_2 = _probe_status(campus_url, timeout, user, password)
        if status_2:
            _save_json(_redact(status_2), "live_status_response_2.json")
            ts1 = (status_data.get("latest") or {}).get("timestamp")
            ts2 = (status_2.get("latest") or {}).get("timestamp")
            if ts1 and ts2 and ts1 != ts2:
                print(f"  Timestamp updated within 10s: {ts1} -> {ts2}")
            else:
                print("  Timestamp did not change within 10s.")
    else:
        print("\nPrimary campus server is currently unreachable.")
        print("Note: When connected to the campus Wi-Fi network, re-run this script.")

    # Probe 2: Secondary endpoint (if set)
    if sec_url:
        print("\n[Probe 3] Testing secondary endpoint...")
        sec_code, sec_body, sec_h = _make_request(sec_url, timeout, sec_user, sec_password)
        if sec_code == 200:
            print(f"  [OK] Secondary endpoint responded HTTP 200 ({len(sec_body or b'')} bytes)")
        elif sec_code is not None:
            print(f"  [WARN] Secondary endpoint HTTP {sec_code}")
        else:
            print(f"  [FAIL] Secondary endpoint unreachable: {sec_h.get('error', '')}")

    # Probe 3: Camera
    print("\n[Probe 4] Testing camera feed...")
    _probe_camera(camera_url, timeout, sec_user, sec_password)

    print("\nProbe execution finished cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
