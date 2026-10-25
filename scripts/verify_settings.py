#!/usr/bin/env python3
"""Verification script for T1-004: Settings loader and .env.example validation.

Validates:
1. .env.example contains all required keys and contains placeholders only (no real secrets or IPs).
2. App settings load cleanly with no GEMINI_API_KEY set (optional API key).
3. Printing / repr of the Settings instance never leaks secret values (e.g. GEMINI_API_KEY or DB password).
"""
import os
import re
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_KEYS = [
    "DATABASE_URL",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "MODEL_REGISTRY_PATH",
    "IMAGE_DATA_PATH",
    "SENSOR_DATA_PATH",
    "PROCESSED_DATA_PATH",
    "CAMPUS_SERVER_URL",
    "CAMPUS_CAMERA_URL",
    "CAMPUS_POLL_INTERVAL_S",
    "CAMPUS_MAX_AGE_S",
    "CAMPUS_REQUEST_TIMEOUT_S",
    "ALLOWED_ORIGINS",
    "MAX_UPLOAD_MB",
    "DEVICE_ADAPTER",
    "SAM2_CHECKPOINT",
    "SAM2_CONFIG",
    "FORCE_CPU",
]


def check_env_example():
    path = Path(".env.example")
    if not path.is_file():
        return False, ".env.example does not exist"
    content = path.read_text(encoding="utf-8")

    missing = []
    for key in REQUIRED_KEYS:
        if not re.search(rf"^{key}=", content, re.MULTILINE):
            missing.append(key)

    if missing:
        return False, f".env.example missing required keys: {missing}"

    # Verify no real secrets or live external campus IPs
    suspicious_patterns = [
        r"AIza[0-9A-Za-z-_]{35}",  # Google API key pattern
        r"(?<!127\.0\.0\.)(?<!localhost)(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)\d+\.\d+",
    ]
    for pat in suspicious_patterns:
        if re.search(pat, content):
            return False, f"Potential real secret or non-localhost IP found in .env.example matching {pat}"

    return True, ".env.example contains all required keys with placeholders only."


def check_settings_load_without_api_key():
    from backend.app.core.config import Settings

    # Ensure GEMINI_API_KEY is not in environment
    env_copy = os.environ.copy()
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]

    try:
        settings = Settings(gemini_api_key=None)
        if settings.gemini_api_key is not None:
            return False, "Expected gemini_api_key to be None when omitted"
        return True, "Settings load cleanly with no GEMINI_API_KEY set."
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to load Settings with no GEMINI_API_KEY: {exc}"
    finally:
        os.environ.clear()
        os.environ.update(env_copy)


def check_secret_redaction():
    from backend.app.core.config import Settings

    secret_key = "AIzaSyFakeSecretKeyForTestingRedaction123"
    db_pass = "super_secret_db_password"
    db_url = f"postgresql://admin:{db_pass}@localhost:5432/testdb"

    settings = Settings(
        gemini_api_key=secret_key,
        database_url=db_url,
    )

    rendered_repr = repr(settings)
    rendered_str = str(settings)

    for rendered in [rendered_repr, rendered_str]:
        if secret_key in rendered:
            return False, "Secret key was leaked in string representation!"
        if db_pass in rendered:
            return False, "Database password was leaked in string representation!"

    # Also verify that when accessed programmatically, the secret can be retrieved
    if settings.gemini_api_key.get_secret_value() != secret_key:
        return False, "SecretStr failed to store the underlying secret correctly"

    return True, "Printing and repr of Settings cleanly redacts secret values."


def main():
    print("=== Running T1-004 Verification ===")
    errors = []

    ok_env, msg_env = check_env_example()
    if not ok_env:
        errors.append(msg_env)
    else:
        print(f"[PASS] {msg_env}")

    ok_load, msg_load = check_settings_load_without_api_key()
    if not ok_load:
        errors.append(msg_load)
    else:
        print(f"[PASS] {msg_load}")

    ok_sec, msg_sec = check_secret_redaction()
    if not ok_sec:
        errors.append(msg_sec)
    else:
        print(f"[PASS] {msg_sec}")

    if errors:
        print("\n--- VERIFICATION FAILED ---")
        for e in errors:
            print(f"ERROR: {e}")
        return 1

    print("\n--- ALL T1-004 ACCEPTANCE CRITERIA VERIFIED ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
