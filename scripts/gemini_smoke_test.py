#!/usr/bin/env python3
"""Gemini API smoke test script (T1-008).

Reads GEMINI_API_KEY and GEMINI_MODEL from environment / .env.
Ensures:
1. API key is NEVER printed, logged, or exposed in tracebacks.
2. If GEMINI_API_KEY is missing or unconfigured, exits cleanly with code 0
   (demonstrating the rule-based fallback provider is default).
3. If configured, executes a single lightweight prompt and reports OK/FAIL with latency.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path so backend modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backend.app.core.config import get_settings
    settings = get_settings()
    _RAW_KEY = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
    _MODEL = settings.gemini_model or "gemini-1.5-flash"
except Exception:  # noqa: BLE001
    # Direct environment fallback if config loader is unavailable
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    _RAW_KEY = os.getenv("GEMINI_API_KEY")
    _MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def _sanitize_text(text: str, secret: str | None) -> str:
    """Strip secret token occurrences from text."""
    if not secret:
        return text
    return text.replace(secret, "**********")


def run_smoke_test() -> int:
    """Execute Gemini smoke test with strict secret protection."""
    # 1. Check if key is absent or placeholder
    if not _RAW_KEY or not _RAW_KEY.strip() or _RAW_KEY.strip().startswith("your_"):
        print("[INFO] GEMINI_API_KEY is not configured.")
        print("[INFO] Local rule-based recommendation fallback remains active as default.")
        return 0

    print(f"[INFO] Probing Gemini API using model '{_MODEL}'...")

    # 2. Invoke Gemini via google-genai SDK (or fallback to google.generativeai)
    start_time = time.perf_counter()
    try:
        try:
            from google import genai
            client = genai.Client(api_key=_RAW_KEY)
            response = client.models.generate_content(
                model=_MODEL,
                contents="Respond with the single word 'healthy' for a test probe.",
            )
            response_text = response.text.strip() if hasattr(response, "text") and response.text else "OK"
        except ImportError:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=_RAW_KEY)
            model = legacy_genai.GenerativeModel(_MODEL)
            response = model.generate_content("Respond with the single word 'healthy' for a test probe.")
            response_text = response.text.strip() if hasattr(response, "text") and response.text else "OK"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        print("[OK] Gemini API smoke test passed.")
        print(f"     Model: {_MODEL}")
        print(f"     Latency: {elapsed_ms:.1f} ms")
        print(f"     Response: {response_text[:40]}")
        return 0

    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        err_msg = _sanitize_text(str(exc), _RAW_KEY)
        print(f"[FAIL] Gemini API probe failed after {elapsed_ms:.1f} ms.")
        print(f"       Reason: {err_msg}")
        print("[INFO] Rule-based recommendation fallback remains active.")
        return 1


if __name__ == "__main__":
    sys.exit(run_smoke_test())
