"""Structured JSON logging with request-ID tracking and secret redaction.

Adheres to TECH_RULES:
- Structured JSON logs (fields: event, request_id, level, timestamp, etc.).
- Never log secrets, passwords, raw API keys, or credentials.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# Context variable for request ID tracking across async calls
request_id_ctx_var: ContextVar[str] = ContextVar("request_id", default="-")

# Secret redaction patterns ordered by specificity
REDACTION_PATTERNS = [
    # 1. Bearer tokens: Bearer <token> or Authorization: Bearer <token>
    (re.compile(r"(?i)(bearer\s+|authorization\s*:\s*bearer\s+)[A-Za-z0-9_\-\.]+"), r"\g<1>***"),
    # 2. Database connection strings with passwords: postgresql://user:password@host
    (re.compile(r"://([^:@]+):([^@]+)@"), r"://\1:***@"),
    # 3. Google AI / Gemini API keys: AIzaSy...
    (re.compile(r"AIzaSy[A-Za-z0-9_-]{33}"), "***REDACTED_GEMINI_KEY***"),
    # 4. Common secret key-values: api_key=..., password=..., secret=..., token=...
    (
        re.compile(r"(?i)(api[_-]?key|password|secret|token|auth(?:orization)?)\s*[:=]\s*['\"]?([^\s'\",]+)['\"]?"),
        r"\1=***",
    ),
]


def redact_secrets(text: str) -> str:
    """Scrub known credentials, passwords, and API keys from a string."""
    for pattern, replacement in REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def sanitize_value(val: Any) -> Any:
    """Recursively sanitize dicts, lists, and strings for JSON output."""
    if isinstance(val, str):
        return redact_secrets(val)
    if isinstance(val, dict):
        sanitized = {}
        for k, v in val.items():
            if any(s in k.lower() for s in ("password", "secret", "api_key", "token", "authorization")):
                sanitized[k] = "***"
            else:
                sanitized[k] = sanitize_value(v)
        return sanitized
    if isinstance(val, (list, tuple)):
        return [sanitize_value(v) for v in val]
    return val


class JSONFormatter(logging.Formatter):
    """Formats log records as structured single-line JSON with request-ID."""

    def format(self, record: logging.LogRecord) -> str:
        # Resolve request_id from record or contextvar
        req_id = getattr(record, "request_id", None) or request_id_ctx_var.get()

        message = record.getMessage()
        sanitized_event = redact_secrets(message)

        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": sanitized_event,
            "request_id": req_id,
            "logger": record.name,
        }

        # Include additional structured fields if passed via extra
        for attr in ("crop", "model_id", "latency_ms", "device", "status_code", "method", "path"):
            val = getattr(record, attr, None)
            if val is not None:
                log_data[attr] = sanitize_value(val)

        # Include exception information if present
        if record.exc_info:
            log_data["exception"] = redact_secrets(self.formatException(record.exc_info))

        return json.dumps(log_data, default=str)


def setup_logging(log_level: str = "INFO", stream: Any = None) -> None:
    """Configure root and application loggers to output structured JSON."""
    root_logger = logging.getLogger()
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicate or unstructured output
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)

    # Align uvicorn and fastapi loggers
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        u_logger = logging.getLogger(logger_name)
        u_logger.handlers.clear()
        u_logger.propagate = True
