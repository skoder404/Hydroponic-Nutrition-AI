"""Application configuration loader using pydantic-settings."""
from __future__ import annotations

import re
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable loading and secret redaction."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application Environment
    environment: str = Field(default="dev", description="Application environment: dev, demo, test")
    host: str = Field(default="127.0.0.1", description="Backend bind host")
    port: int = Field(default=8000, description="Backend bind port")

    # Network & CORS
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins",
    )

    # Database Connection
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/hydroponics",
        description="PostgreSQL connection string",
    )

    # Google Gemini API (Optional: rule fallback operates when missing)
    gemini_api_key: SecretStr | None = Field(
        default=None,
        description="Optional Gemini API key for narrative recommendations",
    )
    gemini_model: str = Field(default="gemini-1.5-flash", description="Gemini model name")

    # Local Storage and Registry Paths
    model_registry_path: str = Field(default="models/manifests", description="Path to model manifests")
    image_data_path: str = Field(default="image_data", description="Path to raw image data")
    sensor_data_path: str = Field(default="sensor_data", description="Path to raw sensor data")
    processed_data_path: str = Field(default="processed_data", description="Path to processed data")

    # Campus Server & Camera Integration (IP/Host in .env only)
    campus_server_url: str = Field(default="http://127.0.0.1:8080", description="Campus device server URL")
    campus_camera_url: str = Field(default="http://127.0.0.1:8080/feed", description="Campus camera feed URL")
    campus_poll_interval_s: int = Field(default=30, description="Polling interval in seconds")
    campus_max_age_s: int = Field(default=120, description="Max acceptable data age in seconds")
    campus_request_timeout_s: int = Field(default=5, description="Timeout for campus server requests")

    # Uploads and Inference
    max_upload_mb: int = Field(default=15, description="Maximum image upload size in MB")
    device_adapter: str = Field(default="mock", description="Hardware device adapter mode: mock, campus, manual")
    force_cpu: bool = Field(default=False, description="Force CPU execution regardless of GPU presence")
    sam2_checkpoint: str = Field(default="models/image/sam2_hiera_small.pt", description="Path to SAM 2.1 weights")
    sam2_config: str = Field(default="sam2_hiera_s.yaml", description="Path to SAM 2.1 model config")

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    def __repr__(self) -> str:
        """String representation with guaranteed redaction of all secrets."""
        redacted_fields = {}
        for key, val in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(val, SecretStr):
                redacted_fields[key] = "**********"
            elif "key" in key.lower() or "secret" in key.lower() or "token" in key.lower():
                redacted_fields[key] = "**********" if val is not None else None
            elif key.lower() == "database_url" and isinstance(val, str):
                # Redact credentials in database URL
                redacted_fields[key] = re.sub(r":([^:@]+)@", ":***@", val)
            else:
                redacted_fields[key] = val

        rendered = ", ".join(f"{k}={v!r}" for k, v in redacted_fields.items())
        return f"{self.__class__.__name__}({rendered})"

    def __str__(self) -> str:
        return self.__repr__()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
