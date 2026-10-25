"""Trusted path validation and secure model artifact loader."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings


class UntrustedModelPathError(ValueError):
    """Raised when an artifact path attempts to escape the trusted model directory."""


def get_trusted_model_root(configured_path: str | Path | None = None) -> Path:
    """Return the resolved root directory for trusted model artifacts.

    If configured_path is not specified, uses get_settings().model_registry_path.
    If the configured path points to 'manifests', uses its parent 'models' directory
    to encompass image, sensor, and manifest artifacts.
    """
    if configured_path is not None:
        target = Path(configured_path)
    else:
        settings = get_settings()
        reg = Path(settings.model_registry_path)
        # If pointing to models/manifests, trusted root covers the parent models/ directory
        target = reg.parent if reg.name in ("manifests", "manifest") else reg

    return target.resolve()


def validate_model_path(
    artifact_path: str | Path,
    trusted_root: str | Path | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve and verify that artifact_path lies strictly under the trusted model root.

    Rejects directory traversal attacks (../) and symlinks whose targets resolve
    outside the trusted root.

    Args:
        artifact_path: Input file path or relative path to check.
        trusted_root: Optional override for the trusted model directory root.
        must_exist: If True, raises FileNotFoundError if the resolved file does not exist.

    Returns:
        The resolved Path object guaranteed to reside within trusted_root.

    Raises:
        UntrustedModelPathError: If the resolved path escapes the trusted root.
        FileNotFoundError: If must_exist is True and the file is not on disk.
    """
    base_root = get_trusted_model_root(trusted_root)
    candidate = Path(artifact_path)

    # Fully resolve symlinks and normalise path (realpath)
    resolved_candidate = candidate.resolve()

    try:
        resolved_candidate.relative_to(base_root)
    except ValueError as exc:
        raise UntrustedModelPathError(
            f"Access denied: path '{artifact_path}' resolves to '{resolved_candidate}', "
            f"which is outside trusted directory root '{base_root}'."
        ) from exc

    if must_exist and not resolved_candidate.exists():
        raise FileNotFoundError(f"Model artifact not found at '{resolved_candidate}'")

    return resolved_candidate


def safe_load_model_artifact(
    artifact_path: str | Path,
    trusted_root: str | Path | None = None,
    loader_fn: Callable[[Path], Any] | None = None,
) -> Any:
    """Load a model artifact only after strict path validation through the guard.

    Guarantees no file outside the trusted root is ever opened, unpickled, or loaded.

    Args:
        artifact_path: Path to model artifact.
        trusted_root: Optional override for trusted root directory.
        loader_fn: Custom callable to execute after path validation. If None,
                   dispatches based on file extension (.json, .joblib, .pt).

    Returns:
        The deserialized model or artifact data.
    """
    valid_path = validate_model_path(artifact_path, trusted_root=trusted_root, must_exist=True)

    if loader_fn is not None:
        return loader_fn(valid_path)

    suffix = valid_path.suffix.lower()
    if suffix in (".joblib", ".pkl", ".pickle"):
        import joblib
        return joblib.load(valid_path)

    if suffix in (".pt", ".pth"):
        import torch
        try:
            return torch.load(valid_path, weights_only=True)
        except TypeError:
            return torch.load(valid_path)

    if suffix == ".json":
        import json
        with open(valid_path, encoding="utf-8") as fh:
            return json.load(fh)

    raise ValueError(f"Unsupported model artifact format: {suffix}")
