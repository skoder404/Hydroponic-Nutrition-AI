"""Model registry for artifact persistence, versioning, promotion, and rollback (T4-009).

Adheres to TECH_RULES and Canonical Overview Sections 24.1 & 24.2:
- Versioned directory structure under models/<modality>/<crop>/v<version>/.
- Explicit CLI-only promotion (never 'newest wins').
- Rollback support with preservation of previous known-good version.
- Strict feature-schema compatibility verification on load (raises SCHEMA_MISMATCH).
- Model artifact loading strictly guarded via backend.app.core.paths trusted path guard.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.core.paths import (
    get_trusted_model_root,
    safe_load_model_artifact,
    validate_model_path,
)
from ml.registry.metadata_schema import ModelMetadata


class SchemaMismatchError(ValueError):
    """Raised when an artifact feature schema does not match the expected schema."""

    error_code: str = "SCHEMA_MISMATCH"

    def __init__(
        self,
        message: str,
        expected: Sequence[str] | None = None,
        actual: Sequence[str] | None = None,
    ) -> None:
        full_message = f"SCHEMA_MISMATCH: {message}"
        super().__init__(full_message)
        self.message = message
        self.expected = list(expected) if expected is not None else []
        self.actual = list(actual) if actual is not None else []


class PromotionPermissionError(PermissionError):
    """Raised when model promotion or rollback is attempted outside the approved CLI."""

    error_code: str = "PROMOTION_CLI_ONLY"


class ModelRegistry:
    """Manages versioned model artifacts, metadata, explicit promotion, and rollback."""

    def __init__(self, root_dir: Path | str | None = None) -> None:
        self.root_dir = Path(root_dir).resolve() if root_dir else get_trusted_model_root()
        self.manifests_dir = self.root_dir / "manifests"
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def _get_active_pointer_file(self, crop: str, modality: str) -> Path:
        clean_crop = crop.strip().lower()
        clean_modality = modality.strip().lower()
        return self.manifests_dir / f"active_{clean_crop}_{clean_modality}.json"

    def _get_version_dir(self, crop: str, modality: str, version: str) -> Path:
        clean_crop = crop.strip().lower()
        clean_modality = modality.strip().lower()
        clean_version = version.strip().lstrip("v")
        return self.root_dir / clean_modality / clean_crop / f"v{clean_version}"

    def save_model(
        self,
        model_obj: Any,
        metadata: ModelMetadata,
        overwrite: bool = False,
    ) -> tuple[Path, Path]:
        """Save a model artifact and its metadata in a versioned directory.

        Parameters
        ----------
        model_obj:
            The in-memory model (sklearn estimator, PyTorch state_dict/module, etc.).
        metadata:
            Validated ModelMetadata instance matching Overview §24.1.
        overwrite:
            Whether to overwrite an existing version directory.

        Returns
        -------
        tuple[Path, Path]:
            (artifact_path, metadata_path)
        """
        version_dir = self._get_version_dir(metadata.crop, metadata.modality, metadata.artifact_version)
        if version_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Version v{metadata.artifact_version} already exists for {metadata.crop}/{metadata.modality} at '{version_dir}'."
            )
        version_dir.mkdir(parents=True, exist_ok=True)

        # Determine target artifact path
        artifact_filename = Path(metadata.artifact_file).name
        artifact_path = version_dir / artifact_filename
        metadata_path = version_dir / "metadata.json"

        # Serialize model object
        suffix = artifact_path.suffix.lower()
        if suffix in (".joblib", ".pkl", ".pickle"):
            import joblib

            joblib.dump(model_obj, artifact_path)
        elif suffix in (".pt", ".pth"):
            import torch

            torch.save(model_obj, artifact_path)
        elif suffix == ".json":
            if isinstance(model_obj, (dict, list)):
                artifact_path.write_text(json.dumps(model_obj, indent=2), encoding="utf-8")
            else:
                artifact_path.write_text(str(model_obj), encoding="utf-8")
        else:
            # Fallback to joblib for generic Python objects
            import joblib

            joblib.dump(model_obj, artifact_path)

        # Update metadata artifact_file to be relative to root_dir
        rel_artifact = artifact_path.relative_to(self.root_dir).as_posix()
        updated_meta_dict = metadata.model_dump()
        updated_meta_dict["artifact_file"] = rel_artifact
        saved_metadata = ModelMetadata.model_validate(updated_meta_dict)
        saved_metadata.save(metadata_path)

        return artifact_path, metadata_path

    def promote(
        self,
        crop: str,
        modality: str,
        version: str,
        called_from_cli: bool = False,
    ) -> dict[str, Any]:
        """Explicitly promote a candidate model version to active status.

        Acceptance Criterion 3: Promotion is CLI-only.
        """
        if not called_from_cli and os.environ.get("ALLOW_API_PROMOTION") != "1":
            raise PromotionPermissionError(
                "Model promotion is restricted to CLI commands only (scripts/promote_model.py). "
                "API-driven or automatic 'newest wins' promotion is prohibited per TECH_RULES."
            )

        clean_crop = crop.strip().lower()
        clean_modality = modality.strip().lower()
        clean_version = version.strip().lstrip("v")

        version_dir = self._get_version_dir(clean_crop, clean_modality, clean_version)
        metadata_path = version_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Candidate model version v{clean_version} not found at '{metadata_path}'."
            )

        # Validate metadata and artifact
        meta = ModelMetadata.load(metadata_path)
        artifact_path = self.root_dir / meta.artifact_file
        validate_model_path(artifact_path, trusted_root=self.root_dir, must_exist=True)

        # Mark metadata status as promoted
        meta_dict = meta.model_dump()
        meta_dict["status"] = "promoted"
        promoted_meta = ModelMetadata.model_validate(meta_dict)
        promoted_meta.save(metadata_path)

        # Read active pointer if already existing
        pointer_file = self._get_active_pointer_file(clean_crop, clean_modality)
        previous_version: str | None = None
        previous_metadata_path: str | None = None
        previous_artifact_path: str | None = None

        if pointer_file.is_file():
            try:
                active_data = json.loads(pointer_file.read_text(encoding="utf-8"))
                previous_version = active_data.get("active_version")
                previous_metadata_path = active_data.get("active_metadata_path")
                previous_artifact_path = active_data.get("active_artifact_path")
            except (json.JSONDecodeError, OSError):
                # Ignore corrupt or unreadable pointer during promotion
                previous_version = None

        # Update active pointer
        active_payload = {
            "crop": clean_crop,
            "modality": clean_modality,
            "active_version": clean_version,
            "active_metadata_path": str(metadata_path.resolve().relative_to(self.root_dir).as_posix()),
            "active_artifact_path": str(artifact_path.resolve().relative_to(self.root_dir).as_posix()),
            "previous_version": previous_version,
            "previous_metadata_path": previous_metadata_path,
            "previous_artifact_path": previous_artifact_path,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        pointer_file.write_text(json.dumps(active_payload, indent=2), encoding="utf-8")
        return active_payload

    def rollback(
        self,
        crop: str,
        modality: str,
        called_from_cli: bool = False,
    ) -> dict[str, Any]:
        """Roll back to the previous known-good model version.

        Acceptance Criterion 1: Promote then rollback restores the previous version.
        Acceptance Criterion 3: Promotion/rollback is CLI-only.
        """
        if not called_from_cli and os.environ.get("ALLOW_API_PROMOTION") != "1":
            raise PromotionPermissionError(
                "Model rollback is restricted to CLI commands only (scripts/promote_model.py)."
            )

        clean_crop = crop.strip().lower()
        clean_modality = modality.strip().lower()
        pointer_file = self._get_active_pointer_file(clean_crop, clean_modality)

        if not pointer_file.is_file():
            raise FileNotFoundError(f"No active model found for {clean_crop}/{clean_modality}.")

        active_data = json.loads(pointer_file.read_text(encoding="utf-8"))
        prev_version = active_data.get("previous_version")
        prev_meta_path_rel = active_data.get("previous_metadata_path")
        prev_art_path_rel = active_data.get("previous_artifact_path")

        if not prev_version or not prev_meta_path_rel or not prev_art_path_rel:
            raise RuntimeError(
                f"No previous version available for rollback of {clean_crop}/{clean_modality}."
            )

        # Verify previous version still exists on disk
        prev_meta_path = self.root_dir / prev_meta_path_rel
        prev_art_path = self.root_dir / prev_art_path_rel
        validate_model_path(prev_art_path, trusted_root=self.root_dir, must_exist=True)

        # Mark current active as deprecated or candidate
        curr_meta_path = self.root_dir / active_data["active_metadata_path"]
        if curr_meta_path.is_file():
            try:
                curr_meta = ModelMetadata.load(curr_meta_path)
                c_dict = curr_meta.model_dump()
                c_dict["status"] = "candidate"
                ModelMetadata.model_validate(c_dict).save(curr_meta_path)
            except (OSError, ValueError):
                # If current metadata cannot be rewritten, proceed with pointer rollback
                pass

        # Mark previous as promoted
        prev_meta = ModelMetadata.load(prev_meta_path)
        p_dict = prev_meta.model_dump()
        p_dict["status"] = "promoted"
        ModelMetadata.model_validate(p_dict).save(prev_meta_path)

        # Swap active and previous in pointer file
        rolled_back_payload = {
            "crop": clean_crop,
            "modality": clean_modality,
            "active_version": prev_version,
            "active_metadata_path": prev_meta_path_rel,
            "active_artifact_path": prev_art_path_rel,
            "previous_version": active_data["active_version"],
            "previous_metadata_path": active_data["active_metadata_path"],
            "previous_artifact_path": active_data["active_artifact_path"],
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "rolled_back": True,
        }
        pointer_file.write_text(json.dumps(rolled_back_payload, indent=2), encoding="utf-8")
        return rolled_back_payload

    def get_active_pointer(self, crop: str, modality: str) -> dict[str, Any]:
        """Retrieve the active pointer dictionary for crop and modality."""
        clean_crop = crop.strip().lower()
        clean_modality = modality.strip().lower()
        pointer_file = self._get_active_pointer_file(clean_crop, clean_modality)
        if not pointer_file.is_file():
            raise FileNotFoundError(
                f"No active promoted model for crop '{clean_crop}' and modality '{clean_modality}'."
            )
        return json.loads(pointer_file.read_text(encoding="utf-8"))

    def get_active_metadata(self, crop: str, modality: str) -> ModelMetadata:
        """Load the ModelMetadata of the currently promoted model."""
        pointer = self.get_active_pointer(crop, modality)
        meta_path = self.root_dir / pointer["active_metadata_path"]
        return ModelMetadata.load(meta_path)

    def load_model(
        self,
        crop: str,
        modality: str,
        version: str | None = None,
        expected_features: Sequence[str] | None = None,
    ) -> tuple[Any, ModelMetadata]:
        """Load a model artifact and metadata with schema validation and path guard.

        Acceptance Criterion 2: Loading an artifact with a mismatched feature list fails with SCHEMA_MISMATCH.
        Load goes through trusted path guard.
        """
        clean_crop = crop.strip().lower()
        clean_modality = modality.strip().lower()

        if version is not None:
            clean_ver = version.strip().lstrip("v")
            version_dir = self._get_version_dir(clean_crop, clean_modality, clean_ver)
            meta_path = version_dir / "metadata.json"
            if not meta_path.is_file():
                raise FileNotFoundError(f"Model version v{clean_ver} not found for {clean_crop}/{clean_modality}.")
            metadata = ModelMetadata.load(meta_path)
            artifact_path = self.root_dir / metadata.artifact_file
        else:
            pointer = self.get_active_pointer(clean_crop, clean_modality)
            meta_path = self.root_dir / pointer["active_metadata_path"]
            metadata = ModelMetadata.load(meta_path)
            artifact_path = self.root_dir / pointer["active_artifact_path"]

        # Verify feature schema compatibility on load (Acceptance Criterion 2)
        if expected_features is not None:
            if metadata.ordered_features is None:
                raise SchemaMismatchError(
                    f"Model for {clean_crop}/{clean_modality} does not define ordered_features, "
                    f"but expected {list(expected_features)}.",
                    expected=expected_features,
                    actual=[],
                )
            if list(metadata.ordered_features) != list(expected_features):
                raise SchemaMismatchError(
                    f"Feature schema mismatch for {clean_crop}/{clean_modality}. "
                    f"Expected {list(expected_features)}, but model expects {list(metadata.ordered_features)}.",
                    expected=expected_features,
                    actual=metadata.ordered_features,
                )

        # Load artifact strictly through the trusted path guard
        model_obj = safe_load_model_artifact(artifact_path, trusted_root=self.root_dir)
        return model_obj, metadata

    def list_versions(self, crop: str, modality: str) -> list[str]:
        """List all registered version strings for crop and modality."""
        clean_crop = crop.strip().lower()
        clean_modality = modality.strip().lower()
        modality_dir = self.root_dir / clean_modality / clean_crop
        if not modality_dir.is_dir():
            return []
        versions = []
        for d in sorted(modality_dir.iterdir()):
            if d.is_dir() and d.name.startswith("v") and (d / "metadata.json").is_file():
                versions.append(d.name.lstrip("v"))
        return versions
