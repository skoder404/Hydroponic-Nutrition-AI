"""Model registry service for startup loading, schema auditing, and runtime serving (T6-002).

Per TECH_RULES and T6-002 Acceptance Criteria:
- Builds ModelRegistry service from MODEL_REGISTRY_PATH at application startup.
- Verifies feature schema compatibility against audited crop sensor schemas.
- Incompatible feature lists yield SCHEMA_MISMATCH for the affected crop only.
- Missing or corrupt artifacts yield MODEL_UNAVAILABLE for the affected crop only.
- Failure of one crop does not break or substitute models for other crops.
- Provides health status reflecting registry integrity ('ok', 'degraded', 'unavailable').
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from backend.app.core.config import get_settings
from backend.app.core.errors import ModelUnavailableError, SchemaMismatchError
from backend.app.core.paths import (
    get_trusted_model_root,
    safe_load_model_artifact,
)
from backend.app.schemas.models import ModelSummary
from ml.registry.metadata_schema import ModelMetadata
from ml.registry.registry import ModelRegistry
from ml.sensor_models.schemas import get_sensor_schema

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """In-memory reference to an active verified model artifact."""

    crop: str
    modality: str
    model_obj: Any
    metadata: ModelMetadata
    status: str
    version: str
    task: str
    family: str
    experimental: bool
    validation_macro_f1: float | None


@dataclass
class FailedModelInfo:
    """Record of a promoted model that failed validation or artifact loading."""

    crop: str
    modality: str
    status: str  # "MODEL_UNAVAILABLE" or "SCHEMA_MISMATCH"
    version: str
    task: str
    family: str
    experimental: bool
    validation_macro_f1: float | None
    error_detail: str
    metadata: ModelMetadata | None = None


class ModelRegistryService:
    """Service managing promoted model lifecycle, schema audit, and inference access."""

    def __init__(self, registry_root: Path | str | None = None) -> None:
        self.root_dir = get_trusted_model_root(registry_root)
        self.registry = ModelRegistry(self.root_dir)
        self._models: dict[tuple[str, str], LoadedModel] = {}
        self._failed_models: dict[tuple[str, str], FailedModelInfo] = {}
        self._initialized: bool = False

    def load_promoted_models(self) -> None:
        """Scan active pointer files, audit schemas, and load valid model artifacts.

        Acceptance Criteria:
        - Removing one artifact breaks only that crop (status=MODEL_UNAVAILABLE).
        - Incompatible feature list is reported as SCHEMA_MISMATCH.
        - Other healthy crops continue serving without interruption.
        """
        self._models.clear()
        self._failed_models.clear()

        manifests_dir = self.registry.manifests_dir
        if not manifests_dir.is_dir():
            logger.warning(
                "Model manifests directory not found at '%s'. Registry is empty.",
                manifests_dir,
            )
            self._initialized = True
            return

        active_pointers = sorted(manifests_dir.glob("active_*.json"))
        logger.info(
            "Scanning model registry at '%s': found %d active pointer(s).",
            self.root_dir,
            len(active_pointers),
        )

        for pointer_path in active_pointers:
            self._process_active_pointer(pointer_path)

        self._initialized = True
        logger.info(
            "Model registry startup loading complete: %d loaded, %d failed.",
            len(self._models),
            len(self._failed_models),
        )

    def _process_active_pointer(self, pointer_path: Path) -> None:
        """Process a single active model pointer file with failure isolation."""
        try:
            pointer_data = json.loads(pointer_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to read active pointer file '%s': %s", pointer_path, exc)
            # Try to infer crop and modality from filename: active_<crop>_<modality>.json
            stem_parts = pointer_path.stem.split("_")
            crop = stem_parts[1] if len(stem_parts) > 1 else "unknown"
            modality = stem_parts[2] if len(stem_parts) > 2 else "sensor"
            self._failed_models[(crop, modality)] = FailedModelInfo(
                crop=crop,
                modality=modality,
                status="MODEL_UNAVAILABLE",
                version="unknown",
                task="classification",
                family="unknown",
                experimental=False,
                validation_macro_f1=None,
                error_detail=f"Pointer file corrupt: {exc}",
            )
            return

        crop = str(pointer_data.get("crop", "")).strip().lower()
        modality = str(pointer_data.get("modality", "")).strip().lower()
        active_ver = str(pointer_data.get("active_version", "1.0.0")).strip()

        if not crop or not modality:
            logger.error("Active pointer '%s' missing crop or modality.", pointer_path)
            return

        key = (crop, modality)

        # 1. Resolve and validate metadata
        meta_rel = pointer_data.get("active_metadata_path")
        if not meta_rel:
            self._failed_models[key] = FailedModelInfo(
                crop=crop,
                modality=modality,
                status="MODEL_UNAVAILABLE",
                version=active_ver,
                task="classification",
                family="unknown",
                experimental=False,
                validation_macro_f1=None,
                error_detail="Active pointer missing 'active_metadata_path'.",
            )
            return

        meta_path = self.root_dir / meta_rel
        if not meta_path.is_file():
            self._failed_models[key] = FailedModelInfo(
                crop=crop,
                modality=modality,
                status="MODEL_UNAVAILABLE",
                version=active_ver,
                task="classification",
                family="unknown",
                experimental=False,
                validation_macro_f1=None,
                error_detail=f"Metadata file '{meta_rel}' not found on disk.",
            )
            logger.warning("Metadata file missing for %s/%s: %s", crop, modality, meta_path)
            return

        try:
            metadata = ModelMetadata.load(meta_path)
        except Exception as exc:
            self._failed_models[key] = FailedModelInfo(
                crop=crop,
                modality=modality,
                status="MODEL_UNAVAILABLE",
                version=active_ver,
                task="classification",
                family="unknown",
                experimental=False,
                validation_macro_f1=None,
                error_detail=f"Invalid model metadata: {exc}",
            )
            logger.error("Failed to load metadata for %s/%s: %s", crop, modality, exc)
            return

        val_f1 = (
            metadata.validation_metrics.get("macro_f1")
            if isinstance(metadata.validation_metrics, dict)
            else None
        )

        # 2. Audit feature schema compatibility (Criterion 2)
        if modality == "sensor":
            try:
                sensor_schema = get_sensor_schema(crop)
                expected_features = list(sensor_schema.feature_names)
            except ValueError:
                expected_features = None

            if expected_features is not None:
                actual_features = metadata.ordered_features
                if actual_features is None or list(actual_features) != expected_features:
                    err_msg = (
                        f"Feature schema mismatch for {crop}/{modality}. "
                        f"Expected {expected_features}, but model artifact requires {actual_features}."
                    )
                    self._failed_models[key] = FailedModelInfo(
                        crop=crop,
                        modality=modality,
                        status="SCHEMA_MISMATCH",
                        version=metadata.artifact_version,
                        task=metadata.task,
                        family=metadata.family,
                        experimental=metadata.experimental,
                        validation_macro_f1=val_f1,
                        error_detail=err_msg,
                        metadata=metadata,
                    )
                    logger.warning(
                        "Schema mismatch detected for %s/%s: %s", crop, modality, err_msg
                    )
                    return

        # 3. Resolve and load model artifact safely (Criterion 1)
        art_rel = pointer_data.get("active_artifact_path") or metadata.artifact_file
        art_path = self.root_dir / art_rel

        if not art_path.is_file():
            err_msg = f"Model artifact '{art_rel}' not found on disk."
            self._failed_models[key] = FailedModelInfo(
                crop=crop,
                modality=modality,
                status="MODEL_UNAVAILABLE",
                version=metadata.artifact_version,
                task=metadata.task,
                family=metadata.family,
                experimental=metadata.experimental,
                validation_macro_f1=val_f1,
                error_detail=err_msg,
                metadata=metadata,
            )
            logger.warning(
                "Model artifact missing for %s/%s: %s", crop, modality, err_msg
            )
            return

        try:
            model_obj = safe_load_model_artifact(art_path, trusted_root=self.root_dir)
        except Exception as exc:
            err_msg = f"Model artifact corrupt or failed to deserialize: {exc}"
            self._failed_models[key] = FailedModelInfo(
                crop=crop,
                modality=modality,
                status="MODEL_UNAVAILABLE",
                version=metadata.artifact_version,
                task=metadata.task,
                family=metadata.family,
                experimental=metadata.experimental,
                validation_macro_f1=val_f1,
                error_detail=err_msg,
                metadata=metadata,
            )
            logger.error("Failed to deserialize artifact for %s/%s: %s", crop, modality, exc)
            return

        # 4. Successfully loaded and verified
        self._models[key] = LoadedModel(
            crop=crop,
            modality=modality,
            model_obj=model_obj,
            metadata=metadata,
            status=metadata.status,  # "promoted"
            version=metadata.artifact_version,
            task=metadata.task,
            family=metadata.family,
            experimental=metadata.experimental,
            validation_macro_f1=val_f1,
        )
        logger.info(
            "Loaded promoted model: %s/%s v%s (family=%s, val_macro_f1=%s, experimental=%s)",
            crop,
            modality,
            metadata.artifact_version,
            metadata.family,
            f"{val_f1:.4f}" if val_f1 is not None else "N/A",
            metadata.experimental,
        )

    def get_model(self, crop: str, modality: str = "sensor") -> tuple[Any, ModelMetadata]:
        """Retrieve the in-memory active model object and metadata.

        Raises:
            SchemaMismatchError: If the model has an incompatible feature list.
            ModelUnavailableError: If the model artifact is missing or corrupted.
        """
        if not self._initialized:
            self.load_promoted_models()

        clean_crop = crop.strip().lower()
        clean_modality = modality.strip().lower()
        key = (clean_crop, clean_modality)

        if key in self._failed_models:
            failure = self._failed_models[key]
            if failure.status == "SCHEMA_MISMATCH":
                raise SchemaMismatchError(
                    message=failure.error_detail,
                    details={
                        "crop": clean_crop,
                        "modality": clean_modality,
                        "code": "SCHEMA_MISMATCH",
                    },
                )
            raise ModelUnavailableError(
                message=failure.error_detail,
                details={
                    "crop": clean_crop,
                    "modality": clean_modality,
                    "code": "MODEL_UNAVAILABLE",
                },
            )

        if key not in self._models:
            raise ModelUnavailableError(
                message=f"No active promoted model registered for crop '{clean_crop}' and modality '{clean_modality}'.",
                details={
                    "crop": clean_crop,
                    "modality": clean_modality,
                    "code": "MODEL_UNAVAILABLE",
                },
            )

        loaded = self._models[key]
        return loaded.model_obj, loaded.metadata

    def get_model_summaries(
        self,
        crop: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> list[ModelSummary]:
        """Expose summary of all promoted models, including failures and experimental status."""
        if not self._initialized:
            self.load_promoted_models()

        clean_crop = crop.strip().lower() if crop else None
        clean_modality = modality.strip().lower() if modality else None

        results: list[ModelSummary] = []

        # Add loaded models
        for (c, m), loaded in self._models.items():
            if clean_crop and c != clean_crop:
                continue
            if clean_modality and m != clean_modality:
                continue

            results.append(
                ModelSummary(
                    crop=loaded.crop,
                    task=loaded.task,
                    modality=loaded.modality,
                    family=loaded.family,
                    version=loaded.version,
                    status=loaded.status,
                    validation_macro_f1=loaded.validation_macro_f1,
                    experimental=loaded.experimental,
                    error_detail=None,
                    classes=loaded.metadata.classes,
                    ordered_features=loaded.metadata.ordered_features,
                    display_mapping=loaded.metadata.display_mapping,
                    split_protocol=loaded.metadata.split_protocol,
                    metrics=loaded.metadata.validation_metrics,
                )
            )

        # Add failed models (isolated failures)
        for (c, m), failed in self._failed_models.items():
            if clean_crop and c != clean_crop:
                continue
            if clean_modality and m != clean_modality:
                continue

            results.append(
                ModelSummary(
                    crop=failed.crop,
                    task=failed.task,
                    modality=failed.modality,
                    family=failed.family,
                    version=failed.version,
                    status=failed.status,
                    validation_macro_f1=failed.validation_macro_f1,
                    experimental=failed.experimental,
                    error_detail=failed.error_detail,
                    classes=failed.metadata.classes if failed.metadata else None,
                    ordered_features=(
                        failed.metadata.ordered_features if failed.metadata else None
                    ),
                    display_mapping=(
                        failed.metadata.display_mapping if failed.metadata else None
                    ),
                    split_protocol=(
                        failed.metadata.split_protocol if failed.metadata else None
                    ),
                    metrics=(
                        failed.metadata.validation_metrics if failed.metadata else None
                    ),
                )
            )

        # Sort predictably by crop, modality, version
        results.sort(key=lambda s: (s.crop, s.modality, s.version))
        return results

    def get_health_status(self) -> str:
        """Evaluate overall registry health state for /health endpoint.

        Returns:
            - 'ok': All configured promoted models loaded and validated cleanly.
            - 'degraded': At least one model is operational, but one or more failed.
            - 'unavailable': No operational models are available.
        """
        if not self._initialized:
            self.load_promoted_models()

        total = len(self._models) + len(self._failed_models)
        if total == 0:
            return "unavailable"

        if len(self._failed_models) == 0:
            return "ok"

        if len(self._models) > 0:
            return "degraded"

        return "unavailable"


# Singleton management
_service_instance: ModelRegistryService | None = None


def get_model_registry_service(
    registry_root: Path | str | None = None,
) -> ModelRegistryService:
    """Return the application singleton ModelRegistryService."""
    global _service_instance
    if _service_instance is None or registry_root is not None:
        _service_instance = ModelRegistryService(registry_root=registry_root)
    return _service_instance


def reset_model_registry_service() -> None:
    """Reset the singleton instance (primarily for isolated test fixtures)."""
    global _service_instance
    _service_instance = None
