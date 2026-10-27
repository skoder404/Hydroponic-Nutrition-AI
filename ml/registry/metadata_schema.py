"""Model metadata schema and validation adhering to Overview Section 24.1 (T4-008).

Provides strongly-typed Pydantic schema for model artifact registration and audit:
- Complete Section 24.1 fields: crop, task, modality, family, commit, dataset hash,
  classes, display mapping, split protocol/counts, feature schemas, input units,
  image conventions, configs, metrics, dependency versions, training date.
- Explicit lifecycle control with 'experimental' boolean flag and 'status' enum.
- Rejects missing mandatory fields, invalid taxonomies, and unmapped classes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CropType = Literal["lettuce", "spinach", "tomato", "tomato_live"]
TaskType = Literal["classification", "segmentation", "feature_extraction"]
ModalityType = Literal["image", "sensor", "feature"]
ModelStatus = Literal["candidate", "promoted", "deprecated", "archived"]
ChannelOrderType = Literal["RGB", "BGR", "grayscale"]


class ModelMetadata(BaseModel):
    """Strongly-typed metadata required for every model artifact per Overview Section 24.1."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Core Identifiers
    model_id: str = Field(
        ..., min_length=3, description="Unique identifier for the model artifact."
    )
    crop: CropType = Field(..., description="Crop family: lettuce, spinach, or tomato.")
    task: TaskType = Field(
        ..., description="Machine learning task performed by the model."
    )
    modality: ModalityType = Field(
        ..., description="Input modality: image, sensor, or feature-based image."
    )
    family: str = Field(
        ...,
        min_length=2,
        description="Model family or architecture (e.g. RandomForest, EfficientNetB0).",
    )
    artifact_version: str = Field(
        ...,
        min_length=1,
        description="Semantic version of the model artifact (e.g. 1.0.0).",
    )
    artifact_file: str = Field(
        ...,
        min_length=1,
        description="Filename or relative path to the serialized model file.",
    )

    # Provenance & Tracking
    commit: str = Field(
        ..., min_length=7, description="Git commit hash of the training codebase."
    )
    training_date: str = Field(
        ..., description="ISO 8601 UTC timestamp when model training completed."
    )
    dataset_manifest_hash: str = Field(
        ...,
        min_length=16,
        description="SHA-256 hash of the dataset manifest used for training.",
    )
    rule_version: str = Field(
        ..., min_length=1, description="Version of the rule/label definition applied."
    )
    preprocessing_version: str = Field(
        ..., min_length=1, description="Version of the preprocessing pipeline applied."
    )

    # Lifecycle & Governance (Acceptance Criterion 2)
    experimental: bool = Field(
        ...,
        description="Flag indicating whether model is experimental or production-ready.",
    )
    status: ModelStatus = Field(
        ...,
        description="Lifecycle status: candidate, promoted, deprecated, or archived.",
    )

    # Taxonomy & Display Mapping
    classes: list[str] = Field(
        ..., min_length=2, description="Ordered list of target class labels."
    )
    display_mapping: dict[str, str] = Field(
        ..., description="Mapping of raw class label to user-facing display label."
    )

    # Dataset & Partitioning
    split_protocol: str = Field(
        ...,
        min_length=5,
        description="Protocol used for partitioning train/validation/test sets.",
    )
    split_counts: dict[str, int] = Field(
        ...,
        description="Sample counts per partition: {'train': N, 'validation': N, 'test': N}.",
    )

    # Feature & Sensor Specifications (Mandatory for sensor/feature models)
    ordered_features: list[str] | None = Field(
        default=None,
        description="Ordered list of feature column names for tabular/sensor models.",
    )
    feature_units: dict[str, str] | None = Field(
        default=None, description="Units associated with each ordered feature."
    )
    feature_set: str | None = Field(
        default=None,
        description="Feature set variant identifier (e.g. 'live_compatible', 'full_8').",
    )

    # Image Specifications (Mandatory for image models)
    input_unit: str = Field(
        ...,
        description="Input semantic unit (always 'leaf' or 'single_leaf' for image models per §24.1).",
    )
    image_size: list[int] | None = Field(
        default=None, description="Expected image dimensions [height, width]."
    )
    image_normalization: dict[str, Any] | None = Field(
        default=None, description="Image normalization parameters (e.g. mean, std)."
    )
    channel_order: ChannelOrderType | None = Field(
        default=None,
        description="Expected image channel convention: RGB, BGR, or grayscale.",
    )

    # Extraction & Segmentation Config
    segmentation_config: dict[str, Any] | None = Field(
        default=None,
        description="Segmentation parameters (e.g. SAM2 threshold, mask rules).",
    )
    feature_config: dict[str, Any] | None = Field(
        default=None,
        description="Feature extractor hyperparameters (e.g. GLCM, HSV thresholds).",
    )

    # Hyperparameters & Inference Configurations
    hyperparameters: dict[str, Any] = Field(
        ..., description="Hyperparameters used for model training."
    )
    confidence_settings: dict[str, Any] = Field(
        ...,
        description="Threshold settings for uncertainty, abstention, or calibration.",
    )
    expected_input_schema: dict[str, Any] = Field(
        ..., description="Schema specification for valid inference inputs."
    )
    dependency_versions: dict[str, str] = Field(
        ...,
        description="Software/runtime dependency versions (python, sklearn, torch, etc.).",
    )

    # Metrics
    validation_metrics: dict[str, Any] = Field(
        ..., description="Evaluation metrics on validation partition per §18.2."
    )
    final_test_metrics: dict[str, Any] | None = Field(
        default=None,
        description="Final held-out test evaluation metrics when available.",
    )

    @field_validator("training_date")
    @classmethod
    def validate_iso_date(cls, v: str) -> str:
        """Validate training_date is a valid ISO 8601 datetime."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception as e:
            raise ValueError(
                f"training_date '{v}' is not a valid ISO 8601 datetime: {e}"
            ) from e
        return v

    @field_validator("classes")
    @classmethod
    def validate_classes_non_empty(cls, v: list[str]) -> list[str]:
        """Validate class list has no empty strings or duplicates."""
        if len(v) != len(set(v)):
            raise ValueError(f"Duplicate labels detected in classes list: {v}")
        if any(not s.strip() for s in v):
            raise ValueError("Class names cannot be empty or whitespace.")
        return v

    @model_validator(mode="after")
    def validate_modality_and_taxonomy(self) -> ModelMetadata:
        """Enforce §24.1 constraints across modality and taxonomy mappings."""
        # Check display mapping completeness
        missing_mappings = set(self.classes) - set(self.display_mapping.keys())
        if missing_mappings:
            raise ValueError(
                f"display_mapping is missing entries for classes: {sorted(missing_mappings)}"
            )

        # Overview §24.1 constraint: input unit is always 'leaf' or 'single_leaf' for image models
        if self.modality in ("image", "feature"):
            clean_unit = self.input_unit.strip().lower()
            if clean_unit not in ("leaf", "single_leaf"):
                raise ValueError(
                    f"input_unit for image/feature models must be 'leaf' or 'single_leaf' per §24.1 (got '{self.input_unit}')."
                )

        # Sensor and feature models must specify ordered features
        if self.modality in ("sensor", "feature") and (
            not self.ordered_features or len(self.ordered_features) == 0
        ):
            raise ValueError(
                f"ordered_features cannot be empty for modality '{self.modality}'."
            )

        # Image models must specify image_size and channel_order
        if self.modality == "image":
            if not self.image_size or len(self.image_size) != 2:
                raise ValueError(
                    "image_size [height, width] is required for image models."
                )
            if not self.channel_order:
                raise ValueError(
                    "channel_order is required for image models (RGB, BGR, or grayscale)."
                )

        # Split counts must contain required partitions
        for split_key in ("train", "validation", "test"):
            if split_key not in self.split_counts:
                raise ValueError(f"split_counts must contain '{split_key}' partition.")

        return self

    def to_json(self, indent: int = 2) -> str:
        """Serialize metadata model to formatted JSON string."""
        return self.model_dump_json(indent=indent)

    def save(self, filepath: Path | str) -> Path:
        """Save metadata model to JSON file."""
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def from_json(cls, json_str: str) -> ModelMetadata:
        """Load and validate metadata from JSON string."""
        return cls.model_validate_json(json_str)

    @classmethod
    def load(cls, filepath: Path | str) -> ModelMetadata:
        """Load and validate metadata from a JSON file."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
