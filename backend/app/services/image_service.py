"""Per-leaf ImageService pipeline for hydroponic plant pathology (T6-006).

Adheres strictly to TECH_RULES and Canonical Overview §24.1, §24.2, §18.2:
- Orchestrates: validated image -> leaf extraction (by input_mode: 'leaf' or 'plant') -> per-leaf model inference.
- Every valid leaf receives an individual classification, calibrated confidence, display label, and uncertainty flag.
- Rejected leaves are appended to `rejected_leaves[]` with failure reasons (never silently dropped).
- If no valid leaf is segmented, raises SegmentationFailedError (code: SEGMENTATION_FAILED, HTTP 422, retryable: True).
- The plant summary contains ONLY leaf counts and per-class histograms; it NEVER computes an average confidence.
- Attach complete preprocessing and hardware provenance.
"""

from __future__ import annotations

import io
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
from PIL import Image, ImageOps
import torch

from backend.app.core.config import get_settings
from backend.app.core.device_manager import DeviceManager
from backend.app.core.errors import (
    InvalidInputError,
    ModelUnavailableError,
    SegmentationFailedError,
)
from backend.app.schemas.prediction import (
    LeafPrediction,
    PredictionResponse,
    PredictionSummary,
    RejectedLeaf,
)
from backend.app.services.model_service import (
    ModelRegistryService,
    get_model_registry_service,
)
from ml.image_models.data import build_transforms
from ml.image_models.efficientnet import build_efficientnet_b0
from ml.labels import IMAGE_CLASSES, get_display_label
from ml.registry import ModelMetadata
from ml.segmentation.leaf_extraction import (
    ExtractedLeaf,
    LeafExtractionConfig,
    LeafExtractionResult,
    extract_leaves,
)
from ml.segmentation.sam2_segmenter import SAM2Segmenter

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_CROPS: tuple[str, ...] = ("lettuce", "spinach", "tomato")


class ImageService:
    """Orchestrator for multi-leaf instance segmentation and per-leaf disease prediction."""

    def __init__(
        self,
        model_registry_service: ModelRegistryService | None = None,
        segmenter: SAM2Segmenter | None = None,
    ) -> None:
        self.registry_service = model_registry_service or get_model_registry_service()
        self.device_manager = DeviceManager()
        self.segmenter = segmenter  # Lazy-loaded by extract_leaves if None
        self._cached_pytorch_models: dict[str, torch.nn.Module] = {}

    def _load_image_rgb(
        self,
        image_input: bytes | Path | str | np.ndarray | Image.Image,
    ) -> np.ndarray:
        """Decode and normalize orientation of input image into a uint8 RGB numpy array."""
        if isinstance(image_input, np.ndarray):
            if image_input.ndim != 3 or image_input.shape[2] != 3:
                raise InvalidInputError(
                    f"Expected 3-channel RGB image numpy array, got shape {image_input.shape}."
                )
            if np.issubdtype(image_input.dtype, np.floating):
                return np.clip(
                    image_input * 255.0 if np.nanmax(image_input) <= 1.0 + 1e-6 else image_input,
                    0,
                    255,
                ).astype(np.uint8)
            return image_input.astype(np.uint8)

        if isinstance(image_input, Image.Image):
            transposed = ImageOps.exif_transpose(image_input)
            return np.array(transposed.convert("RGB"), dtype=np.uint8)

        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.is_file():
                raise InvalidInputError(f"Image file not found at '{img_path}'.")
            try:
                with Image.open(img_path) as pil_img:
                    transposed = ImageOps.exif_transpose(pil_img)
                    return np.array(transposed.convert("RGB"), dtype=np.uint8)
            except Exception as exc:
                raise InvalidInputError(f"Failed to decode image from '{img_path}': {exc}") from exc

        if isinstance(image_input, (bytes, bytearray)):
            try:
                with Image.open(io.BytesIO(image_input)) as pil_img:
                    transposed = ImageOps.exif_transpose(pil_img)
                    return np.array(transposed.convert("RGB"), dtype=np.uint8)
            except Exception as exc:
                raise InvalidInputError(f"Failed to decode image from uploaded bytes: {exc}") from exc

        raise InvalidInputError(f"Unsupported image input type: {type(image_input).__name__}")

    def _get_pytorch_model(
        self,
        crop: str,
        artifact_path_rel: str,
        metadata: ModelMetadata,
    ) -> torch.nn.Module:
        """Instantiate and cache PyTorch evaluation model from active artifact."""
        cache_key = f"{crop}_{metadata.artifact_version}"
        if cache_key in self._cached_pytorch_models:
            return self._cached_pytorch_models[cache_key]

        target_device = torch.device(self.device_manager.get_target_device())
        num_classes = len(metadata.classes)

        model = build_efficientnet_b0(num_classes=num_classes)

        # Artifact path resolved through trusted root
        artifact_path = self.registry_service.root_dir / artifact_path_rel
        if not artifact_path.is_file():
            raise ModelUnavailableError(f"Model artifact not found at '{artifact_path}'.")

        try:
            ckpt = torch.load(artifact_path, map_location=target_device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            model.load_state_dict(state_dict)
            model.to(target_device)
            model.eval()
            self._cached_pytorch_models[cache_key] = model
            logger.info(
                "Loaded PyTorch EfficientNetB0 for %s on device %s", crop, target_device
            )
            return model
        except Exception as exc:
            raise ModelUnavailableError(
                f"Failed to load PyTorch weights for {crop}: {exc}"
            ) from exc

    def predict(
        self,
        image: bytes | Path | str | np.ndarray | Image.Image,
        crop: str,
        input_mode: Literal["leaf", "plant"] = "plant",
        model_family: str | None = None,
        provenance: Literal["live", "manual", "imported"] = "manual",
        mask_threshold: float = 0.50,
        max_leaves: int = 10,
    ) -> PredictionResponse:
        """Execute the per-leaf image classification pipeline.

        Parameters
        ----------
        image:
            Raw image bytes, Path, PIL Image, or RGB numpy array.
        crop:
            Target crop cultivar ('lettuce', 'spinach', 'tomato').
        input_mode:
            'leaf' (single leaf photo; primary mask extracted) or 'plant' (SAM2 multi-leaf extraction).
        model_family:
            Optional architecture override (defaults to promoted registry model).
        provenance:
            Input telemetry origin ('live', 'manual', 'imported').
        mask_threshold:
            SAM2 mask binarization threshold.
        max_leaves:
            Maximum number of leaf proposals to assess per plant.

        Returns
        -------
        PredictionResponse:
            Conforming to TECH_RULES with per-leaf predictions, discrete plant summary,
            and complete provenance.
        """
        clean_crop = crop.strip().lower()
        if clean_crop not in SUPPORTED_IMAGE_CROPS:
            raise InvalidInputError(
                f"Unsupported crop '{crop}'. Registered image crops: {list(SUPPORTED_IMAGE_CROPS)}."
            )

        # 1. Image Decode and Normalization
        img_rgb = self._load_image_rgb(image)
        h, w, _ = img_rgb.shape

        # 2. Retrieve Active Model and Metadata
        try:
            _, metadata = self.registry_service.get_model(clean_crop, modality="image")
        except Exception as exc:
            raise ModelUnavailableError(
                f"Promoted image model for '{clean_crop}' is unavailable: {exc}"
            ) from exc

        # 3. Perform Leaf Extraction & Mask Quality Gating
        is_single_leaf = (input_mode == "leaf")
        extract_cfg = LeafExtractionConfig(
            single_leaf_mode=is_single_leaf,
            max_leaves=max_leaves,
            merge_duplicates=True,
            validation_config=extract_leaves.__globals__["MaskValidationConfig"](
                min_area_ratio=0.005 if is_single_leaf else 0.003,
                max_area_ratio=0.95 if is_single_leaf else 0.85,
            ),
        )

        try:
            extraction_result: LeafExtractionResult = extract_leaves(
                img_rgb,
                segmenter=self.segmenter,
                config=extract_cfg,
            )
        except Exception as exc:
            logger.error("Segmentation routine raised exception for %s: %s", clean_crop, exc)
            raise SegmentationFailedError(
                message=f"Leaf segmentation failed on {clean_crop} image: {exc}",
                details={"crop": clean_crop, "input_mode": input_mode},
            ) from exc

        # Acceptance Criterion 2: Rejected leaves are listed with reasons, not dropped
        rejected_leaves_list: list[RejectedLeaf] = []
        for rej in extraction_result.rejected:
            rejected_leaves_list.append(
                RejectedLeaf(
                    leaf_id=rej.candidate_id,
                    reason=rej.reason,
                    verdict="reject",
                )
            )

        # Invariant: If no valid leaf exists return SEGMENTATION_FAILED (retryable), never a prediction.
        if len(extraction_result.leaves) == 0:
            rejection_summary = "; ".join([f"{r.leaf_id}: {r.reason}" for r in rejected_leaves_list])
            logger.warning(
                "Zero valid leaves extracted for %s (rejected: %d). Raising SEGMENTATION_FAILED.",
                clean_crop,
                len(rejected_leaves_list),
            )
            raise SegmentationFailedError(
                message=(
                    "Segmentation failed to extract any valid leaf from the photograph. "
                    "Please ensure the plant leaves are clearly in focus and well-illuminated."
                ),
                details={
                    "crop": clean_crop,
                    "input_mode": input_mode,
                    "rejected_count": len(rejected_leaves_list),
                    "reasons": rejection_summary,
                },
            )

        # 4. Model Inference per Valid Leaf
        model = self._get_pytorch_model(
            crop=clean_crop,
            artifact_path_rel=metadata.artifact_file,
            metadata=metadata,
        )

        val_transform = build_transforms(split="val", image_size=(224, 224))
        target_device = torch.device(self.device_manager.get_target_device())

        conf_settings = metadata.confidence_settings or {}
        uncertainty_thresh = conf_settings.get("uncertainty_thresholds", {})
        moderate_thresh = float(
            uncertainty_thresh.get("moderate", conf_settings.get("threshold_moderate", 0.50))
        )
        high_thresh = float(
            uncertainty_thresh.get("high", conf_settings.get("threshold_high", 0.75))
        )
        fitted_temp = float(
            conf_settings.get("temperature_scaling", {}).get("fitted_temperature", 1.0)
        )
        score_semantics = str(
            conf_settings.get(
                "score_semantics",
                "Temperature-scaled softmax confidence representing validation alignment (heuristic score).",
            )
        )

        leaves_predictions: list[LeafPrediction] = []

        for idx, leaf in enumerate(extraction_result.leaves):
            leaf_id = f"leaf_{idx + 1:03d}"

            # Prepare RGB crop for model input
            # Using leaf.masked_crop (RGB foreground on black background)
            pil_crop = Image.fromarray(leaf.masked_crop)
            tensor_input = val_transform(pil_crop).unsqueeze(0).to(target_device)

            with torch.no_grad():
                logits = model(tensor_input)
                # Temperature scaling calibration
                scaled_logits = logits / fitted_temp if fitted_temp > 0 else logits
                probs = torch.softmax(scaled_logits, dim=-1).squeeze(0).cpu().numpy()

            pred_idx = int(np.argmax(probs))
            pred_class = metadata.classes[pred_idx]
            display_label = metadata.display_mapping.get(
                pred_class, get_display_label(clean_crop, pred_class, "image")
            )
            raw_confidence = float(probs[pred_idx])
            confidence = round(raw_confidence, 4)

            # Calibrated uncertainty check
            is_uncertain = bool(confidence < moderate_thresh)

            # Compute normalized bounding box in [0.0, 1.0] for exact responsive canvas alignment
            x1, y1, x2, y2 = leaf.bounding_box
            norm_bbox = [
                round(float(x1) / float(w), 4),
                round(float(y1) / float(h), 4),
                round(float(x2) / float(w), 4),
                round(float(y2) / float(h), 4),
            ]
            seg_area = int(np.count_nonzero(leaf.mask)) if hasattr(leaf, "mask") and leaf.mask is not None else None

            leaves_predictions.append(
                LeafPrediction(
                    leaf_id=leaf_id,
                    bbox=norm_bbox,
                    contour_area=seg_area,
                    mask_ref=None,
                    predicted_class=pred_class,
                    display_label=display_label,
                    confidence=confidence,
                    confidence_semantics=score_semantics,
                    uncertain=is_uncertain,
                )
            )

        # 5. Plant-Level Discrete Summary
        # Acceptance Criterion 3: No averaged or merged confidence appears in the response
        class_counts = dict(Counter(lp.predicted_class for lp in leaves_predictions))
        plant_summary = PredictionSummary(
            leaf_count=len(leaves_predictions),
            class_counts=class_counts,
        )

        # Determine dominant pathology across leaves
        # Hydroponic foliar diseases are localized (Rule 16 / §19). If any leaf exhibits a deficiency
        # or pathology, that pathology takes priority over an asymptomatic or edge leaf.
        def _is_healthy_str(c: str) -> bool:
            cl = c.lower()
            return "healthy" in cl or "healthier" in cl

        non_healthy_leaves = [
            lp for lp in leaves_predictions if not _is_healthy_str(lp.predicted_class)
        ]
        if non_healthy_leaves:
            pathology_counts = Counter(lp.predicted_class for lp in non_healthy_leaves)
            max_cnt = max(pathology_counts.values())
            candidate_classes = [cls for cls, cnt in pathology_counts.items() if cnt == max_cnt]
            if len(candidate_classes) == 1:
                dominant_class = candidate_classes[0]
            else:
                dominant_class = max(
                    candidate_classes,
                    key=lambda cls: max(
                        (lp.confidence or 0.0)
                        for lp in non_healthy_leaves
                        if lp.predicted_class == cls
                    ),
                )
        else:
            dominant_class = Counter(lp.predicted_class for lp in leaves_predictions).most_common(1)[0][0]

        dominant_display = metadata.display_mapping.get(
            dominant_class, get_display_label(clean_crop, dominant_class, "image")
        )

        # In multi-leaf mode, top-level confidence is None (never averaged!)
        # In single-leaf mode, top-level reflects the solitary leaf
        if is_single_leaf:
            top_confidence = leaves_predictions[0].confidence
            top_semantics = leaves_predictions[0].confidence_semantics
            top_uncertain = leaves_predictions[0].uncertain
            top_class = leaves_predictions[0].predicted_class
            top_display = leaves_predictions[0].display_label
        else:
            top_confidence = None
            top_semantics = None
            top_uncertain = any(lp.uncertain for lp in leaves_predictions)
            top_class = dominant_class
            top_display = dominant_display

        # 6. Operational Warnings & Disclosed Scientific Limitations
        warnings: list[str] = []
        if rejected_leaves_list:
            warnings.append(
                f"{len(rejected_leaves_list)} candidate mask(s) were rejected during per-leaf quality validation."
            )
        if any(lp.uncertain for lp in leaves_predictions):
            warnings.append(
                "One or more leaves exhibited low model confidence below validation distribution thresholds."
            )

        limitations: list[str] = [
            (
                "Inherited Whole-Plant Weak Labels: Leaf training crops inherit canopy-level "
                "stress labels; asymptomatic leaves on stressed plants may carry stress labels."
            ),
            (
                "Calibrated Softmax Heuristic: Leaf confidence scores reflect post-hoc temperature-scaled "
                "heuristic scores and must not be interpreted as objective Bayesian probabilities."
            ),
            (
                "Greenhouse Hydroponic Boundary: Models validated exclusively on controlled NFT/DWC "
                "hydroponic systems; field-grown or soil-cultivated specimens are unvalidated."
            ),
        ]

        # 7. Construct Comprehensive Response Envelope
        response = PredictionResponse(
            crop=clean_crop,
            modality="image",
            predicted_class=top_class,
            display_label=top_display,
            confidence=top_confidence,
            confidence_semantics=top_semantics,
            uncertain=top_uncertain,
            experimental=metadata.experimental,
            model_id=metadata.model_id,
            model_version=metadata.artifact_version,
            timestamp=datetime.now(timezone.utc),
            provenance=provenance,
            warnings=warnings,
            limitations=limitations,
            input_mode="single_leaf" if is_single_leaf else "plant",
            leaves=leaves_predictions,
            rejected_leaves=rejected_leaves_list,
            summary=plant_summary,
        )

        logger.info(
            "ImageService prediction complete for %s (mode=%s): %d leaves assessed, %d rejected.",
            clean_crop,
            input_mode,
            len(leaves_predictions),
            len(rejected_leaves_list),
        )
        return response


# Module-level singleton
_IMAGE_SERVICE: ImageService | None = None


def get_image_service() -> ImageService:
    """Return the global ImageService instance."""
    global _IMAGE_SERVICE
    if _IMAGE_SERVICE is None:
        _IMAGE_SERVICE = ImageService()
    return _IMAGE_SERVICE


def reset_image_service() -> None:
    """Reset global ImageService singleton (for testing)."""
    global _IMAGE_SERVICE
    _IMAGE_SERVICE = None

