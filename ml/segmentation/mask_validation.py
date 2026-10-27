"""Per-leaf mask validation quality gate for foundation segmentation (T5-009).

Evaluates candidate binary masks from SAM 2.1 or downstream segmenters against
geometric and topological constraints before feature extraction:
1. Emptiness & area bounds: Rejects empty masks, tiny dust/droplet artifacts, and whole-frame floods.
2. Boundary / border contact: Distinguishes cleanly framed leaves from severely truncated edges.
3. Connected components: Enforces single-region coherence vs multi-leaf fragmented conglomerates.
4. Typed verdicts: Returns OK, RETRY, or REJECT with structured metrics and explicit reason strings.

Threshold origins are documented in MaskValidationConfig in accordance with PRD ADR-0003
and will be fine-tuned during the 60-image qualitative mask review (T5-011).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import cv2
import numpy as np


class MaskVerdict(str, Enum):
    """Verdict classification for a candidate leaf segmentation mask."""

    OK = "ok"  # Passes all geometric and topological checks
    RETRY = "retry"  # Marginal defect (e.g. minor border touch or weak fragmentation) that could resolve with re-prompting
    REJECT = "reject"  # Fatal defect (empty, tiny artifact, whole-frame canopy flood, or severe truncation)


@dataclass(frozen=True)
class MaskValidationConfig:
    """Configurable geometric bounds with documented origins.

    Origins:
    - min_area_ratio (0.01): A hydroponic lettuce/spinach/tomato leaf occupies at least
      1% of the sensor frame (e.g. >2,621 px in 512x512). Sub-1% masks correspond to
      algae specks, peat moss flecks, or water droplets.
    - max_area_ratio (0.85): Single-leaf isolation constraint. A single leaf cannot
      fill >85% of the frame without either extreme optical fish-eye distortion or
      the segmenter erroneously grouping the whole canopy / background bench.
    - max_border_contact_ratio (0.08): Fraction of mask perimeter pixels lying directly
      on the 1-pixel frame perimeter. Contacts >8% signify severe edge clipping.
    - retry_border_contact_ratio (0.03): Contacts between 3% and 8% trigger RETRY so
      the pipeline can re-prompt or shift camera crop.
    - min_component_ratio (0.85): Fraction of mask area contained within the single
      largest connected component. Must be >=85% for OK to guarantee single-leaf topology.
    - retry_component_ratio (0.60): Largest component between 60% and 85% triggers RETRY
      (e.g. two overlapping leaves where a negative prompt could disambiguate), whereas
      <60% triggers fatal REJECT.
    """

    min_area_ratio: float = 0.01
    max_area_ratio: float = 0.85
    max_border_contact_ratio: float = 0.08
    retry_border_contact_ratio: float = 0.03
    min_component_ratio: float = 0.85
    retry_component_ratio: float = 0.60


@dataclass(frozen=True)
class MaskMetrics:
    """Geometric and topological measurements calculated on a binary mask."""

    height: int
    width: int
    mask_pixels: int
    total_pixels: int
    area_ratio: float
    border_pixels: int
    border_contact_ratio: float
    num_components: int
    largest_component_pixels: int
    largest_component_ratio: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class MaskValidationResult:
    """Typed validation verdict for a candidate leaf mask."""

    verdict: MaskVerdict
    is_valid: bool
    reason: str
    metrics: MaskMetrics


def compute_mask_metrics(mask: np.ndarray) -> MaskMetrics:
    """Compute geometric and topological properties of a candidate 2D binary mask.

    Args:
        mask: 2D numpy array of booleans or integers (0 and non-zero).

    Returns:
        MaskMetrics dataclass with area, perimeter contact, and connected components.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask array (H, W), got shape {mask.shape}")

    h, w = mask.shape
    total_pixels = h * w
    binary_mask = (mask > 0).astype(np.uint8)
    mask_pixels = int(np.sum(binary_mask))

    if mask_pixels == 0:
        return MaskMetrics(
            height=h,
            width=w,
            mask_pixels=0,
            total_pixels=total_pixels,
            area_ratio=0.0,
            border_pixels=0,
            border_contact_ratio=0.0,
            num_components=0,
            largest_component_pixels=0,
            largest_component_ratio=0.0,
        )

    area_ratio = mask_pixels / float(total_pixels)

    # Perimeter / border contact: count how many mask pixels touch the outer 1-pixel frame border
    border_pixels = int(
        np.sum(binary_mask[0, :])
        + np.sum(binary_mask[-1, :])
        + np.sum(binary_mask[1:-1, 0])
        + np.sum(binary_mask[1:-1, -1])
    )
    border_contact_ratio = border_pixels / float(mask_pixels)

    # Connected components analysis (8-connectivity)
    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    # Background label is 0; foreground components are labels 1 to num_labels-1
    if num_labels > 1:
        foreground_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_component_pixels = int(np.max(foreground_areas))
        largest_component_ratio = largest_component_pixels / float(mask_pixels)
        num_components = int(num_labels - 1)
    else:
        largest_component_pixels = 0
        largest_component_ratio = 0.0
        num_components = 0

    return MaskMetrics(
        height=h,
        width=w,
        mask_pixels=mask_pixels,
        total_pixels=total_pixels,
        area_ratio=area_ratio,
        border_pixels=border_pixels,
        border_contact_ratio=border_contact_ratio,
        num_components=num_components,
        largest_component_pixels=largest_component_pixels,
        largest_component_ratio=largest_component_ratio,
    )


def validate_mask(
    mask: np.ndarray,
    config: MaskValidationConfig | None = None,
) -> MaskValidationResult:
    """Validate a single candidate leaf mask against geometric quality constraints.

    Args:
        mask: 2D numpy array (boolean or integer).
        config: MaskValidationConfig thresholds (defaults to standard config).

    Returns:
        MaskValidationResult with verdict (OK/RETRY/REJECT), boolean validity,
        reason string, and numerical metrics.
    """
    if config is None:
        config = MaskValidationConfig()

    metrics = compute_mask_metrics(mask)

    # 1. Emptiness check
    if metrics.mask_pixels == 0:
        return MaskValidationResult(
            verdict=MaskVerdict.REJECT,
            is_valid=False,
            reason="Mask is empty (0 foreground pixels)",
            metrics=metrics,
        )

    # 2. Tiny artifact check
    if metrics.area_ratio < config.min_area_ratio:
        return MaskValidationResult(
            verdict=MaskVerdict.REJECT,
            is_valid=False,
            reason=(
                f"Mask is too small ({metrics.area_ratio:.4f} < "
                f"min_area_ratio {config.min_area_ratio:.4f}); likely dust or noise"
            ),
            metrics=metrics,
        )

    # 3. Whole-frame flood check
    if metrics.area_ratio > config.max_area_ratio:
        return MaskValidationResult(
            verdict=MaskVerdict.REJECT,
            is_valid=False,
            reason=(
                f"Mask is too large ({metrics.area_ratio:.4f} > "
                f"max_area_ratio {config.max_area_ratio:.4f}); likely background or canopy flood"
            ),
            metrics=metrics,
        )

    # 4. Severe border clipping check
    if metrics.border_contact_ratio > config.max_border_contact_ratio:
        return MaskValidationResult(
            verdict=MaskVerdict.REJECT,
            is_valid=False,
            reason=(
                f"Mask has severe border contact ({metrics.border_contact_ratio:.4f} > "
                f"max_border_contact_ratio {config.max_border_contact_ratio:.4f}); leaf heavily truncated"
            ),
            metrics=metrics,
        )

    # 5. Severe fragmentation check
    if metrics.largest_component_ratio < config.retry_component_ratio:
        return MaskValidationResult(
            verdict=MaskVerdict.REJECT,
            is_valid=False,
            reason=(
                f"Mask is severely fragmented into {metrics.num_components} components "
                f"(largest component ratio {metrics.largest_component_ratio:.4f} < "
                f"{config.retry_component_ratio:.4f})"
            ),
            metrics=metrics,
        )

    # 6. Retry conditions: moderate border touch or moderate fragmentation
    retry_reasons = []
    if metrics.border_contact_ratio > config.retry_border_contact_ratio:
        retry_reasons.append(
            f"border contact ({metrics.border_contact_ratio:.4f} > {config.retry_border_contact_ratio:.4f})"
        )

    if metrics.largest_component_ratio < config.min_component_ratio:
        retry_reasons.append(
            f"fragmented into {metrics.num_components} components (largest {metrics.largest_component_ratio:.4f} < {config.min_component_ratio:.4f})"
        )

    if retry_reasons:
        return MaskValidationResult(
            verdict=MaskVerdict.RETRY,
            is_valid=False,
            reason=f"Candidate requires retry due to: {'; '.join(retry_reasons)}",
            metrics=metrics,
        )

    # 7. Valid single-leaf mask
    return MaskValidationResult(
        verdict=MaskVerdict.OK,
        is_valid=True,
        reason=(
            f"Valid leaf mask ({metrics.mask_pixels} px, area ratio {metrics.area_ratio:.4f}, "
            f"single dominant component {metrics.largest_component_ratio:.4f})"
        ),
        metrics=metrics,
    )


def validate_masks(
    masks: list[np.ndarray],
    config: MaskValidationConfig | None = None,
) -> list[MaskValidationResult]:
    """Validate a collection of candidate leaf masks.

    Args:
        masks: List of 2D binary numpy arrays.
        config: Optional validation configuration.

    Returns:
        List of MaskValidationResult objects corresponding to each input mask.
    """
    if config is None:
        config = MaskValidationConfig()
    return [validate_mask(m, config=config) for m in masks]
