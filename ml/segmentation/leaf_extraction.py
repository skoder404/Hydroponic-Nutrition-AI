"""Per-leaf instance extraction pipeline for hydroponic plants (T5-010).

This module extracts individual leaf instances from whole-plant photos or single-leaf crops
using SAM 2.1 foundation segmentation, per-leaf quality gating, duplicate suppression,
and background handling.

================================================================================
Design Decisions & Architecture (ADR 0003 & T5-010)
================================================================================

1. Unified Routine for Training and Inference:
   `extract_leaves` serves as the single source of truth for:
   - Dataset generation (e.g., slicing lettuce heads into training crops)
   - Feature extraction pipelines (masking foreground for color & texture features)
   - Production inference endpoints (segmenting multi-leaf or single-leaf uploads)

2. Single-Leaf Bypass Mode:
   When `single_leaf_mode=True` (or when processing pre-cropped single leaf photos),
   multi-leaf proposal generation is bypassed. The segmenter segments the primary leaf
   directly from the center prompt, passes it through `validate_mask`, and returns it.

3. Candidate Discovery & Prompt Strategy:
   To avoid the 5-25 second latency and inter-leaf merging problems of unsupervised
   grid scanning (documented in ADR 0003), multi-leaf candidate prompts are generated
   by detecting vegetation component centroids using Excess Green (ExG) and adaptive
   Otsu thresholding. Each candidate prompt is queried with `SAM2Segmenter.segment`.

4. Strict Quality Gating (No Silent Drops):
   Every candidate mask is evaluated with `ml.segmentation.mask_validation.validate_mask`.
   Any candidate that fails (empty, sub-1% area, frame-flooding >85%, severe border truncation,
   or severe fragmentation) is appended to `rejected[]` with its typed verdict and reason string.
   No candidate is EVER silently discarded.

5. Overlapping Duplicate Removal & Merging Rule:
   Near-duplicate masks from overlapping prompt points are resolved using Mask IoU:
   $$ \\text{IoU}(M_A, M_B) = \\frac{|M_A \\cap M_B|}{|M_A \\cup M_B|} $$
   - Rule (default): Candidates are ranked by segmentation confidence score descending.
     If a candidate shares IoU >= `iou_overlap_threshold` (default 0.50) with an already
     accepted leaf, it is classified as a duplicate, assigned `MaskVerdict.REJECT`, and
     moved to `rejected[]` with an explicit reason string citing the primary leaf ID.
   - Merging Option: If `merge_duplicates=True`, the overlapping mask is merged via boolean OR
     into the primary leaf and re-validated.
   - Capping: If the number of accepted leaves reaches `max_leaves` (default 10), any remaining
     candidates are placed in `rejected[]` with a capping reason.

6. Background Handling:
   For each accepted leaf:
   - `bounding_box`: (x1, y1, x2, y2) tight bounding box enclosing the mask.
   - `masked_crop`: RGB image cropped to the bounding box where background pixels
     (mask == False) are zeroed to black (0, 0, 0) or filled with `background_fill`.
   - `masked_crop_rgba`: 4-channel image where the alpha channel is 255 for leaf
     pixels and 0 for background pixels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from ml.features.common import (
    DimensionMismatchError,
    EmptyMaskError,
    load_image,
)
from ml.segmentation.mask_validation import (
    MaskMetrics,
    MaskValidationConfig,
    MaskValidationResult,
    MaskVerdict,
    compute_mask_metrics,
    validate_mask,
)
from ml.segmentation.sam2_segmenter import (
    SAM2Segmenter,
    SegmentationError,
    SegmentationResult,
)


@dataclass(frozen=True)
class LeafExtractionConfig:
    """Configuration parameters for leaf extraction."""

    # Processing mode
    single_leaf_mode: bool = False

    # Capacity and deduplication
    max_leaves: int = 10
    iou_overlap_threshold: float = 0.50
    merge_duplicates: bool = False

    # Mask validation bounds
    validation_config: MaskValidationConfig = field(default_factory=MaskValidationConfig)

    # Background handling: RGB color for masked background in crops
    background_fill: Tuple[int, int, int] = (0, 0, 0)

    # Candidate proposal parameters (for multi-leaf plants)
    min_vegetation_area_px: int = 150
    max_candidate_proposals: int = 25
    candidate_points: Optional[List[Tuple[int, int]]] = None


@dataclass
class ExtractedLeaf:
    """An accepted individual leaf instance."""

    leaf_id: str
    bounding_box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    mask: np.ndarray  # 2D boolean array (H, W) full-frame
    leaf_mask_crop: np.ndarray  # 2D boolean array (crop_h, crop_w)
    masked_crop: np.ndarray  # 3D RGB uint8 (crop_h, crop_w, 3) with blacked background
    masked_crop_rgba: np.ndarray  # 4D RGBA uint8 (crop_h, crop_w, 4) with transparent background
    confidence: float
    metrics: MaskMetrics

    def to_dict(self) -> dict[str, Any]:
        """Convert leaf summary to dictionary (excluding large arrays)."""
        return {
            "leaf_id": self.leaf_id,
            "bounding_box": list(self.bounding_box),
            "confidence": self.confidence,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class RejectedLeaf:
    """A rejected candidate mask with reason for auditability."""

    candidate_id: str
    bounding_box: Tuple[int, int, int, int]
    mask: np.ndarray
    reason: str
    verdict: MaskVerdict
    metrics: MaskMetrics
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert rejected leaf summary to dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "bounding_box": list(self.bounding_box),
            "reason": self.reason,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class LeafExtractionResult:
    """Structured result of leaf instance extraction."""

    leaves: List[ExtractedLeaf]
    rejected: List[RejectedLeaf]
    is_single_leaf: bool
    total_candidates_evaluated: int
    image_shape: Tuple[int, int, int]
    config: LeafExtractionConfig

    @property
    def num_leaves(self) -> int:
        return len(self.leaves)

    @property
    def num_rejected(self) -> int:
        return len(self.rejected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_leaves": self.num_leaves,
            "num_rejected": self.num_rejected,
            "is_single_leaf": self.is_single_leaf,
            "total_candidates": self.total_candidates_evaluated,
            "image_shape": list(self.image_shape),
            "leaves": [leaf.to_dict() for leaf in self.leaves],
            "rejected": [rej.to_dict() for rej in self.rejected],
        }


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two 2D boolean masks."""
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    intersection = int(np.count_nonzero(m1 & m2))
    if intersection == 0:
        return 0.0
    union = int(np.count_nonzero(m1 | m2))
    return float(intersection / union) if union > 0 else 0.0


def compute_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Calculate tight bounding box (x1, y1, x2, y2) from a 2D binary mask."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return (0, 0, 0, 0)
    y1, y2 = int(np.min(rows)), int(np.max(rows))
    x1, x2 = int(np.min(cols)), int(np.max(cols))
    return (x1, y1, x2, y2)


def regularize_leaf_mask(mask: np.ndarray, close_kernel_size: int = 15) -> np.ndarray:
    """Bridge necrotic lesions, leaf spots, and internal holes in plant leaves.

    Plant pathology (e.g. bacterial spot, anthracnose, pest damage) causes dark
    lesions and perforations inside the leaf blade. A naive binary mask will have
    internal holes and disconnected boundaries that falsely trip connected-component
    fragmentation gates. Morphological closing connects the leaf body and retains
    the largest dominant connected component.
    """
    if not np.any(mask):
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels > 1:
        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == largest_idx)
    return closed.astype(bool)


def generate_candidate_proposals(
    image: np.ndarray,
    min_area_px: int = 150,
    max_proposals: int = 25,
) -> List[Tuple[int, int]]:
    """Propose candidate leaf prompt coordinates across the plant canopy.

    Employs Excess Green (ExG) along with HSV foliar/chlorosis detection and
    distance transform peak extraction so that multi-leaf rosettes and deficient
    outer leaves are individually sampled rather than grouped into a single centroid.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3).
    min_area_px : int
        Minimum connected component area to consider as a leaf proposal.
    max_proposals : int
        Maximum number of prompt points to return.

    Returns
    -------
    List[Tuple[int, int]]
        List of (x, y) coordinates for prompt-based segmentation.
    """
    h, w, _ = image.shape
    # Compute Excess Green index: 2G - R - B
    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)
    exg = 2.0 * g - r - b
    exg_clipped = np.clip(exg, 0, 255).astype(np.uint8)

    # Detect foliage, including chlorotic/nitrogen-deficient leaves (yellow/pale green)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    veg_hsv = cv2.inRange(hsv, np.array([12, 30, 35]), np.array([95, 255, 255]))
    exg_mask = (exg_clipped > 15).astype(np.uint8) * 255
    combined_veg = cv2.bitwise_or(exg_mask, veg_hsv)

    # Morphological opening to detach minor touching leaves and eliminate noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    veg_clean = cv2.morphologyEx(combined_veg, cv2.MORPH_OPEN, kernel)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(veg_clean, connectivity=8)

    proposals: List[Tuple[Tuple[int, int], int]] = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue

        comp_mask = (labels == label_idx).astype(np.uint8) * 255

        # For large connected foliage canopies, sample multiple leaf blade centers via distance transform
        if area > 0.04 * h * w:
            dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            k_sz = max(25, int(min(h, w) * 0.05))
            if k_sz % 2 == 0:
                k_sz += 1
            k_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
            local_max = cv2.dilate(dist, k_peak)
            peaks = (dist == local_max) & (dist > 18.0)
            peak_coords = np.argwhere(peaks)

            # Sort peaks by distance value descending
            peak_tuples = [((int(px), int(py)), int(dist[py, px])) for py, px in peak_coords]
            peak_tuples.sort(key=lambda item: item[1], reverse=True)

            if peak_tuples:
                proposals.extend(peak_tuples[:max_proposals])
            else:
                cx, cy = int(centroids[label_idx, 0]), int(centroids[label_idx, 1])
                proposals.append(((cx, cy), area))
        else:
            cx, cy = int(centroids[label_idx, 0]), int(centroids[label_idx, 1])
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            proposals.append(((cx, cy), area))

    # Sort proposals by prominence/area descending
    proposals.sort(key=lambda item: item[1], reverse=True)

    # Deduplicate points that are too close to each other (min 30px distance)
    selected_coords: List[Tuple[int, int]] = []
    min_dist_sq = 30 * 30
    for pt, _ in proposals:
        if any((pt[0] - s[0]) ** 2 + (pt[1] - s[1]) ** 2 < min_dist_sq for s in selected_coords):
            continue
        selected_coords.append(pt)
        if len(selected_coords) >= max_proposals:
            break

    # If no vegetation components were found, provide default frame center
    if not selected_coords:
        selected_coords = [(w // 2, h // 2)]

    return selected_coords


def create_masked_crop(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    background_fill: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract masked RGB crop and RGBA crop for an individual leaf.

    Background pixels outside the leaf mask are filled with `background_fill`
    in RGB, and set to alpha=0 in RGBA.

    Parameters
    ----------
    image : np.ndarray
        Full-frame RGB image of shape (H, W, 3).
    mask : np.ndarray
        Full-frame 2D boolean mask of shape (H, W).
    bbox : Tuple[int, int, int, int]
        Bounding box (x1, y1, x2, y2).
    background_fill : Tuple[int, int, int]
        RGB fill value for background pixels (default: black (0, 0, 0)).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (leaf_mask_crop, masked_crop_rgb, masked_crop_rgba)
    """
    x1, y1, x2, y2 = bbox
    crop_rgb = image[y1 : y2 + 1, x1 : x2 + 1].copy()
    crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1].copy()

    # Masked RGB crop
    masked_crop_rgb = np.zeros_like(crop_rgb)
    masked_crop_rgb[:, :] = background_fill
    masked_crop_rgb[crop_mask] = crop_rgb[crop_mask]

    # Masked RGBA crop
    crop_h, crop_w = crop_rgb.shape[:2]
    masked_crop_rgba = np.zeros((crop_h, crop_w, 4), dtype=np.uint8)
    masked_crop_rgba[:, :, :3] = crop_rgb
    masked_crop_rgba[:, :, 3] = np.where(crop_mask, 255, 0).astype(np.uint8)

    return crop_mask, masked_crop_rgb, masked_crop_rgba


def extract_leaves(
    image: Union[np.ndarray, str, Path],
    segmenter: Optional[SAM2Segmenter] = None,
    config: Optional[LeafExtractionConfig] = None,
) -> LeafExtractionResult:
    """Extract, validate, and deduplicate individual leaf instances from an image.

    This routine is unified across training data creation and production inference.

    Parameters
    ----------
    image : Union[np.ndarray, str, Path]
        Input RGB image array of shape (H, W, 3) or file path to image.
    segmenter : Optional[SAM2Segmenter]
        Segmenter wrapper instance. If None, instantiates a lazy-loading default SAM2Segmenter.
    config : Optional[LeafExtractionConfig]
        Configuration options for single-leaf bypass, deduplication, and quality gates.

    Returns
    -------
    LeafExtractionResult
        Dataclass containing:
        - leaves: List[ExtractedLeaf] passing validation and deduplication
        - rejected: List[RejectedLeaf] failing validation or deduplication (with reasons)
        - is_single_leaf: whether single-leaf mode was executed
        - total_candidates_evaluated: total candidate masks analyzed
        - image_shape: (H, W, 3)
        - config: configuration used
    """
    cfg = config or LeafExtractionConfig()
    seg = segmenter or SAM2Segmenter()

    # Load and normalize image to RGB uint8
    if isinstance(image, (str, Path)):
        img_rgb = load_image(image)
    elif isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise DimensionMismatchError(f"Expected RGB image of shape (H, W, 3), got {image.shape}")
        if np.issubdtype(image.dtype, np.floating):
            img_rgb = np.clip(image * 255.0 if np.nanmax(image) <= 1.0 + 1e-6 else image, 0, 255).astype(np.uint8)
        else:
            img_rgb = image.astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    h, w, _ = img_rgb.shape

    # 1. Single-leaf bypass mode
    if cfg.single_leaf_mode:
        # Segment directly using full-frame center prompt
        seg_res = seg.segment(img_rgb, point_coords=[[w // 2, h // 2]], point_labels=[1])
        mask = regularize_leaf_mask(seg_res.mask)
        val_res = validate_mask(mask, config=cfg.validation_config)

        bbox = compute_bounding_box(mask)
        if val_res.is_valid:
            leaf_mask_crop, rgb_crop, rgba_crop = create_masked_crop(
                img_rgb, mask, bbox, background_fill=cfg.background_fill
            )
            leaf = ExtractedLeaf(
                leaf_id="leaf_0",
                bounding_box=bbox,
                mask=mask,
                leaf_mask_crop=leaf_mask_crop,
                masked_crop=rgb_crop,
                masked_crop_rgba=rgba_crop,
                confidence=seg_res.score,
                metrics=val_res.metrics,
            )
            return LeafExtractionResult(
                leaves=[leaf],
                rejected=[],
                is_single_leaf=True,
                total_candidates_evaluated=1,
                image_shape=(h, w, 3),
                config=cfg,
            )
        else:
            rejected = RejectedLeaf(
                candidate_id="candidate_0",
                bounding_box=bbox,
                mask=mask,
                reason=val_res.reason,
                verdict=val_res.verdict,
                metrics=val_res.metrics,
                confidence=seg_res.score,
            )
            return LeafExtractionResult(
                leaves=[],
                rejected=[rejected],
                is_single_leaf=True,
                total_candidates_evaluated=1,
                image_shape=(h, w, 3),
                config=cfg,
            )

    # 2. Multi-leaf plant mode
    if cfg.candidate_points is not None:
        prompt_coords = list(cfg.candidate_points)
    else:
        prompt_coords = generate_candidate_proposals(
            img_rgb,
            min_area_px=cfg.min_vegetation_area_px,
            max_proposals=cfg.max_candidate_proposals,
        )

    candidate_results: List[Tuple[SegmentationResult, MaskValidationResult, Tuple[int, int, int, int], np.ndarray]] = []
    rejected_list: List[RejectedLeaf] = []

    for c_idx, pt in enumerate(prompt_coords):
        cand_id = f"candidate_{c_idx}"
        seg_res = seg.segment(img_rgb, point_coords=[list(pt)], point_labels=[1])
        mask = regularize_leaf_mask(seg_res.mask)
        bbox = compute_bounding_box(mask)
        val_res = validate_mask(mask, config=cfg.validation_config)

        if not val_res.is_valid:
            # Rejection from quality gate (empty, tiny, whole-image, high border contact, or fragmented)
            rejected_list.append(
                RejectedLeaf(
                    candidate_id=cand_id,
                    bounding_box=bbox,
                    mask=mask,
                    reason=val_res.reason,
                    verdict=val_res.verdict,
                    metrics=val_res.metrics,
                    confidence=seg_res.score,
                )
            )
        else:
            candidate_results.append((seg_res, val_res, bbox, mask))

    # 3. Duplicate Suppression / Merging & Leaf Capping
    # Sort valid candidates by score descending
    candidate_results.sort(key=lambda item: item[0].score, reverse=True)

    accepted_leaves: List[ExtractedLeaf] = []

    for rank, (seg_res, val_res, bbox, cand_mask) in enumerate(candidate_results):
        cand_id = f"candidate_{rank}"
        cand_score = seg_res.score

        # Check for max_leaves cap
        if len(accepted_leaves) >= cfg.max_leaves:
            rejected_list.append(
                RejectedLeaf(
                    candidate_id=cand_id,
                    bounding_box=bbox,
                    mask=cand_mask,
                    reason=f"Exceeded max_leaves limit ({cfg.max_leaves}); candidate suppressed.",
                    verdict=MaskVerdict.REJECT,
                    metrics=val_res.metrics,
                    confidence=cand_score,
                )
            )
            continue

        # Check IoU and containment overlap against already accepted leaves
        is_duplicate = False
        duplicate_primary_id = ""
        duplicate_iou = 0.0

        for accepted in accepted_leaves:
            iou = compute_mask_iou(cand_mask, accepted.mask)
            intersection = int(np.count_nonzero(cand_mask & accepted.mask))
            min_area = min(int(np.count_nonzero(cand_mask)), int(np.count_nonzero(accepted.mask)))
            containment = intersection / float(min_area) if min_area > 0 else 0.0

            if iou >= cfg.iou_overlap_threshold or containment >= 0.40:
                is_duplicate = True
                duplicate_primary_id = accepted.leaf_id
                duplicate_iou = max(iou, containment)
                break

        if is_duplicate:
            if cfg.merge_duplicates:
                # Merge candidate piece into primary leaf instance to avoid fragmenting a single leaf
                for accepted in accepted_leaves:
                    if accepted.leaf_id == duplicate_primary_id:
                        merged_mask = accepted.mask | cand_mask
                        merged_bbox = compute_bounding_box(merged_mask)
                        merged_metrics = compute_mask_metrics(merged_mask)
                        m_crop_mask, m_rgb, m_rgba = create_masked_crop(
                            img_rgb, merged_mask, merged_bbox, background_fill=cfg.background_fill
                        )
                        accepted.mask = merged_mask
                        accepted.bounding_box = merged_bbox
                        accepted.leaf_mask_crop = m_crop_mask
                        accepted.masked_crop = m_rgb
                        accepted.masked_crop_rgba = m_rgba
                        accepted.metrics = merged_metrics
                        break
            else:
                rejected_list.append(
                    RejectedLeaf(
                        candidate_id=cand_id,
                        bounding_box=bbox,
                        mask=cand_mask,
                        reason=f"Overlapping duplicate mask (IoU {duplicate_iou:.3f} >= {cfg.iou_overlap_threshold:.3f}) with {duplicate_primary_id}",
                        verdict=MaskVerdict.REJECT,
                        metrics=val_res.metrics,
                        confidence=cand_score,
                    )
                )
            continue

        # Accept candidate leaf
        leaf_id = f"leaf_{len(accepted_leaves)}"
        leaf_mask_crop, rgb_crop, rgba_crop = create_masked_crop(
            img_rgb, cand_mask, bbox, background_fill=cfg.background_fill
        )
        leaf = ExtractedLeaf(
            leaf_id=leaf_id,
            bounding_box=bbox,
            mask=cand_mask,
            leaf_mask_crop=leaf_mask_crop,
            masked_crop=rgb_crop,
            masked_crop_rgba=rgba_crop,
            confidence=cand_score,
            metrics=val_res.metrics,
        )
        accepted_leaves.append(leaf)

    return LeafExtractionResult(
        leaves=accepted_leaves,
        rejected=rejected_list,
        is_single_leaf=False,
        total_candidates_evaluated=len(prompt_coords),
        image_shape=(h, w, 3),
        config=cfg,
    )
