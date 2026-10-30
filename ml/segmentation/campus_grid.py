"""Calibrated campus hydroponic NFT bench net-cup grid extraction.

Provides fixed high-accuracy spatial region-of-interest (ROI) bounding boxes
for the stationary wide-angle polyhouse CPPLUS IP camera.

Because the camera is rigidly mounted overhead facing the NFT channels:
- Seedling net cup locations remain stationary in pixel space across frames.
- Night IR illumination (grayscale, 2G - R - B = 0) renders ExG thresholding
  ineffective at night, but calibrated spatial ROIs reliably isolate every
  individual plant seedling regardless of diurnal lighting changes.
"""

from __future__ import annotations

from typing import List, Tuple
import cv2
import numpy as np

from ml.segmentation.leaf_extraction import (
    ExtractedLeaf,
    LeafExtractionConfig,
    LeafExtractionResult,
)
from ml.segmentation.mask_validation import MaskMetrics

# 48 calibrated normalized [x1, y1, x2, y2] bounding boxes directly mapped from
# annotated polyhouse plant net cups in the CPPLUS live camera frame.
CAMPUS_NET_CUP_ROIS: List[List[float]] = [
    [0.4678, 0.1372, 0.4844, 0.1545],
    [0.5312, 0.2222, 0.5400, 0.2413],
    [0.5605, 0.3038, 0.5801, 0.3403],
    [0.5898, 0.3299, 0.6104, 0.3611],
    [0.6260, 0.3351, 0.6494, 0.3837],
    [0.6875, 0.3872, 0.7158, 0.4323],
    [0.6982, 0.3628, 0.7080, 0.3802],
    [0.4824, 0.3316, 0.5068, 0.3785],
    [0.4932, 0.2726, 0.5166, 0.3194],
    [0.6182, 0.4358, 0.6455, 0.4844],
    [0.6650, 0.4323, 0.6904, 0.4653],
    [0.6943, 0.5000, 0.7158, 0.5417],
    [0.7529, 0.5365, 0.7832, 0.5903],
    [0.8027, 0.5712, 0.8545, 0.6458],
    [0.4170, 0.3837, 0.4414, 0.4375],
    [0.4766, 0.4080, 0.5000, 0.4601],
    [0.5293, 0.3958, 0.5723, 0.5174],
    [0.5781, 0.5000, 0.5986, 0.5417],
    [0.6152, 0.5382, 0.6367, 0.5799],
    [0.6670, 0.5642, 0.6855, 0.5955],
    [0.4131, 0.4740, 0.4375, 0.5208],
    [0.4482, 0.5156, 0.4902, 0.5851],
    [0.5381, 0.5868, 0.5615, 0.6319],
    [0.6865, 0.5972, 0.7090, 0.6372],
    [0.7549, 0.6615, 0.7793, 0.7014],
    [0.8232, 0.7083, 0.8535, 0.7604],
    [0.8926, 0.7812, 0.9180, 0.8247],
    [0.9297, 0.8038, 0.9502, 0.8472],
    [0.9629, 0.8403, 0.9785, 0.8715],
    [0.3594, 0.5399, 0.3896, 0.5990],
    [0.4258, 0.6250, 0.4512, 0.6736],
    [0.5117, 0.6719, 0.5527, 0.7101],
    [0.6045, 0.6615, 0.6318, 0.7049],
    [0.6797, 0.7135, 0.7041, 0.7656],
    [0.7559, 0.7847, 0.7803, 0.8299],
    [0.8311, 0.8524, 0.8564, 0.8958],
    [0.8965, 0.9132, 0.9160, 0.9462],
    [0.9277, 0.9410, 0.9453, 0.9740],
    [0.4941, 0.6892, 0.5498, 0.7743],
    [0.5586, 0.7726, 0.5850, 0.8194],
    [0.6318, 0.8142, 0.6680, 0.8837],
    [0.6748, 0.8785, 0.7021, 0.9167],
    [0.7461, 0.9149, 0.7754, 0.9809],
    [0.4971, 0.8038, 0.5264, 0.8420],
    [0.7080, 0.9115, 0.7383, 0.9514],
    [0.4482, 0.8750, 0.4668, 0.8924],
    [0.4824, 0.8906, 0.5029, 0.9236],
    [0.4971, 0.9062, 0.5244, 0.9531],
]


def extract_campus_grid_leaves(
    image: np.ndarray,
    rois: List[List[float]] | None = None,
    padding_ratio: float = 0.05,
    max_leaves: int | None = None,
) -> LeafExtractionResult:
    """Extract individual hydroponic plant seedlings from stationary bench net-cup ROIs.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3).
    rois : List[List[float]], optional
        Custom list of normalized [x1, y1, x2, y2] bounding boxes. Defaults to CAMPUS_NET_CUP_ROIS.
    padding_ratio : float
        Fractional padding added to each bounding box to capture outward foliage.
    max_leaves : int, optional
        Maximum number of plants to return. If None, returns all calibrated positions.

    Returns
    -------
    LeafExtractionResult
        Extracted leaf instances conforming to multi-leaf prediction interface.
    """
    h, w, _ = image.shape
    active_rois = rois if rois is not None else CAMPUS_NET_CUP_ROIS
    if max_leaves is not None and max_leaves > 0:
        active_rois = active_rois[:max_leaves]

    leaves: List[ExtractedLeaf] = []

    for idx, roi in enumerate(active_rois):
        x1_norm, y1_norm, x2_norm, y2_norm = roi
        px1 = int(round(x1_norm * w))
        py1 = int(round(y1_norm * h))
        px2 = int(round(x2_norm * w))
        py2 = int(round(y2_norm * h))

        box_w = max(1, px2 - px1)
        box_h = max(1, py2 - py1)

        pad_x = int(round(box_w * padding_ratio))
        pad_y = int(round(box_h * padding_ratio))

        crop_x1 = max(0, px1 - pad_x)
        crop_y1 = max(0, py1 - pad_y)
        crop_x2 = min(w, px2 + pad_x)
        crop_y2 = min(h, py2 + pad_y)

        # Extract RGB crop
        rgb_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_h, crop_w, _ = rgb_crop.shape

        # Synthesize elliptical mask centered on net cup
        mask = np.zeros((h, w), dtype=bool)
        cv2.ellipse(
            mask.view(np.uint8),
            (px1 + box_w // 2, py1 + box_h // 2),
            (box_w // 2, box_h // 2),
            0,
            0,
            360,
            1,
            -1,
        )

        crop_mask = np.ones((crop_h, crop_w), dtype=bool)
        rgba_crop = np.dstack((rgb_crop, (crop_mask * 255).astype(np.uint8)))

        leaf_id = f"plant_{idx + 1:03d}"
        from ml.segmentation.mask_validation import compute_mask_metrics
        metrics = compute_mask_metrics(mask)

        leaves.append(
            ExtractedLeaf(
                leaf_id=leaf_id,
                bounding_box=(crop_x1, crop_y1, crop_x2, crop_y2),
                mask=mask,
                leaf_mask_crop=crop_mask,
                masked_crop=rgb_crop,
                masked_crop_rgba=rgba_crop,
                confidence=0.95,
                metrics=metrics,
            )
        )

    return LeafExtractionResult(
        leaves=leaves,
        rejected=[],
        is_single_leaf=False,
        total_candidates_evaluated=len(leaves),
        image_shape=(h, w, 3),
        config=LeafExtractionConfig(single_leaf_mode=False, max_leaves=len(leaves)),
    )
