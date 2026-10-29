"""SAM 2.1 foundation model segmenter wrapper with lazy-loading, OOM handling, and CPU fallback (T5-008).

Adheres to TECH_RULES & ADR 0003:
- Lazy loading: Checkpoint and predictor are not loaded until first use.
- Device honesty & agnostic execution: Uses DeviceManager to detect GPU/CPU.
- Memory safety: Cleans up VRAM on error/unload.
- OOM resilience: Catches CUDA OOM, frees memory, and triggers documented CPU fallback or raises CudaOOMError.
- Returns binary boolean mask and explicit config/version descriptor string.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from backend.app.core.device import DeviceManager, get_device_manager

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_CHECKPOINT = "models/image/sam2.1_hiera_tiny.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_VERSION_TAG = "sam2.1_hiera_tiny:v1"


# ==============================================================================
# Typed Exceptions
# ==============================================================================


class CudaOOMError(RuntimeError):
    """Raised when CUDA runs out of memory during SAM 2.1 inference or model loading."""


class SegmentationError(RuntimeError):
    """Raised when segmentation fails due to invalid inputs or processing errors."""


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class SegmentationResult:
    """Structured result returned by the SAM 2.1 segmenter."""

    mask: np.ndarray  # 2D boolean array (H, W)
    config_version: str  # Config and checkpoint version identifier
    device_used: str  # 'cuda:0' or 'cpu'
    score: float  # Confidence score of the selected mask
    fallback_triggered: bool = False  # True if CPU fallback was triggered by OOM


# ==============================================================================
# Segmenter Implementation
# ==============================================================================


class SAM2Segmenter:
    """Wrapper around Meta's SAM 2.1 for leaf-level instance segmentation."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config_path: str | None = None,
        force_cpu: bool | None = None,
        fallback_on_oom: bool = True,
    ) -> None:
        resolved_cp = (
            checkpoint_path or os.getenv("SAM2_CHECKPOINT") or DEFAULT_CHECKPOINT
        )
        self.checkpoint_path = Path(str(resolved_cp))
        self.config_path = str(config_path or os.getenv("SAM2_CONFIG", DEFAULT_CONFIG))
        self.device_manager = (
            DeviceManager(force_cpu=force_cpu)
            if force_cpu is not None
            else get_device_manager()
        )
        self.fallback_on_oom = fallback_on_oom

        # Lazy loading state: nothing is loaded in __init__
        self._model: Any = None
        self._predictor: Any = None
        self._loaded_device: str | None = None
        self._fallback_triggered: bool = False

    @property
    def is_loaded(self) -> bool:
        """Return True if model weights and predictor are currently in memory."""
        return self._model is not None and self._predictor is not None

    @property
    def loaded_device(self) -> str | None:
        """Return the device currently hosting the loaded model, if any."""
        return self._loaded_device

    @property
    def config_version_string(self) -> str:
        """Return formal configuration and version string for audit trails."""
        return f"{SAM2_VERSION_TAG}:{self.config_path}"

    def free_memory(self) -> None:
        """Release cached GPU memory and invoke garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def unload(self) -> None:
        """Explicitly unload model from memory and free GPU resources."""
        self._model = None
        self._predictor = None
        self._loaded_device = None
        self.free_memory()

    def load_model(
        self,
        target_device: str | None = None,
        simulate_oom: bool = False,
    ) -> None:
        """Load SAM 2.1 into memory with OOM detection and CPU fallback."""
        device = target_device or self.device_manager.get_device_str()

        # If already loaded on the requested device, do nothing
        if self.is_loaded and self._loaded_device == device and not simulate_oom:
            return

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        resolved_checkpoint = self.checkpoint_path
        if not resolved_checkpoint.is_absolute():
            resolved_checkpoint = PROJECT_ROOT / resolved_checkpoint

        if not resolved_checkpoint.exists():
            raise FileNotFoundError(
                f"SAM 2.1 checkpoint file missing: {resolved_checkpoint}"
            )

        # Simulate OOM condition if requested
        if simulate_oom:
            self.free_memory()
            if self.fallback_on_oom and device.startswith("cuda"):
                # Trigger documented CPU fallback
                self.load_model(target_device="cpu")
                self._fallback_triggered = True
                return
            raise CudaOOMError("Simulated CUDA out-of-memory during SAM 2.1 load.")

        try:
            sam_model = build_sam2(
                self.config_path, str(resolved_checkpoint), device=device
            )
            self._model = sam_model
            self._predictor = SAM2ImagePredictor(sam_model)
            self._loaded_device = device
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            is_oom = "out of memory" in str(exc).lower() or isinstance(
                exc, torch.cuda.OutOfMemoryError
            )
            self.free_memory()

            if is_oom and self.fallback_on_oom and device.startswith("cuda"):
                # Fallback to CPU as documented in ADR 0003
                self.load_model(target_device="cpu")
                self._fallback_triggered = True
                return

            if is_oom:
                raise CudaOOMError(
                    f"CUDA OOM loading SAM 2.1 on {device}: {exc}"
                ) from exc
            raise

    def segment(
        self,
        image: np.ndarray | str | Path,
        point_coords: np.ndarray | list[list[int]] | None = None,
        point_labels: np.ndarray | list[int] | None = None,
        box: np.ndarray | list[int] | None = None,
        simulate_oom: bool = False,
    ) -> SegmentationResult:
        """Perform leaf instance segmentation returning binary mask and config string.

        Args:
            image: Input RGB numpy array or path to image file.
            point_coords: Optional Nx2 array of prompt point coordinates.
            point_labels: Optional array of prompt point labels (1=foreground, 0=background).
            box: Optional bounding box [x1, y1, x2, y2].
            simulate_oom: When True, simulates CUDA OOM during inference to verify fallback.

        Returns:
            SegmentationResult containing 2D boolean mask and provenance metadata.
        """
        # Ensure model is lazily loaded on first use
        if not self.is_loaded:
            self.load_model()

        # Load and validate image input
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.is_absolute():
                img_path = PROJECT_ROOT / img_path
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise SegmentationError(f"Failed to read image from path: {img_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img_rgb = image
        else:
            raise SegmentationError(f"Unsupported image type: {type(image)}")

        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise SegmentationError(
                f"Image must have shape (H, W, 3), got {img_rgb.shape}"
            )

        h, w, _ = img_rgb.shape

        # Use fast single-leaf center point prompt if no prompt coordinates supplied
        input_points: np.ndarray | None
        input_labels: np.ndarray | None
        if point_coords is None and box is None:
            input_points = np.array([[w // 2, h // 2]])
            input_labels = np.array([1])
        else:
            input_points = np.array(point_coords) if point_coords is not None else None
            input_labels = np.array(point_labels) if point_labels is not None else None

        input_box = np.array(box) if box is not None else None

        # Check for simulated OOM during inference
        if (
            simulate_oom
            and self._loaded_device
            and self._loaded_device.startswith("cuda")
        ):
            self.free_memory()
            if self.fallback_on_oom:
                # Fallback to CPU and retry
                self.load_model(target_device="cpu")
                self._fallback_triggered = True
                return self.segment(
                    image=img_rgb,
                    point_coords=input_points,
                    point_labels=input_labels,
                    box=input_box,
                    simulate_oom=False,
                )
            raise CudaOOMError("Simulated CUDA OOM during SAM 2.1 inference.")

        try:
            self._predictor.set_image(img_rgb)
            masks, scores, _ = self._predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=input_box,
                multimask_output=True,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            is_oom = "out of memory" in str(exc).lower() or isinstance(
                exc, torch.cuda.OutOfMemoryError
            )
            self.free_memory()

            if is_oom and self.fallback_on_oom and self._loaded_device != "cpu":
                # Fallback to CPU as documented in ADR 0003
                self.load_model(target_device="cpu")
                self._fallback_triggered = True
                return self.segment(
                    image=img_rgb,
                    point_coords=input_points,
                    point_labels=input_labels,
                    box=input_box,
                    simulate_oom=False,
                )

            if is_oom:
                raise CudaOOMError(
                    f"CUDA OOM during SAM 2.1 inference on {self._loaded_device}: {exc}"
                ) from exc
            raise

        # Select highest scoring mask
        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(bool)
        best_score = float(scores[best_idx])

        return SegmentationResult(
            mask=best_mask,
            config_version=self.config_version_string,
            device_used=str(self._loaded_device),
            score=round(best_score, 4),
            fallback_triggered=self._fallback_triggered,
        )
