"""EfficientNetB0 transfer learning architecture module (T5-022).

Conforms to TECH_RULES and Canonical Overview:
- Base model: torchvision.models.efficientnet_b0 with pretrained ImageNet-1K weights.
- Verified weights licence: ImageNet Terms of Access / BSD 3-Clause (torchvision).
  Pretrained weights ported from Tan & Le (2019) 'EfficientNet: Rethinking Model Scaling'.
- Head replacement: Classification head replaced to match the target crop's class count
  (lettuce=4, spinach=6, tomato=11, or arbitrary num_classes).
- Configurable freeze-then-unfreeze schedule with structured logging and transition auditing.
- Preserves canonical input size (3, 224, 224) and ImageNet normalisation (mean/std).
- Records weights identifier, input size, normalisation, and licence in metadata.
- Fully compatible with PyTorch autograd and the generic training pipeline (T5-020).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Final, Literal

import torch
import torchvision.models as tv_models
from torch import nn
from torchvision.models.efficientnet import EfficientNet_B0_Weights

from ml.labels import IMAGE_CLASSES

LOGGER = logging.getLogger("ml.image_models.efficientnet")

# Canonical ImageNet input specification
DEFAULT_INPUT_SIZE: Final[tuple[int, int]] = (224, 224)
DEFAULT_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
DEFAULT_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)
DEFAULT_DROPOUT_RATE: Final[float] = 0.2
DEFAULT_IN_FEATURES: Final[int] = 1280

WEIGHTS_LICENSE_INFO: Final[str] = (
    "ImageNet Terms of Access / BSD 3-Clause (torchvision). Pretrained weights are ported from "
    "Tan & Le (2019) 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks'. "
    "Permissible for research, academic evaluation, and benchmarking under standard ImageNet terms."
)
DEFAULT_WEIGHTS_IDENTIFIER: Final[str] = "EfficientNet_B0_Weights.IMAGENET1K_V1"


@dataclass
class FreezeScheduleConfig:
    """Configurable freeze-then-unfreeze schedule for fine-tuning EfficientNetB0.

    Attributes:
        initial_freeze: Whether to freeze backbone parameters on initialization.
        unfreeze_at_epoch: Epoch number (1-indexed) at which to trigger unfreezing. If None, remains frozen.
        unfreeze_mode: Strategy for unfreezing:
            - 'all': unfreeze all backbone parameters (entire features extractor).
            - 'top_blocks': unfreeze only the top N inverted residual / conv blocks.
            - 'none': keep backbone frozen throughout training.
        num_unfrozen_blocks: Number of top feature blocks to unfreeze when unfreeze_mode == 'top_blocks'.
                             EfficientNetB0 features has 9 blocks (indices 0..8); num_unfrozen_blocks=2 unfreezes blocks 7 and 8.
    """

    initial_freeze: bool = True
    unfreeze_at_epoch: int | None = 5
    unfreeze_mode: Literal["all", "top_blocks", "none"] = "all"
    num_unfrozen_blocks: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EfficientNetConfig:
    """Hyperparameters and configuration for EfficientNetB0 transfer learning."""

    crop: str | None = None
    num_classes: int | None = None
    weights: str | EfficientNet_B0_Weights | None = "DEFAULT"
    dropout_rate: float = DEFAULT_DROPOUT_RATE
    fc_dim: int | None = (
        None  # Optional intermediate layer: 1280 -> fc_dim -> num_classes
    )
    freeze_schedule: FreezeScheduleConfig = field(default_factory=FreezeScheduleConfig)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if isinstance(self.weights, EfficientNet_B0_Weights):
            data["weights"] = str(self.weights)
        return data


class EfficientNetB0(nn.Module):
    """EfficientNetB0 transfer learning architecture with configurable head and freeze schedule."""

    def __init__(
        self,
        crop: str | None = None,
        num_classes: int | None = None,
        weights: str | EfficientNet_B0_Weights | None = "DEFAULT",
        dropout_rate: float = DEFAULT_DROPOUT_RATE,
        fc_dim: int | None = None,
        freeze_schedule: FreezeScheduleConfig | None = None,
        config: EfficientNetConfig | None = None,
    ) -> None:
        """Initialize the EfficientNetB0 transfer learning model.

        Args:
            crop: Target crop cultivar ('lettuce', 'spinach', 'tomato').
            num_classes: Number of target output classes. Required if crop is not supplied.
            weights: Pretrained weights identifier or object ('DEFAULT', 'IMAGENET1K_V1', None).
            dropout_rate: Dropout probability in the replaced classifier head.
            fc_dim: Optional hidden dimension for an intermediate layer in the head.
            freeze_schedule: Configuration for freeze/unfreeze lifecycle.
            config: Optional comprehensive EfficientNetConfig. Overrides individual kwargs if passed.

        Raises:
            ValueError: If neither num_classes nor crop is provided, or if crop is unknown.
        """
        super().__init__()

        # Resolve config
        if config is not None:
            resolved_crop = config.crop
            resolved_num_classes = config.num_classes
            resolved_weights = config.weights
            resolved_dropout = config.dropout_rate
            resolved_fc_dim = config.fc_dim
            resolved_schedule = config.freeze_schedule
        else:
            resolved_crop = crop
            resolved_num_classes = num_classes
            resolved_weights = weights
            resolved_dropout = dropout_rate
            resolved_fc_dim = fc_dim
            resolved_schedule = freeze_schedule or FreezeScheduleConfig()

        # 1. Resolve output class count and names
        if resolved_crop is not None:
            clean_crop = resolved_crop.strip().lower()
            if clean_crop not in IMAGE_CLASSES:
                raise ValueError(
                    f"Unknown crop '{resolved_crop}'. Valid crops are: {sorted(IMAGE_CLASSES.keys())}"
                )
            self.crop: str | None = clean_crop
            self.classes: list[str] = list(IMAGE_CLASSES[clean_crop])
            self.num_classes: int = len(self.classes)
        elif resolved_num_classes is not None:
            if resolved_num_classes <= 0:
                raise ValueError(
                    f"num_classes must be positive, got {resolved_num_classes}"
                )
            self.crop = None
            self.classes = [f"class_{i}" for i in range(resolved_num_classes)]
            self.num_classes = resolved_num_classes
        else:
            raise ValueError(
                "Either 'num_classes' or 'crop' must be provided to EfficientNetB0."
            )

        self.dropout_rate = resolved_dropout
        self.fc_dim = resolved_fc_dim
        self.freeze_schedule = resolved_schedule
        self.schedule_history: list[dict[str, Any]] = []

        # 2. Resolve weights identifier and object
        weights_obj, weights_id = self._resolve_weights(resolved_weights)
        self.weights_identifier: str = weights_id

        # 3. Construct base model
        LOGGER.info(
            "Constructing EfficientNetB0 with weights='%s' (classes=%d, crop=%s)...",
            self.weights_identifier,
            self.num_classes,
            self.crop,
        )
        self.base_model = tv_models.efficientnet_b0(weights=weights_obj)

        # 4. Replace classifier head (Acceptance Criterion 1)
        in_features = self.base_model.classifier[1].in_features
        assert in_features == DEFAULT_IN_FEATURES, (
            f"Unexpected in_features: {in_features}"
        )
        self.in_features = in_features

        if self.fc_dim is not None and self.fc_dim > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dropout_rate, inplace=True),
                nn.Linear(in_features, self.fc_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(p=self.dropout_rate, inplace=True),
                nn.Linear(self.fc_dim, self.num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dropout_rate, inplace=True),
                nn.Linear(in_features, self.num_classes),
            )
        self.base_model.classifier = self.classifier

        # 5. Apply initial freeze schedule if configured (Acceptance Criterion 2)
        self._backbone_is_frozen: bool = False
        if self.freeze_schedule.initial_freeze:
            self.freeze_backbone()

    @staticmethod
    def _resolve_weights(
        weights_arg: str | EfficientNet_B0_Weights | None,
    ) -> tuple[EfficientNet_B0_Weights | None, str]:
        """Map weights specification to (torchvision_weights_object, weights_identifier_string)."""
        if weights_arg is None or str(weights_arg).lower() in ("none", ""):
            return None, "none"

        if isinstance(weights_arg, EfficientNet_B0_Weights):
            return weights_arg, str(weights_arg)

        clean_str = str(weights_arg).strip().upper()
        if clean_str in (
            "DEFAULT",
            "IMAGENET1K_V1",
            "EFFICIENTNET_B0_WEIGHTS.DEFAULT",
            "EFFICIENTNET_B0_WEIGHTS.IMAGENET1K_V1",
        ):
            return EfficientNet_B0_Weights.DEFAULT, DEFAULT_WEIGHTS_IDENTIFIER

        raise ValueError(
            f"Unsupported weights specification: '{weights_arg}'. "
            f"Expected 'DEFAULT', 'IMAGENET1K_V1', None, or an EfficientNet_B0_Weights instance."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base model and custom classifier head."""
        return self.base_model(x)

    def freeze_backbone(self) -> dict[str, int]:
        """Freeze all parameters in the feature extraction backbone (features).

        The classifier head parameters remain trainable.
        """
        for param in self.base_model.features.parameters():
            param.requires_grad = False
        self._backbone_is_frozen = True

        counts = self.count_parameters()
        event = {
            "action": "freeze_backbone",
            "epoch": None,
            "mode": "all",
            "trainable_parameters": counts["trainable"],
            "frozen_parameters": counts["frozen"],
            "total_parameters": counts["total"],
        }
        self.schedule_history.append(event)
        LOGGER.info(
            "FreezeSchedule: Backbone features frozen. Trainable params: %d | Frozen params: %d",
            counts["trainable"],
            counts["frozen"],
        )
        return counts

    def unfreeze_backbone(
        self,
        mode: Literal["all", "top_blocks"] = "all",
        num_blocks: int | None = None,
        epoch: int | None = None,
    ) -> dict[str, int]:
        """Unfreeze backbone parameters per the requested mode.

        Args:
            mode: 'all' to unfreeze all 9 feature blocks, or 'top_blocks' for the top N blocks.
            num_blocks: Number of blocks from the top to unfreeze (if mode == 'top_blocks').
            epoch: Current epoch for schedule audit logging.
        """
        total_blocks = len(self.base_model.features)  # 9 blocks in EfficientNetB0

        if mode == "all":
            for param in self.base_model.features.parameters():
                param.requires_grad = True
            self._backbone_is_frozen = False
            unfrozen_desc = "all 9 blocks"
        elif mode == "top_blocks":
            n = (
                num_blocks
                if num_blocks is not None
                else self.freeze_schedule.num_unfrozen_blocks
            )
            n = max(1, min(total_blocks, n))
            cutoff_idx = total_blocks - n

            for idx, block in enumerate(self.base_model.features):
                requires_grad = idx >= cutoff_idx
                for param in block.parameters():
                    param.requires_grad = requires_grad

            self._backbone_is_frozen = False
            unfrozen_desc = f"top {n} blocks (indices {cutoff_idx}..{total_blocks - 1})"
        else:
            raise ValueError(
                f"Unsupported unfreeze mode '{mode}'. Expected 'all' or 'top_blocks'."
            )

        counts = self.count_parameters()
        event = {
            "action": "unfreeze_backbone",
            "epoch": epoch,
            "mode": mode,
            "unfrozen_description": unfrozen_desc,
            "trainable_parameters": counts["trainable"],
            "frozen_parameters": counts["frozen"],
            "total_parameters": counts["total"],
        }
        self.schedule_history.append(event)
        LOGGER.info(
            "FreezeSchedule: Unfroze %s at epoch %s. Trainable params: %d | Frozen params: %d",
            unfrozen_desc,
            epoch if epoch is not None else "manual",
            counts["trainable"],
            counts["frozen"],
        )
        return counts

    def step_schedule(self, epoch: int) -> bool:
        """Step the freeze/unfreeze schedule at the start of a training epoch.

        Args:
            epoch: Current training epoch (1-indexed).

        Returns:
            True if an unfreeze transition occurred, False otherwise.
        """
        if self.freeze_schedule.unfreeze_mode == "none":
            return False

        target_epoch = self.freeze_schedule.unfreeze_at_epoch
        if (
            target_epoch is not None
            and epoch >= target_epoch
            and self._backbone_is_frozen
        ):
            LOGGER.info(
                "FreezeSchedule triggered transition at epoch %d (target threshold: %d).",
                epoch,
                target_epoch,
            )
            self.unfreeze_backbone(
                mode=self.freeze_schedule.unfreeze_mode,
                num_blocks=self.freeze_schedule.num_unfrozen_blocks,
                epoch=epoch,
            )
            return True

        return False

    def count_parameters(self) -> dict[str, int]:
        """Count total, trainable, and frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
        }

    def get_shape_table(
        self,
        input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    ) -> list[dict[str, Any]]:
        """Trace layer-by-layer tensor activations for auditability and shape verification."""
        was_training = self.training
        self.eval()

        table: list[dict[str, Any]] = []
        hooks = []

        def make_hook(layer_name: str):
            def hook(_module: nn.Module, _inp: Any, output: Any) -> None:
                if isinstance(output, torch.Tensor):
                    shape_list = list(output.shape)
                elif (
                    isinstance(output, (tuple, list))
                    and output
                    and isinstance(output[0], torch.Tensor)
                ):
                    shape_list = list(output[0].shape)
                else:
                    shape_list = []
                table.append({"layer": layer_name, "output_shape": shape_list})

            return hook

        try:
            # Register hooks on feature blocks and head
            for idx, block in enumerate(self.base_model.features):
                hooks.append(
                    block.register_forward_hook(make_hook(f"features.block_{idx}"))
                )
            hooks.append(
                self.base_model.avgpool.register_forward_hook(make_hook("avgpool"))
            )
            hooks.append(
                self.base_model.classifier.register_forward_hook(
                    make_hook("classifier_head")
                )
            )

            dummy = torch.zeros(input_shape)
            with torch.no_grad():
                self.forward(dummy)
        finally:
            for h in hooks:
                h.remove()
            self.train(was_training)

        return table

    def get_metadata(self) -> dict[str, Any]:
        """Compile comprehensive architectural and provenance metadata (Acceptance Criterion 3)."""
        counts = self.count_parameters()
        return {
            "model_family": "EfficientNetB0",
            "architecture": "efficientnet_b0",
            "weights_identifier": self.weights_identifier,
            "weights_license": WEIGHTS_LICENSE_INFO,
            "input_size": [3, DEFAULT_INPUT_SIZE[0], DEFAULT_INPUT_SIZE[1]],
            "normalization": {
                "mean": list(DEFAULT_MEAN),
                "std": list(DEFAULT_STD),
            },
            "crop": self.crop,
            "num_classes": self.num_classes,
            "classes": self.classes,
            "dropout_rate": self.dropout_rate,
            "fc_dim": self.fc_dim,
            "in_features": self.in_features,
            "total_parameters": counts["total"],
            "trainable_parameters": counts["trainable"],
            "frozen_parameters": counts["frozen"],
            "freeze_schedule": self.freeze_schedule.to_dict(),
            "schedule_history": self.schedule_history,
        }


def build_efficientnet_b0(
    crop: str | None = None,
    num_classes: int | None = None,
    weights: str | EfficientNet_B0_Weights | None = "DEFAULT",
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    fc_dim: int | None = None,
    freeze_schedule: FreezeScheduleConfig | None = None,
    config: EfficientNetConfig | None = None,
) -> EfficientNetB0:
    """Factory function constructing a configured EfficientNetB0 transfer learning model."""
    return EfficientNetB0(
        crop=crop,
        num_classes=num_classes,
        weights=weights,
        dropout_rate=dropout_rate,
        fc_dim=fc_dim,
        freeze_schedule=freeze_schedule,
        config=config,
    )
