"""Custom configurable Convolutional Neural Network architecture module (T5-019).

Adheres to TECH_RULES and Canonical Overview:
- Small configurable CNN with modular conv blocks, batch normalisation, pooling, dropout, and classifier head.
- Output dimension matches the crop's canonical class count (lettuce=4, spinach=6, tomato=11).
- Provides parameter count computation and structured layer-by-layer shape table logging.
- Fully compatible with PyTorch autograd and the generic training loop (T5-020).
"""

from __future__ import annotations

import logging
from typing import Any, Final

import torch
from torch import nn

from ml.labels import IMAGE_CLASSES

LOGGER = logging.getLogger("ml.image_models.custom_cnn")

# Default architectural hyperparameters
DEFAULT_CONV_CHANNELS: Final[tuple[int, ...]] = (32, 64, 128, 256)
DEFAULT_DROPOUT_RATE: Final[float] = 0.3
DEFAULT_FC_DIM: Final[int] = 256


class ConvBlock(nn.Module):
    """Modular 2D Convolutional Block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,  # BatchNorm includes affine bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class CustomCNN(nn.Module):
    """Small, robust, configurable Convolutional Neural Network for hydroponic leaf diagnosis."""

    def __init__(
        self,
        num_classes: int | None = None,
        crop: str | None = None,
        in_channels: int = 3,
        conv_channels: tuple[int, ...] = DEFAULT_CONV_CHANNELS,
        dropout_rate: float = DEFAULT_DROPOUT_RATE,
        fc_dim: int = DEFAULT_FC_DIM,
    ) -> None:
        """Initialize the CustomCNN model.

        Args:
            num_classes: Number of target output classes. Required if crop is not supplied.
            crop: Target crop cultivar ('lettuce', 'spinach', 'tomato').
            in_channels: Number of input image color channels (default 3 for RGB).
            conv_channels: Sequence of channel counts for sequential conv blocks.
            dropout_rate: Dropout probability in the classifier head.
            fc_dim: Hidden representation dimension in classifier head.

        Raises:
            ValueError: If neither num_classes nor crop is provided, or if crop is unknown.
        """
        super().__init__()

        # Resolve output class count
        if crop is not None:
            clean_crop = crop.strip().lower()
            if clean_crop not in IMAGE_CLASSES:
                raise ValueError(
                    f"Unknown crop '{crop}'. Valid crops are: {sorted(IMAGE_CLASSES.keys())}"
                )
            resolved_num_classes = len(IMAGE_CLASSES[clean_crop])
            self.crop = clean_crop
        elif num_classes is not None:
            if num_classes <= 0:
                raise ValueError(f"num_classes must be positive, got {num_classes}")
            resolved_num_classes = num_classes
            self.crop = None
        else:
            raise ValueError(
                "Either 'num_classes' or 'crop' must be provided to CustomCNN."
            )

        self.num_classes = resolved_num_classes
        self.in_channels = in_channels
        self.conv_channels = tuple(conv_channels)
        self.dropout_rate = dropout_rate
        self.fc_dim = fc_dim

        # 1. Feature Extractor: Sequential Conv Blocks
        blocks: list[nn.Module] = []
        current_in = in_channels
        for idx, out_ch in enumerate(self.conv_channels):
            blocks.append(ConvBlock(in_channels=current_in, out_channels=out_ch))
            current_in = out_ch
        self.features = nn.Sequential(*blocks)

        # 2. Adaptive Spatial Pooling (Resolution Agnostic)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 3. Dense Classification Head
        final_conv_ch = self.conv_channels[-1] if self.conv_channels else in_channels
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_conv_ch, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_dim, resolved_num_classes),
        )

        # Weight initialization (He / Kaiming normal for ReLU)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize layer weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing raw unnormalized class logits.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        feats = self.features(x)
        pooled = self.adaptive_pool(feats)
        logits = self.classifier(pooled)
        return logits

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters in the model.

        Args:
            trainable_only: If True, counts only parameters requiring gradients.

        Returns:
            Total parameter count.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_shape_table(
        self,
        input_shape: tuple[int, ...] = (1, 3, 224, 224),
    ) -> list[dict[str, Any]]:
        """Generate a table tracing layer names, output tensor shapes, and parameter counts.

        Args:
            input_shape: Mock input dimensions (B, C, H, W).

        Returns:
            List of dictionaries containing layer summary statistics.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                table: list[dict[str, Any]] = []
                x = torch.zeros(input_shape, dtype=torch.float32)

                table.append(
                    {
                        "stage": "Input",
                        "layer": "InputImage",
                        "output_shape": list(x.shape),
                        "num_parameters": 0,
                    }
                )

                # Trace conv feature blocks
                for i, block in enumerate(self.features):
                    x = block(x)
                    num_params = sum(p.numel() for p in block.parameters())
                    table.append(
                        {
                            "stage": f"ConvBlock_{i + 1}",
                            "layer": f"ConvBlock_{i + 1} (Conv-BN-ReLU-MaxPool)",
                            "output_shape": list(x.shape),
                            "num_parameters": num_params,
                        }
                    )

                # Trace adaptive pooling
                x = self.adaptive_pool(x)
                table.append(
                    {
                        "stage": "Pooling",
                        "layer": "AdaptiveAvgPool2d",
                        "output_shape": list(x.shape),
                        "num_parameters": 0,
                    }
                )

                # Trace classifier head
                flattened = nn.Flatten()(x)
                table.append(
                    {
                        "stage": "Classifier",
                        "layer": "Flatten",
                        "output_shape": list(flattened.shape),
                        "num_parameters": 0,
                    }
                )

                fc1 = self.classifier[1]
                x_fc1 = fc1(flattened)
                table.append(
                    {
                        "stage": "Classifier",
                        "layer": f"Linear ({flattened.shape[1]} -> {self.fc_dim})",
                        "output_shape": list(x_fc1.shape),
                        "num_parameters": sum(p.numel() for p in fc1.parameters()),
                    }
                )

                bn_fc = self.classifier[2]
                x_bn = bn_fc(x_fc1)
                table.append(
                    {
                        "stage": "Classifier",
                        "layer": "BatchNorm1d",
                        "output_shape": list(x_bn.shape),
                        "num_parameters": sum(p.numel() for p in bn_fc.parameters()),
                    }
                )

                head = self.classifier[5]
                out = head(x_bn)
                table.append(
                    {
                        "stage": "Classifier",
                        "layer": f"Linear ({self.fc_dim} -> {self.num_classes})",
                        "output_shape": list(out.shape),
                        "num_parameters": sum(p.numel() for p in head.parameters()),
                    }
                )

                return table
        finally:
            self.train(was_training)

    def log_architecture_summary(
        self,
        input_shape: tuple[int, ...] = (1, 3, 224, 224),
        logger: logging.Logger | None = None,
    ) -> str:
        """Log parameter count and format shape table as human-readable string.

        Args:
            input_shape: Mock input dimensions.
            logger: Optional logger to send info messages to.

        Returns:
            Formatted architecture summary string.
        """
        active_logger = logger or LOGGER
        total_params = self.count_parameters(trainable_only=False)
        trainable_params = self.count_parameters(trainable_only=True)
        table = self.get_shape_table(input_shape=input_shape)

        lines: list[str] = [
            "=" * 78,
            f"CustomCNN Architecture Summary (num_classes={self.num_classes}, crop={self.crop})",
            "=" * 78,
            f"{'Stage':<16} {'Layer':<36} {'Output Shape':<18} {'Parameters':<10}",
            "-" * 78,
        ]

        for row in table:
            shape_str = str(row["output_shape"])
            params_str = (
                f"{row['num_parameters']:,}" if row["num_parameters"] > 0 else "-"
            )
            lines.append(
                f"{row['stage']:<16} {row['layer']:<36} {shape_str:<18} {params_str:<10}"
            )

        lines.extend(
            [
                "-" * 78,
                f"Total Parameters:     {total_params:,}",
                f"Trainable Parameters: {trainable_params:,}",
                "=" * 78,
            ]
        )

        summary_text = "\n".join(lines)
        active_logger.info(
            "CustomCNN model configured: crop=%s classes=%d total_params=%d trainable_params=%d",
            self.crop,
            self.num_classes,
            total_params,
            trainable_params,
        )
        active_logger.info("\n%s", summary_text)
        return summary_text


def build_custom_cnn(
    crop: str | None = None,
    num_classes: int | None = None,
    **kwargs: Any,
) -> CustomCNN:
    """Factory function instantiating a CustomCNN with appropriate output dimension."""
    return CustomCNN(crop=crop, num_classes=num_classes, **kwargs)
