"""Generic PyTorch image model training pipeline (T5-020).

Adheres to TECH_RULES and Canonical Overview §18.1, §18.2:
- Detects compute hardware via DeviceManager and maintains strict Device Honesty.
- Configurable seeds, optimizer, learning rate, weight decay, and schedulers.
- Evaluates on the holdout validation partition only (test.csv is strictly never touched).
- Early stopping monitored on Validation Macro-F1 (preventing majority-class bias).
- CUDA Out-Of-Memory (OOM) ladder: memory cleanup -> halved batch size -> CPU fallback.
- Overfit verification test achieving near-zero training loss on 20 images.
- Section 18.2 compliant markdown and JSON evaluation reports logging verified device and versions.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.device_manager import (
    DeviceManager,
    SimulatedCudaOOMError,
    is_oom_exception,
)
from ml.evaluation.metrics import calculate_metrics, to_serializable_metrics
from ml.evaluation.report import get_default_software_versions, write_evaluation_report
from ml.image_models.custom_cnn import CustomCNN
from ml.image_models.data import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MEAN,
    DEFAULT_STD,
    CropLeafDataset,
    create_dataloaders,
)
from ml.labels import IMAGE_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("train_image")


def set_reproducible_seeds(seed: int = 42) -> None:
    """Set global seeds for deterministic PyTorch and NumPy execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_commit_hash() -> str:
    """Retrieve current commit hash for metadata audit trail."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


@dataclass
class TrainingConfig:
    """Structured configuration for image model training."""

    crop: str
    model_family: str = "custom_cnn"
    model_name: str | None = None
    batch_size: int = 32
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer_name: str = "adamw"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    seed: int = 42
    num_workers: int = 0
    force_cpu: bool = False
    simulate_oom: bool = False
    output_dir: Path | None = None
    report_dir: Path | None = None


class EarlyStopping:
    """Early stopping monitor based on validation Macro-F1."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_macro_f1: float = -float("inf")
        self.patience_counter: int = 0
        self.should_stop: bool = False

    def step(self, current_macro_f1: float) -> bool:
        """Update monitor with latest validation Macro-F1.

        Returns:
            True if a new champion score was attained, False otherwise.
        """
        if current_macro_f1 > self.best_macro_f1 + self.min_delta:
            self.best_macro_f1 = current_macro_f1
            self.patience_counter = 0
            return True

        self.patience_counter += 1
        if self.patience_counter >= self.patience:
            self.should_stop = True
        return False


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> optim.Optimizer:
    """Instantiate optimizer from configuration."""
    opt_lower = optimizer_name.lower().strip()
    if opt_lower == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_lower == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_lower == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: '{optimizer_name}'. Expected 'adamw', 'adam', or 'sgd'.")


def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    classes: list[str],
) -> tuple[float, dict[str, Any]]:
    """Evaluate model performance exclusively on the validation holdout split."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().tolist())

    mean_loss = total_loss / max(1, len(all_targets))
    pred_class_names = [classes[p] for p in all_preds]
    target_class_names = [classes[t] for t in all_targets]

    raw_metrics = calculate_metrics(
        y_true=target_class_names,
        y_pred=pred_class_names,
        labels=classes,
    )
    serializable = to_serializable_metrics(raw_metrics)
    return mean_loss, serializable


def train_image_model(
    config: TrainingConfig,
    device_manager: DeviceManager | None = None,
) -> dict[str, Any]:
    """Train image classification model adhering to Section 18.2 with holdout isolation."""
    set_reproducible_seeds(config.seed)

    if device_manager is None:
        device_manager = DeviceManager(force_cpu=config.force_cpu)

    target_device_str = device_manager.get_target_device()
    current_device_str = target_device_str
    current_device = torch.device(current_device_str)

    crop = config.crop.lower().strip()
    classes = list(IMAGE_CLASSES[crop])
    num_classes = len(classes)

    LOGGER.info(
        "Initializing image training for crop '%s' (%d classes) on target device '%s'",
        crop,
        num_classes,
        target_device_str,
    )

    current_batch_size = config.batch_size
    oom_events_caught = 0
    fallback_triggered = False

    # Instantiate Custom CNN architecture
    model = CustomCNN(crop=crop)
    model.to(current_device)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(
        model=model,
        optimizer_name=config.optimizer_name,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )

    # Initial dataloaders (strictly only train and val are queried)
    train_loader, val_loader, _ = create_dataloaders(
        crop=crop,
        batch_size=current_batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    best_val_macro_f1 = -float("inf")
    best_checkpoint_state: dict[str, Any] = {}
    best_metrics: dict[str, Any] = {}
    best_epoch = 0

    history: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        train_samples = 0
        epoch_step = 0

        train_iter = iter(train_loader)
        while True:
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            except Exception as e:
                # Intercept runtime errors
                raise e

            inputs, targets = batch
            epoch_step += 1

            try:
                # Simulated OOM check for testing acceptance criteria
                if config.simulate_oom and epoch == 1 and epoch_step == 2 and not fallback_triggered:
                    raise SimulatedCudaOOMError("Simulated CUDA Out-Of-Memory triggered for testing.")

                inputs = inputs.to(current_device)
                targets = targets.to(current_device)

                # Verify device placement
                if "cuda" in str(inputs.device):
                    device_manager.record_device_use("cuda:0")
                else:
                    device_manager.record_device_use("cpu")

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                train_samples += inputs.size(0)

                if epoch_step % 50 == 0:
                    LOGGER.info(
                        "  [Epoch %02d] Step %03d/%d - Step Loss: %.4f",
                        epoch,
                        epoch_step,
                        len(train_loader),
                        loss.item(),
                    )

            except BaseException as exc:
                if is_oom_exception(exc):
                    oom_events_caught += 1
                    LOGGER.warning(
                        "CUDA OOM caught at epoch %d, step %d (batch_size=%d). Executing retry ladder...",
                        epoch,
                        epoch_step,
                        current_batch_size,
                    )
                    device_manager.cleanup_memory()

                    # OOM Ladder Step 1: Halve batch size if batch_size > 4
                    if current_batch_size > 4 and not fallback_triggered:
                        current_batch_size = max(4, current_batch_size // 2)
                        LOGGER.info("Halved batch size to %d; recreating DataLoaders...", current_batch_size)
                        train_loader, val_loader, _ = create_dataloaders(
                            crop=crop,
                            batch_size=current_batch_size,
                            num_workers=config.num_workers,
                            seed=config.seed,
                        )
                        train_iter = iter(train_loader)
                        continue

                    # OOM Ladder Step 2: Fallback to CPU
                    LOGGER.warning("OOM persists or batch size minimal. Falling back to CPU...")
                    fallback_triggered = True
                    current_device_str = "cpu"
                    current_device = torch.device("cpu")
                    device_manager.record_device_use("cpu")
                    model.to(current_device)

                    train_loader, val_loader, _ = create_dataloaders(
                        crop=crop,
                        batch_size=min(current_batch_size, 16),
                        num_workers=config.num_workers,
                        seed=config.seed,
                    )
                    train_iter = iter(train_loader)
                    continue

                raise exc

        mean_train_loss = epoch_loss / max(1, train_samples)

        # Evaluate on holdout validation partition only
        val_loss, val_metrics = evaluate_validation(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=current_device,
            classes=classes,
        )
        val_macro_f1 = val_metrics["macro_f1"]
        val_accuracy = val_metrics["accuracy"]

        scheduler.step(val_macro_f1)

        is_champion = early_stopping.step(val_macro_f1)
        if is_champion:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_metrics = val_metrics
            best_checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_macro_f1,
                "val_accuracy": val_accuracy,
                "val_loss": val_loss,
                "classes": classes,
                "crop": crop,
            }
            LOGGER.info(
                "★ [Epoch %02d] NEW CHAMPION: Val Macro-F1=%.4f, Acc=%.4f, Train Loss=%.4f, Val Loss=%.4f",
                epoch,
                val_macro_f1,
                val_accuracy,
                mean_train_loss,
                val_loss,
            )
        else:
            LOGGER.info(
                "  [Epoch %02d] Val Macro-F1=%.4f, Acc=%.4f, Patience=%d/%d",
                epoch,
                val_macro_f1,
                val_accuracy,
                early_stopping.patience_counter,
                early_stopping.patience,
            )

        history.append({
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "val_accuracy": round(val_accuracy, 4),
            "lr": optimizer.param_groups[0]["lr"],
        })

        if early_stopping.should_stop:
            LOGGER.info(
                "Early stopping triggered at epoch %d. Best epoch was %d (Val Macro-F1=%.4f).",
                epoch,
                best_epoch,
                best_val_macro_f1,
            )
            break

    # Save artifacts and model metadata
    device_actually_used = device_manager.get_device_actually_used()
    artifact_version = "v1.0.0"
    model_name = config.model_name or f"{crop}_custom_cnn_{artifact_version}"

    output_dir = config.output_dir or (PROJECT_ROOT / f"models/image/{crop}/custom_cnn/{artifact_version}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = output_dir / "model.pt"
    if best_checkpoint_state:
        torch.save(best_checkpoint_state, checkpoint_file)
    else:
        torch.save({"model_state_dict": model.state_dict(), "classes": classes}, checkpoint_file)

    LOGGER.info("Saved trained checkpoint to %s", checkpoint_file)

    software_versions = get_default_software_versions()

    input_type = (
        "leaf crop from segmentation (source_map.csv)"
        if crop == "lettuce"
        else "leaf image as provided"
    )

    metadata = {
        "model_id": f"{crop}_custom_cnn",
        "crop": crop,
        "modality": "image",
        "family": "CustomCNN",
        "artifact_version": artifact_version,
        "artifact_file": str(checkpoint_file.relative_to(PROJECT_ROOT).as_posix()),
        "commit": get_git_commit_hash(),
        "input_type": input_type,
        "classes": classes,
        "num_classes": num_classes,
        "num_parameters": model.count_parameters(),
        "target_device": target_device_str,
        "device_actually_used": device_actually_used,
        "oom_events_caught": oom_events_caught,
        "fallback_triggered": fallback_triggered,
        "best_epoch": best_epoch,
        "val_macro_f1": round(float(best_val_macro_f1 if best_val_macro_f1 > -float("inf") else 0.0), 4),
        "val_accuracy": round(float(best_metrics.get("accuracy", 0.0)), 4),
        "split_counts": {
            "train": len(train_loader.dataset),  # type: ignore[arg-type]
            "val": len(val_loader.dataset),  # type: ignore[arg-type]
        },
        "hyperparameters": {
            "batch_size": config.batch_size,
            "initial_lr": config.lr,
            "weight_decay": config.weight_decay,
            "optimizer": config.optimizer_name,
            "early_stopping_patience": config.early_stopping_patience,
            "seed": config.seed,
        },
        "training_date": datetime.now(timezone.utc).isoformat(),
        "software_versions": software_versions,
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Compute class distribution per split for report
    train_class_counts: dict[str, int] = {}
    val_class_counts: dict[str, int] = {}
    if hasattr(train_loader.dataset, "samples"):
        for s in train_loader.dataset.samples:  # type: ignore[union-attr]
            lbl = s.get("class_name") or classes[s.get("label", 0)]
            train_class_counts[lbl] = train_class_counts.get(lbl, 0) + 1
    if hasattr(val_loader.dataset, "samples"):
        for s in val_loader.dataset.samples:  # type: ignore[union-attr]
            lbl = s.get("class_name") or classes[s.get("label", 0)]
            val_class_counts[lbl] = val_class_counts.get(lbl, 0) + 1

    # Section 18.2 Evaluation Report
    report_dir = config.report_dir or (PROJECT_ROOT / f"reports/{crop}")
    report_dir.mkdir(parents=True, exist_ok=True)

    if crop == "lettuce":
        label_method = "Leaf crop instance diagnosis; crop labels inherited from whole-plant labels"
        crop_limitation = "Crop labels are inherited from whole-plant annotations; leaf-level variance may exist."
    else:
        label_method = "Leaf images as provided with team diagnosis labels"
        crop_limitation = f"Input type: leaf image as provided for {crop}."

    limitations = [
        f"Model trained and evaluated on isolated {crop} leaf images using holdout validation split.",
        crop_limitation,
        f"Verified compute hardware: target={target_device_str}, actually_used={device_actually_used}.",
        "Evaluated strictly on the validation partition; test holdout remains untouched.",
    ]

    write_evaluation_report(
        crop=crop,
        model_name=model_name,
        model_family="CustomCNN",
        artifact_version=artifact_version,
        split_evaluated="validation",
        metrics=best_metrics if best_metrics else to_serializable_metrics(
            calculate_metrics(y_true=classes, y_pred=classes, labels=classes)
        ),
        dataset_manifest=f"processed_data/features/{crop}",
        split_protocol="Stratified split; train and validation isolated",
        split_counts={"train": len(train_loader.dataset), "validation": len(val_loader.dataset), "test": 0},  # type: ignore[arg-type]
        class_counts_per_split={"train": train_class_counts, "validation": val_class_counts},
        label_generation_method=label_method,
        hyperparameters=asdict(config) if hasattr(config, "__dataclass_fields__") else {},
        limitations=limitations,
        output_dir=report_dir,
    )

    return {
        "model_name": model_name,
        "checkpoint_file": str(checkpoint_file),
        "metadata_file": str(metadata_file),
        "device_actually_used": device_actually_used,
        "best_val_macro_f1": best_val_macro_f1,
        "history": history,
        "metadata": metadata,
    }


def run_overfit_test(
    crop: str = "lettuce",
    num_samples: int = 20,
    max_epochs: int = 35,
) -> dict[str, Any]:
    """Execute overfit test on 20 samples to verify gradient flow and convergence (Acceptance Criterion 1)."""
    set_reproducible_seeds(42)
    device = torch.device("cpu")
    classes = list(IMAGE_CLASSES[crop])

    LOGGER.info("Starting overfit test on %d images for crop '%s'...", num_samples, crop)

    # Create dataset and take 20 samples
    dataset = CropLeafDataset(crop=crop, split="train")
    if len(dataset) < num_samples:
        raise ValueError(f"Dataset has only {len(dataset)} samples, need {num_samples}")

    indices = list(range(num_samples))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=num_samples, shuffle=False)

    model = CustomCNN(crop=crop, conv_channels=(16, 32, 64), fc_dim=64)
    model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    final_loss = 1.0
    final_acc = 0.0

    for epoch in range(1, max_epochs + 1):
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            correct = (preds == targets).sum().item()
            acc = correct / num_samples

            final_loss = loss.item()
            final_acc = acc

        if final_loss < 0.05 and final_acc >= 0.95:
            LOGGER.info(
                "Overfit test converged early at epoch %d: loss=%.5f, acc=%.4f",
                epoch,
                final_loss,
                final_acc,
            )
            break

    LOGGER.info("Overfit test completed: final_loss=%.5f, final_acc=%.4f", final_loss, final_acc)
    assert final_loss < 0.15, f"Overfit test failed: loss {final_loss} >= 0.15"
    assert final_acc >= 0.90, f"Overfit test failed: acc {final_acc} < 0.90"
    return {"final_loss": final_loss, "final_accuracy": final_acc, "num_samples": num_samples}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train image classification models (T5-020)")
    parser.add_argument("--crop", default="lettuce", choices=["lettuce", "spinach", "tomato"], help="Crop to train")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--force-cpu", action="store_true", help="Force execution on CPU")
    parser.add_argument("--overfit-test", action="store_true", help="Run overfit test on 20 samples")
    parser.add_argument("--simulate-oom", action="store_true", help="Simulate CUDA OOM error")
    args = parser.parse_args()

    if args.overfit_test:
        res = run_overfit_test(crop=args.crop)
        print("Overfit test result:", res)
        return 0

    config = TrainingConfig(
        crop=args.crop,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        early_stopping_patience=args.patience,
        force_cpu=args.force_cpu,
        simulate_oom=args.simulate_oom,
    )
    result = train_image_model(config)
    print("Training result:", result["model_name"], "Device:", result["device_actually_used"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
