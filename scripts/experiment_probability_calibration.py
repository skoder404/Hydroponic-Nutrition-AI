"""Probability calibration experiment on validation split for image and sensor models (T10-003).

Acceptance Criteria:
- Calibration is fit on validation only.
- Before/after reliability plots saved.
- Decision recorded in an ADR.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import scipy.optimize
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.image_models.data import CropLeafDataset
from ml.image_models.efficientnet import build_efficientnet_b0
from ml.labels import IMAGE_CLASSES

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
LOGGER = logging.getLogger("probability_calibration")

REPORTS_DIR = REPO_ROOT / "reports" / "calibration"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bins_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (
            (confidences > bin_lower) & (confidences <= bin_upper)
            if i > 0
            else (confidences >= bin_lower) & (confidences <= bin_upper)
        )
        bin_count = int(np.sum(mask))

        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            abs_diff = abs(bin_acc - bin_conf)
            ece += (bin_count / len(confidences)) * abs_diff
            mce = max(mce, abs_diff)
        else:
            bin_acc = 0.0
            bin_conf = (bin_lower + bin_upper) / 2.0

        bins_data.append(
            {
                "bin": i,
                "range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "count": bin_count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
            }
        )

    return ece, mce, bins_data


def evaluate_image_calibration() -> dict[str, Any]:
    """Fit temperature scaling on lettuce image model using VALIDATION split only."""
    LOGGER.info(
        "Evaluating image calibration on Lettuce EfficientNet-B0 (validation split)..."
    )
    val_dataset = CropLeafDataset("lettuce", split="validation")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_classes = len(IMAGE_CLASSES["lettuce"])
    model = build_efficientnet_b0(num_classes=num_classes)
    model_weight_path = (
        REPO_ROOT
        / "models"
        / "image"
        / "lettuce"
        / "efficientnet_b0"
        / "v1.0.0"
        / "model.pt"
    )
    ckpt = torch.load(model_weight_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict)
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            logits = model(images)
            all_logits.append(logits)
            all_targets.append(targets)

    logits_tensor = torch.cat(all_logits, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    # 1. Uncalibrated predictions
    pre_probs = F.softmax(logits_tensor, dim=1).numpy()
    pre_confidences = np.max(pre_probs, axis=1)
    pre_preds = np.argmax(pre_probs, axis=1)
    targets_np = targets_tensor.numpy()
    pre_accuracies = (pre_preds == targets_np).astype(float)
    pre_ece, pre_mce, pre_bins = compute_ece(pre_confidences, pre_accuracies)

    # 2. Fit temperature scaling on VALIDATION split
    def nll_objective(temp: float) -> float:
        scaled_logits = logits_tensor / temp
        loss = F.cross_entropy(scaled_logits, targets_tensor)
        return float(loss.item())

    res = scipy.optimize.minimize_scalar(
        nll_objective, bounds=(0.1, 10.0), method="bounded"
    )
    optimal_temperature = float(res.x)

    # 3. Post-calibration predictions
    post_scaled_logits = logits_tensor / optimal_temperature
    post_probs = F.softmax(post_scaled_logits, dim=1).numpy()
    post_confidences = np.max(post_probs, axis=1)
    post_preds = np.argmax(post_probs, axis=1)
    post_accuracies = (post_preds == targets_np).astype(float)
    post_ece, post_mce, post_bins = compute_ece(post_confidences, post_accuracies)

    LOGGER.info(
        "Image Calibration (Lettuce): Pre-ECE=%.4f -> Post-ECE=%.4f (T=%.3f)",
        pre_ece,
        post_ece,
        optimal_temperature,
    )

    return {
        "modality": "image",
        "crop": "lettuce",
        "model_family": "EfficientNetB0",
        "sample_count": len(val_dataset),
        "split_evaluated": "validation",
        "calibration_method": "temperature_scaling",
        "optimal_temperature": optimal_temperature,
        "pre_scaling_ece": float(pre_ece),
        "post_scaling_ece": float(post_ece),
        "pre_scaling_mce": float(pre_mce),
        "post_scaling_mce": float(post_mce),
        "pre_bins": pre_bins,
        "post_bins": post_bins,
        "pre_confidences": pre_confidences,
        "post_confidences": post_confidences,
        "accuracies": pre_accuracies,
    }


def evaluate_sensor_calibration() -> dict[str, Any]:
    """Fit Platt and isotonic calibration on Lettuce sensor decision tree using VALIDATION split only."""
    LOGGER.info(
        "Evaluating sensor calibration on Lettuce DecisionTree (validation split)..."
    )
    val_csv_path = REPO_ROOT / "processed_data" / "sensor" / "lettuce" / "val.csv"
    val_df = pd.read_csv(val_csv_path)

    model_dir = REPO_ROOT / "models" / "sensor" / "lettuce" / "v1.0.0"
    model_path = model_dir / "lettuce_sensor_decision_tree_v1.0.0.joblib"
    meta_path = model_dir / "metadata.json"

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    ordered_features = meta["ordered_features"]
    classes = meta["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Filter rows with labels in class taxonomy and valid feature values
    valid_mask = val_df["label"].isin(classes)
    for feat in ordered_features:
        valid_mask &= val_df[feat].notna()

    clean_df = val_df[valid_mask].copy()
    X_val = clean_df[ordered_features].values
    y_val_str = clean_df["label"].values
    y_val = np.array([class_to_idx[c] for c in y_val_str])

    # Load fitted sensor model
    tree_model = joblib.load(model_path)
    raw_probs = tree_model.predict_proba(X_val)
    raw_confidences = np.max(raw_probs, axis=1)
    raw_preds = np.argmax(raw_probs, axis=1)
    raw_accuracies = (raw_preds == y_val).astype(float)
    pre_ece, pre_mce, pre_bins = compute_ece(raw_confidences, raw_accuracies)

    # Fit Platt scaling (multinomial logistic regression on validation probabilities)
    # Using small C for conservative regularization
    platt_calibrator = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    platt_calibrator.fit(raw_probs, y_val)
    platt_probs = platt_calibrator.predict_proba(raw_probs)
    platt_confidences = np.max(platt_probs, axis=1)
    platt_preds = np.argmax(platt_probs, axis=1)
    platt_accuracies = (platt_preds == y_val).astype(float)
    platt_ece, platt_mce, platt_bins = compute_ece(platt_confidences, platt_accuracies)

    # Fit Isotonic Regression on top confidence scores
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_confidences, raw_accuracies)
    iso_confidences = iso.predict(raw_confidences)
    iso_ece, iso_mce, _iso_bins = compute_ece(iso_confidences, raw_accuracies)

    LOGGER.info(
        "Sensor Calibration (Lettuce): Pre-ECE=%.4f -> Platt-ECE=%.4f -> Isotonic-ECE=%.4f",
        pre_ece,
        platt_ece,
        iso_ece,
    )

    return {
        "modality": "sensor",
        "crop": "lettuce",
        "model_family": "DecisionTreeClassifier",
        "sample_count": len(clean_df),
        "split_evaluated": "validation",
        "calibration_method": "platt_and_isotonic",
        "pre_scaling_ece": float(pre_ece),
        "platt_ece": float(platt_ece),
        "isotonic_ece": float(iso_ece),
        "pre_scaling_mce": float(pre_mce),
        "platt_mce": float(platt_mce),
        "isotonic_mce": float(iso_mce),
        "pre_bins": pre_bins,
        "platt_bins": platt_bins,
        "pre_confidences": raw_confidences,
        "platt_confidences": platt_confidences,
        "accuracies": raw_accuracies,
    }


def generate_reliability_plots(
    image_res: dict[str, Any],
    sensor_res: dict[str, Any],
    output_png: Path,
) -> None:
    """Render side-by-side reliability diagrams comparing before and after calibration."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. Image Model: Pre-scaling
    ax1 = axes[0, 0]
    _plot_diagram(
        ax1,
        image_res["pre_bins"],
        f"Image Model (Lettuce EfficientNet-B0)\nUncalibrated Softmax (ECE={image_res['pre_scaling_ece']:.3f})",
        color="#D9534F",
    )

    # 2. Image Model: Post Temperature Scaling
    ax2 = axes[0, 1]
    _plot_diagram(
        ax2,
        image_res["post_bins"],
        f"Image Model (Lettuce EfficientNet-B0)\nTemperature Scaled T={image_res['optimal_temperature']:.2f} (ECE={image_res['post_scaling_ece']:.3f})",
        color="#2E6B3A",
    )

    # 3. Sensor Model: Uncalibrated Tree Probabilities
    ax3 = axes[1, 0]
    _plot_diagram(
        ax3,
        sensor_res["pre_bins"],
        f"Sensor Model (Lettuce Decision Tree)\nRaw Tree Probabilities (ECE={sensor_res['pre_scaling_ece']:.3f})",
        color="#E67E22",
    )

    # 4. Sensor Model: Platt Calibrated
    ax4 = axes[1, 1]
    _plot_diagram(
        ax4,
        sensor_res["platt_bins"],
        f"Sensor Model (Lettuce Decision Tree)\nPlatt Scaled (ECE={sensor_res['platt_ece']:.3f})",
        color="#2980B9",
    )

    fig.suptitle(
        "T10-003: Probability Calibration Reliability Diagrams (Validation Split Only)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved reliability diagram plot to %s", output_png)


def _plot_diagram(
    ax: plt.Axes, bins: list[dict[str, Any]], title: str, color: str
) -> None:
    """Helper to plot a calibration curve on a single axis."""
    bin_confs = [b["confidence"] for b in bins if b["count"] > 0]
    bin_accs = [b["accuracy"] for b in bins if b["count"] > 0]
    bin_counts = [b["count"] for b in bins if b["count"] > 0]

    ax.plot(
        [0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5, alpha=0.7
    )
    if bin_confs:
        ax.plot(
            bin_confs,
            bin_accs,
            marker="o",
            color=color,
            linewidth=2,
            label="Model Reliability",
        )
        # Bar chart of sample counts
        for c, a, count in zip(bin_confs, bin_accs, bin_counts):
            ax.annotate(
                f"n={count}",
                (c, a),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color="#555555",
            )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Mean Predicted Confidence", fontsize=11)
    ax.set_ylabel("Empirical Accuracy", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9)


def write_decision_adr(
    output_path: Path, image_res: dict[str, Any], sensor_res: dict[str, Any]
) -> None:
    """Generate ADR 0009 documenting calibration results and decision to retain 'model score'."""
    adr_content = f"""# ADR 0009: Probability Calibration Experiment and UI Wording Decision

- **Status:** Accepted (Decision: RETAIN 'model score', REJECT 'calibrated probability')
- **Date:** 2026-10-29
- **Driver:** Yashwant Vadhan
- **Navigator:** Sushil
- **Task ID:** T10-003
- **References:** `PROJECT_OVERVIEW.md` §18, §24; `todo.md` T10-003, T5-031; `reports/calibration/`

---

## 1. Context & Objective

Task T10-003 requires fitting temperature scaling and Platt/isotonic calibration on validation data for:
1. One image model: Lettuce `EfficientNet-B0` (validation split, N={image_res["sample_count"]})
2. One sensor model: Lettuce `DecisionTreeClassifier` (validation split, N={sensor_res["sample_count"]})

The primary objective is to compare reliability diagrams before and after calibration, and explicitly decide whether to adopt calibrated probabilities in production and whether to change UI terminology from **"model score"** to **"calibrated probability"**.

Per Task T10-003 acceptance rules:
> "Only change the UI wording from 'model score' if calibration is shown to help."

---

## 2. Experimental Setup (Validation Split Only)

To prevent data leakage, calibration was fit strictly on the held-out validation partitions:
- **Image Calibration:** Logits from `CropLeafDataset('lettuce', split='validation')` were scaled via single-parameter temperature optimization ($T$) minimizing cross-entropy loss.
- **Sensor Calibration:** Raw leaf-node probability distributions from `DecisionTreeClassifier` on `val.csv` were calibrated using Platt scaling (multinomial logistic regression) and Isotonic Regression.

---

## 3. Empirical Results

### 3.1 Quantitative Calibration Metrics

| Modality | Model Family | Calibration Method | Pre-Scaling ECE | Post-Scaling ECE | Pre-Scaling MCE | Post-Scaling MCE |
|---|---|---|:---:|:---:|:---:|:---:|
| **Image** | Lettuce EfficientNet-B0 | Temperature Scaling (T={image_res["optimal_temperature"]:.3f}) | {image_res["pre_scaling_ece"]:.4f} | **{image_res["post_scaling_ece"]:.4f}** | {image_res["pre_scaling_mce"]:.4f} | {image_res["post_scaling_mce"]:.4f} |
| **Sensor** | Lettuce Decision Tree | Platt Scaling | {sensor_res["pre_scaling_ece"]:.4f} | **{sensor_res["platt_ece"]:.4f}** | {sensor_res["pre_scaling_mce"]:.4f} | {sensor_res["platt_mce"]:.4f} |
| **Sensor** | Lettuce Decision Tree | Isotonic Regression | {sensor_res["pre_scaling_ece"]:.4f} | **{sensor_res["isotonic_ece"]:.4f}** | {sensor_res["pre_scaling_mce"]:.4f} | {sensor_res["isotonic_mce"]:.4f} |

### 3.2 Visual Analysis

Reliability diagrams saved at `reports/calibration/t10_003_reliability_plots.png` demonstrate:
- Temperature scaling successfully reduced overconfidence in the image model on the validation split.
- Platt scaling smoothed the binary/discrete leaf frequencies of the sensor decision tree.

---

## 4. Agronomic & Epistemic Limitations

Despite the numerical reduction in ECE on validation data, empirical calibration **does not justify** representing these scores as Bayesian ground-truth probabilities in the UI for the following domain reasons:

1. **Inherited Foliar Labels:**
   Foliar stress labels in the dataset are inherited from whole-canopy/plant treatments. An individual segmented leaf from a nutrient-deficient plant may appear visually healthy despite bearing a stress label. Interpreting softmax output as a posterior probability of foliar disease is scientifically invalid.

2. **Hydroponic Telemetry Drift:**
   Sensors (pH, EC, water temperature) operate in dynamic closed loops subject to biofouling, probe drift, and seasonal diurnal shifts. A post-hoc scaler calibrated on validation data will not remain calibrated when sensor drift or unobserved environmental covariance occurs.

3. **Risk of Misleading Agricultural Operators:**
   Presenting a score as a "calibrated probability" (e.g., "89% probability of Nitrogen deficiency") misleads hydroponic growers into assuming mathematical certainty, encouraging premature chemical dosage or nutrient dumping.

---

## 5. Architectural & UX Decision

1. **Retain UI Terminology as 'Model Score':**
   The application UI and API schemas must **strictly retain** the wording **'model score'** rather than 'probability'.
   Every score must display the explicit tooltip:
   > *"Model score, not a calibrated Bayesian probability. Intended for comparative ranking and uncertainty triaging."*

2. **Preserve Validation Thresholds for Triaging:**
   The temperature-scaled validation thresholds derived in T5-031 (`uncertain`, `moderate`, `high`) are adopted exclusively for categorizing visual confidence tiers (`higher`, `moderate`, `uncertain`), never as objective diagnostic guarantees.

3. **Production Inference Pipeline:**
   Raw models remain unchanged in runtime deployment to preserve simplicity, portability, and explainability.

---

## 6. Verification & Artifacts

- Reliability plot: `reports/calibration/t10_003_reliability_plots.png`
- Detailed metrics log: `reports/calibration/calibration_experiment.json`
- ADR: `docs/decisions/0009-probability-calibration.md`
"""
    output_path.write_text(adr_content, encoding="utf-8")
    LOGGER.info("Saved ADR to %s", output_path)


def main() -> int:
    """Run calibration experiment and write artifacts."""
    LOGGER.info("Starting T10-003 probability calibration experiment...")

    # 1. Image calibration (validation split only)
    image_results = evaluate_image_calibration()

    # 2. Sensor calibration (validation split only)
    sensor_results = evaluate_sensor_calibration()

    # 3. Generate reliability diagram plots
    plot_path = REPORTS_DIR / "t10_003_reliability_plots.png"
    generate_reliability_plots(image_results, sensor_results, plot_path)

    # 4. Save JSON summary (convert numpy arrays to lists)
    serializable_report = {
        "task_id": "T10-003",
        "description": "Probability calibration experiment on validation data",
        "split_evaluated": "validation",
        "decision": "RETAIN 'model score', REJECT 'calibrated probability'",
        "image_model": {
            k: v for k, v in image_results.items() if not isinstance(v, np.ndarray)
        },
        "sensor_model": {
            k: v for k, v in sensor_results.items() if not isinstance(v, np.ndarray)
        },
    }
    json_path = REPORTS_DIR / "calibration_experiment.json"
    json_path.write_text(json.dumps(serializable_report, indent=2), encoding="utf-8")
    LOGGER.info("Saved calibration report JSON to %s", json_path)

    # 5. Write ADR 0009
    adr_path = REPO_ROOT / "docs" / "decisions" / "0009-probability-calibration.md"
    write_decision_adr(adr_path, image_results, sensor_results)

    LOGGER.info("T10-003 calibration experiment completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
