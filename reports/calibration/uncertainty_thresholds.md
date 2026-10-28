# Uncertainty Thresholds & Reliability Calibration Report

> **Task Reference:** T5-031  
> **Evaluation Partition:** `validation` only (fit on validation; test set strictly preserved)  
> **Target Models:** Promoted `EfficientNetB0` classifiers for Lettuce, Spinach, and Tomato  
> **Methodology:** Validation confidence percentile partitioning & Platt/Guo Temperature Scaling  

---

## 1. Executive Summary & Score Semantics

In accordance with TECH_RULES Rule 13, raw softmax outputs do not represent calibrated objective probabilities. Plant-level stress labels were assigned down to segmented leaf crops during harvesting and dataset curation, introducing inherent foliar label ambiguity (e.g. a visually unaffected leaf on a nitrogen-stressed plant).

### Canonical Score Semantics Statement
> *"Raw softmax confidence is an uncalibrated heuristic score and must not be interpreted as an objective Bayesian probability of ground truth. In particular, whole-canopy plant stress labels are inherited down to individual leaves, so an individual leaf with high visual health might still carry an inherited stress label. In production, scores at or above the high threshold represent strong alignment with validation distribution; scores below the moderate threshold trigger the 'uncertain / manual inspection recommended' status chip."*

---

## 2. Derived Uncertainty Thresholds Table

Thresholds are established strictly from empirical accuracy and quantile curves on the isolated **validation partition**:

| Crop | Promoted Model | Val Samples | High Confidence (>=) | Moderate Band | Uncertain Band (<) | Fitted Temp ($T$) | Pre-ECE | Post-ECE |
|---|---|---:|---:|:---:|---:|---:|---:|---:|
| **Lettuce** | `lettuce_efficientnet_b0_v1.0.0` | 128 | **0.788** | 0.487 - 0.788 | **< 0.487** | 1.46 | 0.1790 | **0.1109** |
| **Spinach** | `spinach_efficientnet_b0_v1.0.0` | 2,745 | **0.7** | 0.476 - 0.7 | **< 0.476** | 0.901 | 0.0288 | **0.0286** |
| **Tomato** | `tomato_efficientnet_b0_v1.0.0` | 5,034 | **0.881** | 0.543 - 0.881 | **< 0.543** | 0.706 | 0.0695 | **0.0106** |

---

## 3. Reliability Calibration & Temperature Scaling

1. **Temperature Optimization Formulation**:
   $$p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
   Where $T > 0$ is optimized strictly on validation cross-entropy loss.
2. **Expected Calibration Error (ECE)**:
   - For all three models, temperature scaling reduced Expected Calibration Error across the 10 empirical confidence bins.
   - Preserves argmax class rankings (Macro-F1 is mathematically invariant under monotonic scaling $T > 0$) while significantly aligning confidence magnitude with empirical true precision.

---

## 4. Generated Artifacts

- **Combined Reliability Plot**: `reports/calibration/reliability_plots_combined.png`
- **Lettuce Plot**: `reports/calibration/reliability_plot_lettuce.png`
- **Spinach Plot**: `reports/calibration/reliability_plot_spinach.png`
- **Tomato Plot**: `reports/calibration/reliability_plot_tomato.png`
- **Metadata Updates**: Synced to each model's `metadata.json` (`uncertainty_thresholds`, `score_semantics`, `temperature_scaling`).
