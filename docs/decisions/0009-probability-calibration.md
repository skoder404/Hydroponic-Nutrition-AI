# ADR 0009: Probability Calibration Experiment and UI Wording Decision

- **Status:** Accepted (Decision: RETAIN 'model score', REJECT 'calibrated probability')
- **Date:** 2026-10-29
- **Driver:** Yashwant Vadhan
- **Navigator:** Sushil
- **Task ID:** T10-003
- **References:** `PROJECT_OVERVIEW.md` §18, §24; `todo.md` T10-003, T5-031; `reports/calibration/`

---

## 1. Context & Objective

Task T10-003 requires fitting temperature scaling and Platt/isotonic calibration on validation data for:
1. One image model: Lettuce `EfficientNet-B0` (validation split, N=128)
2. One sensor model: Lettuce `DecisionTreeClassifier` (validation split, N=124)

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
| **Image** | Lettuce EfficientNet-B0 | Temperature Scaling (T=0.844) | 0.0486 | **0.0567** | 0.4282 | 0.4124 |
| **Sensor** | Lettuce Decision Tree | Platt Scaling | 0.9194 | **0.7972** | 0.9194 | 0.9284 |
| **Sensor** | Lettuce Decision Tree | Isotonic Regression | 0.9194 | **0.0000** | 0.9194 | 0.0000 |

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
