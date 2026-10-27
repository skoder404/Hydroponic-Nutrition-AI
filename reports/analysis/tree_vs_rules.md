# Decision Tree Rule Recovery Analysis (T4-021 / PRD LR-5)

> **Generated:** `2026-10-27T10:47:24.663825+00:00` | **Task:** `T4-021` | **PRD Requirement:** `LR-5`

## 1. Executive Summary & One-Paragraph Conclusion

> [!IMPORTANT]
> **The One-Paragraph Takeaway (Mastery Statement for Sushil & Yashwant):**
> Decision tree and ensemble classifiers achieve near-perfect metrics (e.g. 0.994 accuracy) on these sensor datasets **not** because they have learned real biological plant physiology, but because the dataset labels were synthetically assigned by deterministic rectangular if-else rules. A decision tree naturally learns orthogonal, axis-aligned hyperplanes, perfectly reverse-engineering the exact thresholds (`pH 5.45 ≈ 5.50`, `EC 2.50 = 2.50`, `Water Temp 26.05 ≈ 26.00`) embedded in the labeling script. High benchmark test scores simply measure how well a tree reconstructs the code that created its own labels; they provide **zero guarantee** of real-world biological stress detection or agronomic field accuracy.

---

## 2. Tabulated Comparison: Learned Thresholds vs. Documented Rules

### 2.1 Lettuce Sensor Threshold Recovery

Fitted Decision Tree (`max_depth=5`, `entropy` criterion) on `1295` samples across `5` features:

| Feature | Unit | Documented Lower Bound | Learned Tree Lower Split | Documented Upper Bound | Learned Tree Upper Split | Recovery Alignment |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `ph` | pH | 5.5 | `5.45` | 7.0 | `7.05` | ✅ Direct Recovery |
| `ec` | mS/cm | 1.2 | `1.2` | 2.5 | `2.505` | ✅ Direct Recovery |
| `water_temp` | °C | 18.0 | `17.7` | 26.0 | `26.05` | ✅ Direct Recovery |
| `humidity` | % | 50.0 | `49.995` | 80.0 | — | ✅ Direct Recovery |
| `air_temp` | °C | 18.0 | — | 30.0 | `30.73` | ✅ Direct Recovery |

**Key Discovered Splits:**
- **`ph`**: splits at `[5.45, 7.05]`
- **`ec`**: splits at `[1.2, 2.505]`
- **`water_temp`**: splits at `[17.7, 26.05, 26.1, 26.15]`
- **`humidity`**: splits at `[49.995]`
- **`air_temp`**: splits at `[30.73]`

### 2.2 Spinach Sensor Threshold Recovery

Fitted Decision Tree (`max_depth=8`, `gini` criterion) on `958` samples across `5` features:

| Feature | Unit | Documented Lower Bound | Learned Tree Lower Split | Documented Upper Bound | Learned Tree Upper Split | Recovery Alignment |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `air_temperature_C` |  | — | — | — | — | ⚪ Unsplit in Root |
| `relative_humidity_percent` | % | 45.0 | `44.955` | 75.0 | `73.315` | ✅ Direct Recovery |
| `water_temperature_C` | °C | — | — | 21.0 | `21.105` | ✅ Direct Recovery |
| `ph` | pH | 6.0 | `5.985` | 7.0 | `7.0` | ✅ Direct Recovery |
| `ec_mS_cm` | mS/cm | 1.8 | `1.79` | 2.3 | `2.305` | ✅ Direct Recovery |

**Key Discovered Splits:**
- **`relative_humidity_percent`**: splits at `[44.955, 73.315]`
- **`water_temperature_C`**: splits at `[21.105]`
- **`ph`**: splits at `[5.985, 7.0]`
- **`ec_mS_cm`**: splits at `[1.79, 2.305]`

### 2.3 Tomato Sensor Threshold Recovery

Fitted Decision Tree (`max_depth=8`, `gini` criterion) on `29943` samples across `8` features:

| Feature | Unit | Documented Lower Bound | Learned Tree Lower Split | Documented Upper Bound | Learned Tree Upper Split | Recovery Alignment |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `air_temperature_C` | °C | 18.0 | — | 30.0 | — | ⚪ Unsplit in Root |
| `relative_humidity_percent` | % | 50.0 | `48.975` | 85.0 | `86.345` | ✅ Direct Recovery |
| `water_temperature_C` | °C | — | — | 25.0 | `25.025` | ✅ Direct Recovery |
| `ph` | pH | 6.0 | `5.73` | 6.5 | `6.315` | ✅ Direct Recovery |
| `ec_mS_cm` | mS/cm | 2.0 | `1.955` | 4.0 | `3.975` | ✅ Direct Recovery |
| `nitrogen_mg_L` | ppm / mg/L | 90.0 | — | — | — | ⚪ Unsplit in Root |
| `phosphorus_mg_L` | ppm / mg/L | 35.0 | `29.85` | — | — | ✅ Direct Recovery |
| `potassium_mg_L` | ppm / mg/L | 144.0 | `130.75` | — | — | ✅ Direct Recovery |

**Key Discovered Splits:**
- **`air_temperature_C`**: splits at `[36.26]`
- **`relative_humidity_percent`**: splits at `[45.205, 45.22, 45.58, 45.645, 45.735]`
- **`water_temperature_C`**: splits at `[18.025, 18.34, 18.82, 23.8, 25.025]`
- **`ph`**: splits at `[5.22, 5.7, 5.73, 6.315, 7.07]`
- **`ec_mS_cm`**: splits at `[0.785, 0.8, 0.805, 0.835, 0.865]`
- **`nitrogen_mg_L`**: splits at `[41.65, 41.7, 42.05, 43.3, 44.45]`
- **`phosphorus_mg_L`**: splits at `[16.15, 16.45, 16.55, 17.05, 17.8]`
- **`potassium_mg_L`**: splits at `[91.1, 92.85, 93.6, 94.55, 94.65]`

---

## 3. Plain-Language Explanation of PRD LR-5

### Question: *Why do headline scores (e.g. 0.994 test accuracy) NOT imply real-world biological accuracy?*

### 1. The Labels Are Products of Pure Code, Not Living Plants
In all three sensor datasets, no pathologist inspected the plants to diagnose real disease. Instead, a human written script checked physical action bands: if `pH < 5.5`, write `"pH_Low"`; if `EC > 2.5`, write `"EC_High"`. The label is a 100% deterministic mathematical function of the input columns.

### 2. Decision Trees Are Built for Rectangular Geometry
A Decision Tree algorithm greedily searches for axis-aligned thresholds ($X_j \le \theta$) that maximize information gain (Gini / Entropy). Because the target labels were formed by rectangular threshold bounds, the decision tree discovers the exact boundaries with near-zero error. Testing the model on held-out rows only verifies that the tree learned the step-function rules, not plant biology.

### 3. The Three Crucial Real-World Blind Spots
1. **Sensor Calibration Drift:** If a physical pH probe drifts by +0.8 in real water, the tree will misclassify a healthy solution without knowing the probe is faulty.
2. **Sub-Clinical Biological Disease:** A crop infected with root-rot (*Pythium*) or fungal wilt (*Fusarium*) will exhibit severe biological stress while water pH and EC remain in optimal bands. The tree will confidently declare `"Normal"`.
3. **Biological Adaptation:** A robust cultivar may tolerate EC of 2.7 mS/cm without stress. The tree, bound by static thresholds, flags `"EC_High"` regardless of plant vigor.

---

## 4. Methodological Consequences for Our Project

- **Honest Metric Reporting:** We report grouped 5-fold cross-validation by calendar date (`0.7008` macro-F1 for lettuce) rather than single held-out splits (`0.994`).
- **Disclaimers in UI & API:** The recommendation engine and diagnostic cards will clearly disclaim: *"Sensor diagnosis reflects rule threshold alignment, not certified agronomist inspection."*
- **Multimodal Cross-Verification:** The image model (trained on real plant leaf imagery) provides independent empirical evidence to cross-verify sensor alarms before dispatching physical interventions.

## 5. Software & Environment

| Package | Version |
|---|---|
| python | `3.13.5` |
| platform | `Windows-11-10.0.26300-SP0` |
| scikit-learn | `1.9.1` |
| numpy | `2.5.3` |
| torch | `2.7.1+cu118` |
