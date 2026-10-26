# ADR 0006: Sensor Validation Protocol and Grouped Cross-Validation

- **Status:** Accepted
- **Date:** 2026-10-26
- **Driver:** Sushil
- **Navigator / Reviewer:** Yashwant
- **Task ID:** T4-002
- **Dependencies:** T2-010, T2-011, T2-012, T4-001
- **References:**
  - `docs/learning/data-leakage-and-grouped-timebased-splits.md` (T4-001)
  - `notebooks/01_leakage_demo.ipynb` (T4-001)
  - `processed_data/sensor/lettuce/manifest.json` (T2-010)
  - `processed_data/sensor/spinach/manifest.json` (T2-011)
  - `processed_data/sensor/tomato/manifest.json` (T2-012)
  - `scripts/make_sensor_validation.py`

---

## 1. Context & Problem Statement

In hydroponic nutrient monitoring, sensor measurements (pH, electrical conductivity, water temperature, ambient air temperature, relative humidity) exhibit severe temporal autocorrelation ($\rho(1) > 0.90$). Consecutive telemetry readings logged every few minutes share physical tank dynamics, diurnal cycles, and unobserved environmental confounders.

Standard machine learning practices that rely on uniform random splitting (`train_test_split` or standard `KFold`) violate the fundamental independent and identically distributed (i.i.d.) assumption. As proven empirically in task **T4-001** (`notebooks/01_leakage_demo.ipynb` and `docs/learning/data-leakage-and-grouped-timebased-splits.md`):

1. **Temporal Interpolation Leakage:** Shuffling rows randomly scatters consecutive readings from the exact same calendar day or multi-hour block across both the training and validation sets. Models memorize local day-specific baseline noise instead of general physical stress boundaries, reporting falsely optimistic performance (e.g., 99.8% Macro-F1) that collapses upon deployment to future days (e.g., dropping to 72.1% Macro-F1).
2. **Duplicate-Oversampling Leakage (Lettuce Flaw):** In naive workflows (such as the legacy lettuce report), minority classes (e.g., `pH_Low` with only 7 real training rows) are oversampled *prior* to cross-validation or hyperparameter search (`GridSearchCV`). When duplicate copies of the same physical reading are distributed across both the training fold and validation fold, the cross-validation score is severely compromised by evaluating on exact training duplicates.

To guarantee scientific rigor, reproducible model selection, and zero data leakage across all ML training tasks (Phase 4), the team must adopt a formal, immutable validation protocol.

---

## 2. Decision: Sensor Validation & Evaluation Protocol

We adopt a three-tier partitioned evaluation protocol across all sensor datasets:
1. **Partition 1: Training Set ($S_{\text{train}}$)** — Used solely for fitting feature scalers, oversamplers, and model weights during model training.
2. **Partition 2: Validation Set ($S_{\text{val}}$)** — Grouped holdout sets created in Phase 2 used strictly for model selection, hyperparameter tuning comparisons, and early stopping.
3. **Partition 3: Test Set ($S_{\text{test}}$)** — Run-once final holdout set evaluated exactly once when the final champion model is locked. It is never accessed during feature selection, model selection, or hyperparameter exploration.

---

## 3. Crop-Specific Group Keys and Partitions

Each crop dataset possesses distinct operational logging physics, temporal structure, and experimental designs. Group keys are defined as follows:

| Crop | Grouping Key Name | Definition / Derivation | Training Partition | Validation Partition | Test Partition |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Lettuce** | `(experiment, date)` | Experiment ID (`EXP.1`, `EXP.2`) and calendar date (`YYYY-MM-DD`). | 41 experiment-days (430 real rows, oversampled to 1,295 rows) | 12 experiment-days (125 real rows, every 4th day of EXP.1/2) | EXP.3 complete run (162 rows, untouched real distribution) |
| **Spinach** | `bench_time_block` | `bench + '\|' + date + '\|' + (hour // 6)` (6-hour contiguous logging block per bench). | 74 whole blocks (958 rows, unresampled) | 14 whole blocks (162 rows, 14.46% of training pool) | 22 whole blocks (280 rows, untouched 80/20 split) |
| **Tomato** | `calendar_month` | `source_timestamp.str[:7]` (`YYYY-MM`). | 89 whole calendar months (29,943 rows, unresampled) | 16 whole calendar months (5,377 rows, 15.22% of training pool) | 27 whole calendar months (8,830 rows, untouched original test months) |

### Key Invariants:
- **Zero Group Overlap:**
  $$G(S_{\text{train}}) \cap G(S_{\text{val}}) = \emptyset, \quad G(S_{\text{train}}) \cap G(S_{\text{test}}) = \emptyset, \quad G(S_{\text{val}}) \cap G(S_{\text{test}}) = \emptyset$$
- **Zero Sample ID Overlap:**
  $$\text{IDs}(S_{\text{train}}) \cap \text{IDs}(S_{\text{val}}) = \emptyset, \quad \text{IDs}(S_{\text{train}}) \cap \text{IDs}(S_{\text{test}}) = \emptyset, \quad \text{IDs}(S_{\text{val}}) \cap \text{IDs}(S_{\text{test}}) = \emptyset$$

---

## 4. Hyperparameter Search & The Oversampling-Inside-Fold Rule

When optimizing hyperparameters inside the training partition (e.g. tree depth, SVM $C$/$\gamma$, regularizations in T4-015):

### 4.1 Grouped Cross-Validation Inside Training Set
- All hyperparameter search (`GridSearchCV`, `RandomizedSearchCV`, or Bayesian optimization) conducted inside $S_{\text{train}}$ **must** use `GroupKFold` or `LeaveOneGroupOut` partitioned on the exact group key specified in Section 3.
- Standard random `KFold` or `StratifiedKFold` is strictly prohibited on raw sensor telemetry.

### 4.2 The Oversampling-Inside-Fold Rule
- **The Defect:** If the training dataset contains oversampled or synthesized rows (e.g., duplicated instances of `pH_Low` or synthetic samples) *prior* to splitting into CV folds, identical samples inevitably land in both the training fold and the validation fold. This yields fraudulent validation accuracy and conceals severe overfitting.
- **The Protocol:** 
  1. Hyperparameter cross-validation **must start from un-oversampled real rows** of the training partition.
  2. Oversampling (e.g. `RandomOverSampler`, SMOTE, or class replication) and feature scaling (`StandardScaler`, `MinMaxScaler`) **must occur strictly inside each cross-validation fold**.
  3. The oversampler is fit solely on fold training rows (`X_fold_train`, `y_fold_train`). The fold validation rows (`X_fold_val`, `y_fold_val`) must remain untouched, real, unresampled observations.
  4. In scikit-learn workflows, this is enforced using `imblearn.pipeline.Pipeline`:
     ```python
     from imblearn.pipeline import Pipeline
     from imblearn.over_sampling import RandomOverSampler
     from sklearn.model_selection import GroupKFold, GridSearchCV
     from sklearn.tree import DecisionTreeClassifier

     pipeline = Pipeline([
         ("scaler", StandardScaler()),
         ("oversampler", RandomOverSampler(random_state=42)),
         ("classifier", DecisionTreeClassifier(random_state=42))
     ])

     gkf = GroupKFold(n_splits=5)
     cv = GridSearchCV(
         estimator=pipeline,
         param_grid=param_grid,
         cv=gkf,
         scoring="f1_macro"
     )
     cv.fit(X_train_real, y_train_real, groups=train_groups)
     ```

---

## 5. Model Selection & Final Test Evaluation

1. **Model Selection & Early Stopping:**
   - Candidate models (Decision Trees, Random Forests, SVMs, Gradient Boosting, PyTorch MLPs) are compared and ranked on the holdout **Validation Set** ($S_{\text{val}}$) using Macro-averaged F1 score (`macro-F1`).
   - Hyperparameter choices, threshold selections, and model architectures are locked based on $S_{\text{val}}$ performance.
2. **Run-Once Test Evaluation:**
   - The test partition ($S_{\text{test}}$) is touched **exactly once** in task **T4-024**.
   - No hyperparameter, feature subset, decision boundary, or model architecture may be altered in response to test set metrics.
   - The final test results are reported verbatim in the model evaluation reports and project documentation.

---

## 6. Consequences

### Positive
- Completely eliminates temporal autocorrelation leakage and duplicate-row oversampling leakage.
- Ensures all reported cross-validation and validation metrics reflect true out-of-distribution temporal extrapolation.
- Establishes a shared, mathematically sound protocol across both collaborators (Sushil and Yashwant).
- Protects the final test set as an untouched benchmark for academic integrity.

### Negative / Trade-offs
- Validation macro-F1 numbers will appear lower than legacy un-grouped reports (e.g., 85%–94% instead of artificial 99%+), which is scientifically honest and expected.
- Pipeline execution is slightly slower because oversampling and scaling are re-computed inside each fold.

---

## 7. Sign-off & Review

Both team members have reviewed and agreed to adhere to this validation protocol throughout Phase 4, Phase 5, and Phase 6:

- **Driver (Author):** Sushil — *Signed off 2026-10-26*
- **Navigator (Reviewer):** Yashwant — *Signed off 2026-10-26*
