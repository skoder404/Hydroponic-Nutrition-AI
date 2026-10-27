# Sensor Model Promotion Rationale (T4-025)

## Executive Summary
In accordance with Canonical Overview Sections 15, 18.2, and 24.2, one sensor classification model has been explicitly promoted for each of the three supported crop families (`lettuce`, `spinach`, `tomato`). Promotion was performed strictly via the CLI tool `scripts/promote_model.py`.

## Promoted Models

| Crop | Selected Model | Family | Version | Validation Macro-F1 | Test Macro-F1 | Experimental | Manifest Pointer |
|---|---|---|---|---|---|---|---|
| **Lettuce** | `lettuce_sensor_decision_tree_v1.0.0` | `DecisionTreeClassifier` | `1.0.0` | `0.7461` | `0.7613` | `false` | `models/manifests/active_lettuce_sensor.json` |
| **Spinach** | `spinach_sensor_decision_tree_v1.0.0` | `DecisionTreeClassifier` | `1.0.0` | `1.0000` | `1.0000` | `false` | `models/manifests/active_spinach_sensor.json` |
| **Tomato** | `tomato_sensor_random_forest_v1.0.0` | `RandomForestClassifier` | `1.0.0` | `0.9689` | `0.9685` | `true` | `models/manifests/active_tomato_sensor.json` |

## Crop-Specific Rationale

### 1. Lettuce (`lettuce_sensor_decision_tree_v1.0.0`)
- **Metric Performance:** Validation macro-F1 tied at 0.7461 with `RandomForestClassifier`.
- **Tie-Break Hierarchy:** Per Section 18.2 step 4, the Decision Tree was chosen over Random Forest due to model parsimony and complete white-box explainability.
- **Run-Once Test Evaluation:** Held-out test split of 162 samples confirmed generalization: Macro-F1: `0.7613`, Accuracy: `0.9877`.
- **Status:** Promoted (`experimental: false`).

### 2. Spinach (`spinach_sensor_decision_tree_v1.0.0`)
- **Metric Performance:** Validation macro-F1 reached 1.0000 across all classes.
- **Parsimony:** Simple interpretable rules cleanly separate nutrient deficiency classes.
- **Run-Once Test Evaluation:** Evaluated on 280 held-out test rows yielding Macro-F1: `1.0000`, Accuracy: `1.0000`.
- **Status:** Promoted (`experimental: false`).

### 3. Tomato (`tomato_sensor_random_forest_v1.0.0`)
- **Metric Performance:** Outperformed all other architectures (DT, SVM, LogReg, KNN) with validation Macro-F1 of `0.9689`.
- **Run-Once Test Evaluation:** Evaluated on 8,830 held-out test rows yielding Macro-F1: `0.9685`, Accuracy: `0.9689`.
- **Experimental Flag:** Enforced as `experimental: true` per Overview Section 15 and ADR 0005 due to synthetic/rule-derived ground truth.
- **Status:** Promoted (`experimental: true`).

## Verification & Rollback Testing
- Verified via `scripts/verify_model_registry.py`.
- Rollback functionality tested with dummy model artifact v0.9.0, successfully confirming non-destructive restore of active pointers.
