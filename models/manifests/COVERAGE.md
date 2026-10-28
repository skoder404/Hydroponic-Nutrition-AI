# Model Registry Coverage Matrix and Untrained Combinations (T5-033)

> **Document Status:** Authoritative Architectural Manifest  
> **Task ID:** T5-033  
> **Adherence:** Canonical Overview §6, §14.2, §15, §18.2, §24.1, §24.2, and TECH_RULES  
> **Last Updated:** 2026-10-29  

---

## 1. Executive Summary

This document establishes the official **Coverage Manifest** of the Hydroponic Nutrition AI model registry. It details all supported, promoted, candidate, and unsupported crop-modality combinations across the system. 

Per Overview §24.2 and TECH_RULES, the Frontend and API must strictly bind to this coverage matrix. The UI **must never display, imply, or offer** models or modalities that do not exist or have not been promoted to active status.

---

## 2. Modality & Crop Coverage Matrix

| Crop Family | Modality | Promoted Active Model | Architecture / Family | Version | Status | Validation Macro-F1 | Test Macro-F1 | Experimental | Manifest Pointer |
|---|---|---|---|---|---|---|---|---|---|
| **Lettuce** | `sensor` | `lettuce_sensor_decision_tree_v1.0.0` | `DecisionTreeClassifier` | `1.0.0` | **Promoted** | `0.7461` | `0.7613` | `false` | `models/manifests/active_lettuce_sensor.json` |
| **Lettuce** | `image` | `lettuce_image_efficientnet_b0_v1.0.0` | `EfficientNetB0` | `1.0.0` | **Promoted** | `0.9119` | `0.8356` | `false` | `models/manifests/active_lettuce_image.json` |
| **Lettuce** | `feature` | `lettuce_feature_random_forest_v1.0.0` | `RandomForest` | `1.0.0` | *Candidate* (Unpromoted) | `0.4930` | *N/A* | `false` | *Artifact in `models/image/lettuce/random_forest/`* |
| **Spinach** | `sensor` | `spinach_sensor_decision_tree_v1.0.0` | `DecisionTreeClassifier` | `1.0.0` | **Promoted** | `1.0000` | `1.0000` | `false` | `models/manifests/active_spinach_sensor.json` |
| **Spinach** | `image` | `spinach_image_efficientnet_b0_v1.0.0` | `EfficientNetB0` | `1.0.0` | **Promoted** | `0.7256` | `0.7212` | `false` | `models/manifests/active_spinach_image.json` |
| **Spinach** | `feature` | `spinach_feature_random_forest_v1.0.0` | `RandomForest` | `1.0.0` | *Candidate* (Unpromoted) | `0.5404` | *N/A* | `false` | *Artifact in `models/image/spinach/random_forest/`* |
| **Tomato** | `sensor` | `tomato_sensor_random_forest_v1.0.0` | `RandomForestClassifier` | `1.0.0` | **Promoted** | `0.9689` | `0.9685` | `true` | `models/manifests/active_tomato_sensor.json` |
| **Tomato** | `image` | `tomato_image_efficientnet_b0_v1.0.0` | `EfficientNetB0` | `1.0.0` | **Promoted** | `0.8748` | `0.8696` | `false` | `models/manifests/active_tomato_image.json` |
| **Tomato** | `feature` | `tomato_feature_random_forest_v1.0.0` | `RandomForest` | `1.0.0` | *Candidate* (Unpromoted) | `0.4346` | *N/A* | `false` | *Artifact in `models/image/tomato/random_forest/`* |
| **Tomato Live** | `sensor` | `tomato_live_sensor_decision_tree_v1.0.0` | `DecisionTreeClassifier` | `1.0.0` | *Candidate* | `0.7712` | *N/A* | `true` | `models/sensor/tomato_live/v1.0.0/metadata.json` |

---

## 3. Explicitly Unsupported & Untrained Combinations

The following combinations are **strictly not supported** in this release. System components (FastAPI endpoints, services, UI components) must enforce these boundaries:

### 3.1. Tomato Live Image & Feature Models (`tomato_live / image`, `tomato_live / feature`)
- **Status:** **NOT TRAINED / UNSUPPORTED**.
- **Rationale:** The physical campus IoT greenhouse testbed provides live streaming Modbus sensor telemetry (EC, pH, Water Temperature, Air Temperature / Humidity) over WebSocket. No live streaming RTSP/IP camera is installed over the canopy.
- **UI Policy:** The `/live` dashboard must strictly render telemetry timeseries, drift gauges, and live sensor diagnosis chips. It must never prompt for or offer live camera stream diagnosis.

### 3.2. Spinach Live & Lettuce Live Telemetry (`spinach / live`, `lettuce / live`)
- **Status:** **NOT TRAINED / UNSUPPORTED**.
- **Rationale:** The Modbus hardware IoT gateway is currently deployed exclusively to the tomato hydroponic testbed. Lettuce and spinach sensor models are batch/manual diagnosis models evaluated against preserved calendar holdouts.
- **UI Policy:** The `/live` dashboard crop selector defaults to `tomato` and disables live streaming selection for lettuce and spinach.

### 3.3. Classical Handcrafted Feature Models as Primary Active Artifacts (`modality="feature"`)
- **Status:** **RETAINED AS RESEARCH CANDIDATES / NOT PROMOTED TO ACTIVE**.
- **Rationale:** Classical feature extraction (colour vegetation indices ExG/ExR, HSV color ratio masks, GLCM texture, morphological shape features) followed by RandomForest and SVM was trained across all three crops in Phase 5 (T5-014, T5-015, T5-017, T5-018). In head-to-head fair validation comparisons (T5-028, T5-029, T5-030), `EfficientNetB0` outperformed RandomForest by:
  - **Lettuce:** $+41.88\%$ Macro-F1 (`0.9119` vs `0.4930`)
  - **Spinach:** $+18.52\%$ Macro-F1 (`0.7256` vs `0.5404`)
  - **Tomato:** $+44.02\%$ Macro-F1 (`0.8748` vs `0.4346`)
- **Deployment Policy:** Active image pointers in `models/manifests/` resolve to `EfficientNetB0`. Classical feature models remain preserved in versioned experiment directories (`models/image/<crop>/random_forest/` and `models/image/<crop>/svm/`) for ablation reproducibility and edge feature benchmarks.

### 3.4. Direct Whole-Canopy Unsegmented Classification Models
- **Status:** **ARCHITECTURALLY PROHIBITED**.
- **Rationale:** Direct full-frame image classification on cluttered hydroponic greenhouse photos suffers from severe background confounders (pumps, PVC pipes, nutrient tanks, lighting glare). Overview §24.1 mandates `input_unit: "leaf"`.
- **Pipeline Enforcement:** The image pipeline (`ImageService`, T6-006) strictly enforces leaf extraction:
  - `input_mode="leaf"`: validates single-leaf isolation and extracts the leaf mask.
  - `input_mode="plant"`: runs SAM 2.1 (`sam2.1_hiera_tiny.pt`) to segment individual leaves.
  - No direct prediction is ever emitted without an isolated leaf crop.

### 3.5. Alternative Heavy Deep Learning Architectures (ResNet-50, Vision Transformer, Swin, YOLO)
- **Status:** **OUT OF SCOPE / NOT TRAINED**.
- **Rationale:** `EfficientNetB0` achieved superior validation and test macro-F1 while maintaining a minimal parameter footprint ($4.01\text{M}$ to $4.02\text{M}$ parameters) and CPU inference latency under $50\text{ms}$ per leaf, satisfying hardware budget constraints (HP Victus RTX 3050 and CPU campus fallback policy per Overview §25).

---

## 4. Model Metadata Governance & Verification

All promoted models adhere to the canonical schema defined in `ml.registry.metadata_schema.ModelMetadata`:
- **Path Guard:** Enforced via `backend.app.core.paths.validate_model_path` ensuring zero directory traversal.
- **Audit Logging:** Final test evaluations logged immutably in `reports/final_eval_log.jsonl`.
- **Schema Compatibility:** Checked at startup by `backend.app.services.model_service.ModelRegistryService`.
- **Promotion / Rollback:** Executed strictly via CLI (`scripts/promote_model.py`) maintaining audit trails in `models/manifests/active_<crop>_<modality>.json`.
