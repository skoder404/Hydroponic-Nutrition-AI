# Model Comparison & Selection Report: Lettuce Leaf Image Diagnosis

> **Task Reference:** T5-028  
> **Crop:** `lettuce`  
> **Modality:** `image` (leaf-level crop input)  
> **Evaluation Partition:** `validation` (128 leaf samples across 4 classes)  
> **Split Manifest Hash (SHA256):** `5d9c1be1d4c7338c60a3f56ccf66b803bfe7333eb8d4330d5b55d6dc51c33f52`  
> **Selection Metric:** Primary: `validation_macro_f1` (per TECH_RULES §18.1 / Overview §18)

---

## 1. Executive Summary & Selection Decision

In accordance with TECH_RULES Rule 18 and Overview §18, all candidate image models for lettuce were evaluated on the **identical leaf-level validation split** (`5d9c1be1d4c7338c60a3f56ccf66b803bfe7333eb8d4330d5b55d6dc51c33f52`), containing 128 individually segmented leaf crops.

### Selected Model
* **Model Name:** `lettuce_efficientnet_b0_v1.0.0`
* **Family:** Transfer Learning (`EfficientNetB0` fine-tuned)
* **Validation Macro-F1:** **`0.9119`** (vs Custom CNN `0.7045`, OpenCV+RF `0.4930`, OpenCV+SVM `0.4840`)
* **Validation Accuracy:** **`0.9453`** (121 / 128 correct)
* **Decision Rationale:**
  1. `EfficientNetB0` outperforms the scratch Custom CNN by **+20.74% Macro-F1** (`0.9119` vs `0.7045`).
  2. `EfficientNetB0` outperforms classical handcrafted OpenCV feature models (Random Forest and SVM) by over **+41.8% Macro-F1**.
  3. `EfficientNetB0` demonstrates robust generalization on minority deficiency classes, achieving **1.0000 F1 on Phosphorus Deficiency** and **0.9767 F1 on Potassium Deficiency**, and substantially higher recall on Nitrogen Deficiency (`0.9000` vs `0.4000` on CNN).

---

## 2. Comprehensive Model Comparison Table

All models below consume individual segmented leaves extracted during preprocessing (`source_map.csv` / SAM leaf crops). No whole-canopy images or unsegmented frames were used.

| Model Candidate | Family | Input Modality | Split Manifest SHA256 | Parameters / Complexity | Val Accuracy | Val Macro-Precision | Val Macro-Recall | **Val Macro-F1** (Primary) | Rank |
|---|---|---|---|---|---:|---:|---:|---:|:---:|
| **`lettuce_efficientnet_b0_v1.0.0`** | **EfficientNetB0** | Individual Leaf Crop (224x224 RGB) | `5d9c1be1d4c7338c60a3f56ccf66b803bfe7333eb8d4330d5b55d6dc51c33f52` | **4,012,672 params** (15.3 MB fp32) | **94.53%** | **0.8958** | **0.9438** | **0.9119** | **1 (Selected)** |
| `lettuce_custom_cnn_v1.0.0` | Custom CNN (4 Conv) | Individual Leaf Crop (224x224 RGB) | `5d9c1be1d4c7338c60a3f56ccf66b803bfe7333eb8d4330d5b55d6dc51c33f52` | 456,228 params (1.7 MB fp32) | 80.47% | 0.7514 | 0.6792 | 0.7045 | 2 |
| `lettuce_feature_random_forest_v1.0.0` | OpenCV Features + RF | Handcrafted 7-dim GLCM/Color Vector | `5d9c1be1d4c7338c60a3f56ccf66b803bfe7333eb8d4330d5b55d6dc51c33f52` | 150 trees (20,050 decision nodes) | 60.94% | 0.5106 | 0.4901 | 0.4930 | 3 |
| `lettuce_feature_svm_v1.0.0` | OpenCV Features + SVM | Handcrafted 7-dim GLCM/Color Vector | `5d9c1be1d4c7338c60a3f56ccf66b803bfe7333eb8d4330d5b55d6dc51c33f52` | 497 support vectors (RBF kernel, C=2.0) | 51.56% | 0.5428 | 0.5071 | 0.4840 | 4 |

---

## 3. Per-Class F1-Score Breakdown

Validation split class distribution ($N=128$):
* `Healthier`: 63 samples
* `PhosphorusDeficiency`: 33 samples
* `PotassiumDeficiency`: 22 samples
* `NitrogenDeficiency`: 10 samples (severe minority stress class)

| Target Class | Support | `lettuce_efficientnet_b0` F1 | `lettuce_custom_cnn` F1 | `lettuce_feature_rf` F1 | `lettuce_feature_svm` F1 |
|---|---:|---:|---:|---:|---:|
| `Healthier` | 63 | **0.9508** | 0.9104 | 0.7931 | 0.6200 |
| `NitrogenDeficiency` | 10 | **0.7200** | 0.5000 | 0.1212 | 0.1633 |
| `PhosphorusDeficiency` | 33 | **1.0000** | 0.7576 | 0.6769 | 0.6429 |
| `PotassiumDeficiency` | 22 | **0.9767** | 0.6500 | 0.3810 | 0.5098 |
| **Macro Average** | **128** | **0.9119** | **0.7045** | **0.4930** | **0.4840** |

### Per-Class Detailed Performance for Selected Model (`lettuce_efficientnet_b0_v1.0.0`)

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `Healthier` | 0.9831 | 0.9206 | 0.9508 | 63 |
| `NitrogenDeficiency` | 0.6000 | 0.9000 | 0.7200 | 10 |
| `PhosphorusDeficiency` | 1.0000 | 1.0000 | 1.0000 | 33 |
| `PotassiumDeficiency` | 1.0000 | 0.9545 | 0.9767 | 22 |

---

## 4. Input-Mode Verification & Fair Comparison Protocol

1. **Leaf-Level Input Confirmation**:
   - Every model was trained and evaluated strictly on individual leaf crops segmented from raw plant images.
   - For deep learning models (`CustomCNN`, `EfficientNetB0`), input is a normalized 3-channel RGB tensor ($224 \times 224 \times 3$) centered on an isolated leaf.
   - For classical feature models (`RandomForest`, `SVM`), input is a 7-dimensional engineered feature vector (`exg_mean`, `exr_mean`, `yellow_pixel_ratio`, `brown_pixel_ratio`, `mean_hue`, `mean_saturation`, `contour_compactness`) computed strictly within the leaf contour boundary.
2. **Data Split Leakage Prevention**:
   - All models used the exact same split manifest file (`processed_data/splits/image_splits_lettuce.json` / `image_splits_lettuce.csv`), ensuring zero train-validation leakage.
   - Verification SHA256: `5d9c1be1d4c7338c60a3f56ccf66b803bfe7333eb8d4330d5b55d6dc51c33f52`.

---

## 5. Agronomic & Operational Observations

1. **Deficiency Symptom Recognition**:
   - `EfficientNetB0`'s pretrained ImageNet feature extractors excel at distinguishing subtle marginal chlorosis (potassium deficiency) and purple/bronze venation (phosphorus deficiency), achieving near-perfect separation (1.00 and 0.98 F1).
   - Handcrafted color and texture features (RF/SVM) fail dramatically on Nitrogen Deficiency ($F_1 < 0.17$) due to overlapping yellowing indices with natural leaf aging and background variations.
2. **Computational Trade-Offs**:
   - While `CustomCNN` requires fewer parameters (456K vs 4.01M), its lower Macro-F1 (`0.7045`) results in significant misclassification between Healthy and nutrient-deficient crops, which poses unacceptable agronomic risk.
   - `EfficientNetB0` evaluates in $<25\text{ ms}$ on GPU and $<85\text{ ms}$ on CPU per leaf, well within the real-time inference latency budget ($<1.5\text{ s}$).

---

## 6. Selection Recommendation

Promote **`lettuce_efficientnet_b0_v1.0.0`** as the active production model for the lettuce leaf image classification pipeline.

