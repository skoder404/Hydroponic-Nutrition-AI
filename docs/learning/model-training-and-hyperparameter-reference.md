# Comprehensive Model Training, Hyperparameters & Fine-Tuning Study Guide

> **Scope:** Complete architectural and implementation reference for all model families in Hydroponic Nutrition AI:
> 1. **Sensor Tabular Telemetry Models** (`DecisionTree`, `RandomForest`, `SVM`, `KNN`, `LogisticRegression`)
> 2. **Custom Convolutional Neural Networks** (`CustomCNN`)
> 3. **Transfer Learning & Deep Fine-Tuning** (`EfficientNetB0`)
> 4. **Handcrafted Foliar Feature Models** (`OpenCV Features + RF/SVM`)

---

## 1. Sensor Tabular Telemetry Models

### 1.1 Architecture & Pipeline Design
Sensor models consume numerical telemetry streams (pH, EC in $\text{mS/cm}$, water temperature, ambient temperature, relative humidity, and dissolved oxygen).

- **Pipeline Bundle:** Preprocessing is strictly encapsulated inside `SensorPipeline` ([`ml/sensor_models/train_sensor.py`](file:///d:/Hydroponic-Nutrition-AI/ml/sensor_models/train_sensor.py#L36-L75)):
  - `imputer`: `SimpleImputer(strategy="median")`
  - `scaler`: `StandardScaler()` (fitted strictly on training rows; identity/passthrough for trees)
  - `classifier`: Scikit-Learn estimator

### 1.2 Code Locations & Exact Training Lines
- **Primary Source File:** [`ml/sensor_models/train_sensor.py`](file:///d:/Hydroponic-Nutrition-AI/ml/sensor_models/train_sensor.py)
- **Tuning Engine:** [`ml/sensor_models/tune.py`](file:///d:/Hydroponic-Nutrition-AI/ml/sensor_models/tune.py)
- **Crop Orchestrators:**
  - Lettuce: [`scripts/run_lettuce_experiments.py`](file:///d:/Hydroponic-Nutrition-AI/scripts/run_lettuce_experiments.py)
  - Spinach: [`scripts/run_spinach_experiments.py`](file:///d:/Hydroponic-Nutrition-AI/scripts/run_spinach_experiments.py)
  - Tomato: [`scripts/run_tomato_experiments.py`](file:///d:/Hydroponic-Nutrition-AI/scripts/run_tomato_experiments.py)

#### Key Line Numbers:
- **Lines 77–161 (`get_classifier`)**: Instantiates the classifier with specific hyperparameters.
- **Lines 164–178 (`build_sensor_pipeline`)**: Assembles `imputer`, `StandardScaler`, and `classifier`.
- **Line 333 (The Exact Training Execution)**:
  ```python
  pipeline.fit(X_train, y_train)
  ```
- **Line 336 (Validation Inference)**:
  ```python
  y_val_pred = pipeline.predict(X_val)
  ```
- **Lines 411–414 (Artifact Persistence)**:
  ```python
  joblib.dump(pipeline, artifact_path)
  ```

### 1.3 Hyperparameters & Search Spaces
Loaded from versioned YAML files in [`ml/sensor_models/configs/`](file:///d:/Hydroponic-Nutrition-AI/ml/sensor_models/configs/):

| Model Family | Search Space Grid | Promoted Model & Parameters |
|---|---|---|
| **Decision Tree** | `criterion: ['gini', 'entropy']`<br>`max_depth: [3, 5, 8, None]`<br>`min_samples_leaf: [1, 2, 4]` | **Lettuce**: `criterion='gini'`, `max_depth=5`, `min_samples_leaf=1` (Val Macro-F1: `0.8872`)<br>**Spinach**: `criterion='gini'`, `max_depth=3`, `min_samples_leaf=1` (Val Macro-F1: `1.0000`) |
| **Random Forest** | `n_estimators: [25, 50, 100]`<br>`max_depth: [4, 6, None]`<br>`min_samples_leaf: [1, 2]`<br>`class_weight: [None, 'balanced']` | **Tomato**: `n_estimators=100`, `max_depth=None`, `min_samples_leaf=1`, `class_weight=None` (Val Macro-F1: `0.9685`) |
| **SVM (SVC)** | `C: [0.5, 1.0, 2.5]`<br>`gamma: ['scale', 0.05, 0.1]`<br>`kernel: ['rbf', 'linear']`<br>`probability: True` | Scaled features, RBF kernel with Platt sigmoid probability calibration |
| **KNN** | `n_neighbors: [3, 5, 7, 9]`<br>`weights: ['uniform', 'distance']` | Standardized Euclidean distance metric |
| **Logistic Regression** | `C: [0.05, 0.2, 1.0, 5.0]`<br>`class_weight: [None, 'balanced']`<br>`max_iter: 1000` | L2-regularized multinomial loss |

### 1.4 How Sensor Models Were Tuned
1. **Guarded Group K-Fold (`GuardedGroupKFold`)**:
   Standard random splits cause severe diurnal autocorrelation leakage in continuous sensor logging. Tuning groups rows physically:
   - **Lettuce**: Grouped by calendar day (`date` / `day_index`).
   - **Spinach**: Grouped by bench tank and 6-hour temporal block (`bench|date|block`).
   - **Tomato**: Grouped by hydroponic batch (`experiment`).
2. **Objective Metric**: Grid search ranks parameter combinations strictly by **validation Macro-F1**, protecting minority stress classes.
3. **No Test Leakage**: `test.csv` was strictly isolated and never accessed during hyperparameter tuning (TECH_RULES Rule 3).

---

## 2. Custom Convolutional Neural Networks (`CustomCNN`)

### 2.1 Architecture
Designed as a lightweight, scratch-trained baseline specifically sized for small agricultural datasets to prevent immediate overfitting.

- **Source File**: [`ml/image_models/custom_cnn.py`](file:///d:/Hydroponic-Nutrition-AI/ml/image_models/custom_cnn.py)
- **Layer Breakdown**:
  1. `ConvBlock 1`: `Conv2d(3, 32, k=3, p=1)` $\rightarrow$ `BatchNorm2d(32)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (Output: $112 \times 112$)
  2. `ConvBlock 2`: `Conv2d(32, 64, k=3, p=1)` $\rightarrow$ `BatchNorm2d(64)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (Output: $56 \times 56$)
  3. `ConvBlock 3`: `Conv2d(64, 128, k=3, p=1)` $\rightarrow$ `BatchNorm2d(128)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (Output: $28 \times 28$)
  4. `ConvBlock 4`: `Conv2d(128, 256, k=3, p=1)` $\rightarrow$ `BatchNorm2d(256)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (Output: $14 \times 14$)
  5. Classifier Head: `Linear(256 * 14 * 14, 256)` $\rightarrow$ `BatchNorm1d(256)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(256, num_classes)`.
- **Parameter Count**: $\approx 456,000 - 458,000$ parameters (1.7 MB fp32).

### 2.2 Code Locations & Exact Training Lines
In [`ml/image_models/train_image.py`](file:///d:/Hydroponic-Nutrition-AI/ml/image_models/train_image.py):
- **Line 265**: `model = CustomCNN(crop=crop)`
- **Line 288**: `criterion = nn.CrossEntropyLoss()`
- **Lines 289–294**: `optimizer = create_optimizer(model, optimizer_name="adamw", lr=1e-3, weight_decay=1e-4)`
- **Line 362**: `model.train()`
- **Lines 398–402 (The Forward & Backward Optimization Step)**:
  ```python
  optimizer.zero_grad()
  outputs = model(inputs)
  loss = criterion(outputs, targets)
  loss.backward()
  optimizer.step()
  ```

### 2.3 Hyperparameters & Tuning Enhancements
- **Batch Size**: `64` (with dynamic halving retry ladder upon CUDA Out-Of-Memory).
- **Learning Rate**: `0.001` ($1 \times 10^{-3}$).
- **Weight Decay**: `0.0001` ($1 \times 10^{-4}$).
- **Scheduler**: `ReduceLROnPlateau(mode="max", factor=0.5, patience=2)` monitoring validation Macro-F1 (lines 295–297).
- **Early Stopping**: `EarlyStopping(patience=3, min_delta=0.0001)` stops training if validation Macro-F1 stagnates (lines 132–146).
- **Augmentation**: Applied strictly to training split (random flips, $\pm 15^\circ$ rotations, color jitter).

---

## 3. Transfer Learning & Deep Fine-Tuning (`EfficientNetB0`)

### 3.1 Architecture
Leverages inverted residual bottleneck blocks with depthwise separable convolutions and Squeeze-and-Excitation (SE) optimization, initialized from torchvision pretrained ImageNet-1K weights (`EfficientNet_B0_Weights.IMAGENET1K_V1`).

- **Source File**: [`ml/image_models/efficientnet.py`](file:///d:/Hydroponic-Nutrition-AI/ml/image_models/efficientnet.py)
- **Head Replacement**: Default 1000-class head replaced with:
  ```python
  nn.Sequential(
      nn.Dropout(p=0.2, inplace=True),
      nn.Linear(in_features=1280, out_features=num_classes)
  )
  ```
- **Parameter Count**: $4,012,672$ (Lettuce), $4,015,234$ (Spinach), $4,021,639$ (Tomato).

### 3.2 Two-Stage Freeze-Then-Unfreeze Fine-Tuning
Fine-tuning deep neural networks from scratch on limited data risks **catastrophic forgetting** if the large pretrained backbone gradients are updated simultaneously with a randomized classifier head.

The project implements an explicit two-stage schedule configured via `FreezeScheduleConfig` ([`ml/image_models/efficientnet.py`](file:///d:/Hydroponic-Nutrition-AI/ml/image_models/efficientnet.py#L46-L59)):

```
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Frozen Backbone (Epochs 1-2)                                  │
│ [Pretrained Backbone (4,007,548 params FROZEN)] ──> [New Head TRAIN]  │
│ LR = 0.001 | Optimizer = AdamW                                        │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ Trigger at Epoch config.unfreeze_at_epoch
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Fine-Tuning (Epochs 3+)                                       │
│ [Backbone Blocks 7-8 UNFREEZE] ───────────────> [Classifier Head TRAIN]│
│ Fine-Tuning LR = 0.0001 (10x smaller) | Optimizer = AdamW Reconfigured│
└────────────────────────────────────────────────────────────────────────┘
```

#### Exact Line Numbers:
In [`ml/image_models/train_image.py`](file:///d:/Hydroponic-Nutrition-AI/ml/image_models/train_image.py):
- **Lines 249–261**: Instantiates `FreezeScheduleConfig` and builds model with initially frozen backbone.
- **Lines 334–361 (The Stage 2 Unfreeze Transition)**:
  ```python
  unfrozen = model.step_schedule(epoch)
  if unfrozen:
      optimizer = create_optimizer(
          model=model,
          optimizer_name=config.optimizer_name,
          lr=config.finetune_lr,          # Drops from 1e-3 to 1e-4
          weight_decay=config.weight_decay,
      )
  ```
- **Lines 398–402**: Forward pass, CrossEntropyLoss, backprop, optimizer step.

### 3.3 Post-Processing Probability Calibration (Temperature Scaling)
Implemented in `T5-031` ([`scripts/derive_uncertainty_thresholds.py`](file:///d:/Hydroponic-Nutrition-AI/scripts/derive_uncertainty_thresholds.py)):
- Solves overconfidence without changing class predictions or Macro-F1:
  $$q_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
- Optimizes $T > 0$ strictly on validation cross-entropy loss.
- **Results**:
  - **Lettuce**: $T=1.460$, Expected Calibration Error dropped from $0.1790 \rightarrow \mathbf{0.1109}$.
  - **Spinach**: $T=0.901$, ECE dropped to $\mathbf{0.0286}$.
  - **Tomato**: $T=0.706$, ECE dropped from $0.0695 \rightarrow \mathbf{0.0106}$.

---

## 4. Handcrafted Foliar Feature Models (`OpenCV Features + RF/SVM`)

### 4.1 Feature Extraction Pipeline
Before training, raw leaves or segmented SAM2 leaf masks are converted into low-dimensional tabular descriptors to test classical computer vision hypotheses.

- **Extraction Engine**: [`scripts/build_features.py`](file:///d:/Hydroponic-Nutrition-AI/scripts/build_features.py)
- **Features Extracted**:
  - **Color Indices**:
    - Excess Green: $\text{ExG} = 2G - R - B$
    - Excess Red: $\text{ExR} = 1.4R - G$
    - Mean Hue & Saturation in HSV space
    - Yellow & Brown necrotic pixel ratios
  - **Texture (GLCM)**: Gray-Level Co-occurrence Matrix Contrast and Entropy.
  - **Morphology / Shape**: Compactness ($\frac{P^2}{4\pi A}$), Solidity ($\frac{A}{\text{ConvexArea}}$), and Bounding Aspect Ratio.

### 4.2 Training Pipeline & Exact Lines
- **Primary Source File**: [`ml/image_models/train_features.py`](file:///d:/Hydroponic-Nutrition-AI/ml/image_models/train_features.py)

#### Key Line Numbers:
- **Lines 156–157**: Fits the `StandardScaler` **strictly** on training feature rows:
  ```python
  scaler = StandardScaler()
  scaler.fit(X_train)
  ```
- **Lines 160–180**: Instantiates the classifier:
  ```python
  classifier = RandomForestClassifier(**clf_params)  # Line 168
  classifier = SVC(**clf_params)                     # Line 179
  ```
- **Line 186 (The Pipeline Fit Call)**:
  ```python
  pipeline = FeaturePipeline([("scaler", scaler), ("classifier", classifier)])
  pipeline.fit(X_train, y_train)
  ```
- **Line 189**: Validation inference `val_preds = pipeline.predict(X_val)`.
- **Line 216**: Model persistence via `joblib.dump(pipeline, artifact_path)`.

### 4.3 Hyperparameters & Tuning Enhancements
- **Random Forest**:
  - `n_estimators`: `100 - 150`
  - `max_depth`: `8` (prevents memorization of noise in 7-dimensional space)
  - `min_samples_split`: `3`
  - `min_samples_leaf`: `2`
  - `class_weight`: `"balanced"` (adjusts weights inversely proportional to class frequencies)
- **SVM**:
  - `C`: `2.0`
  - `kernel`: `"rbf"`
  - `gamma`: `"scale"`
  - `probability`: `True` (Platt scaling)
  - `class_weight`: `"balanced"`
- **Why Classical Features Were Defeated in Comparison**:
  - Handcrafted color/texture descriptors achieved only $\approx 43\% - 52\%$ Macro-F1 across crops because subtle visual symptoms (e.g. initial chlorosis vs natural leaf maturity) overlap heavily in basic ExG/GLCM space, whereas deep convolutional hierarchies learn fine-grained spatial lesion patterns.
