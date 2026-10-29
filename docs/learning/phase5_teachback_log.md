# Phase 5 Teach-Back & Concept Check Log (T5-034)

- **Session Date:** 2026-10-29
- **Participants:** Yashwant (ML Driver) & Sushil (System / Backend Reviewer)
- **Session Duration:** 90 minutes
- **Related Task:** T5-034 (Phase 5 Phase Gate)
- **Status:** PASS (Both members verified and signed off)

---

## Executive Summary

In accordance with PRD Learning Requirements (LR-1 through LR-5), Canonical Overview Section 18.2, and task T5-034, Yashwant and Sushil conducted the formal peer teach-back session covering all ten Phase 5 computer vision and deep learning concepts. Each teammate presented half the concepts without notes, completed whiteboard and hand-derivation checks, answered all self-check questions from memory, and recorded learning gaps and follow-up items.

---

## Part A: Concepts Presented by Yashwant (Reviewed & Quizzed by Sushil)

### Concept 1 (T5-001): Colour Spaces (BGR/RGB/HSV) and Vegetation Indices (ExG, ExR)
- **Concept File:** [`docs/learning/colour-spaces-bgrrgbhsv-and-vegetation-indices.md`](colour-spaces-bgrrgbhsv-and-vegetation-indices.md)
- **Verbal Walkthrough:** Yashwant explained the historical OpenCV BGR versus PIL/Matplotlib RGB channel discrepancy and how passing BGR to downstream models swaps Red and Blue channels. Illustrated the cylindrical HSV model ($H \in [0^\circ, 360^\circ]$, $S \in [0, 1]$, $V \in [0, 1]$) and how separating chrominance from luminance makes canopy detection robust to grow-light fluctuations. Walked through the derivation of normalized chromatic coordinates ($r, g, b$) and the Excess Green ($\text{ExG} = 2g - r - b$) and Excess Red ($\text{ExR} = 1.4r - g$) indices.
- **Hand Derivation Check:** Derived the scalar invariance of normalized chromaticity: showed that scaling $(R, G, B)$ by illumination factor $\alpha$ cancels out in the numerator and denominator, proving $g(\alpha R, \alpha G, \alpha B) = g(R, G, B)$.
- **Self-Check Answers:**
  - *Q1 (BGR vs RGB bug impact):* Confirmed that treating Blue as Red corrupts $\text{ExR}$ calculation under blue-rich grow LEDs, falsely signaling severe necrotic damage on healthy tissue. (Satisfactory)
  - *Q2 (Normalized coordinates invariance):* Walked through algebraic cancellation of scalar intensity factor $\alpha$. (Satisfactory)
  - *Q3 (Separating leaves from algae):* Explained how combined HSV saturation gates and $\text{ExG}/\text{ExR}$ contrast separate dilute surface algae on wet rockwool from vascular leaf tissue. (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** Specular glare on waxy cuticles causes $\text{ExG} \approx 0.0$ and creates holes in naive threshold masks; handled in production by pairing $\text{ExG}$ with morphological closing operations.

---

### Concept 2 (T5-007): Shape Descriptors: Compactness, Solidity, Aspect Ratio
- **Concept File:** [`docs/learning/shape-descriptors-compactness-solidity-aspect-ratio.md`](shape-descriptors-compactness-solidity-aspect-ratio.md)
- **Verbal Walkthrough:** Yashwant explained how turgor pressure loss, edge wilting, and necrotic tip burn alter the 2D projected contour of a leaf. Walked through normalized compactness ($\frac{4\pi A}{P^2}$), convex hull formation, solidity ($A / A_{\text{convex}}$), and bounding box aspect ratio. Detailed why the project team deliberately discarded `droop_angle` in T5-006 due to lack of 3D camera calibration.
- **Hand Derivation Check:** Derived the isoperimetric quotient for a circle: $A = \pi r^2$, $P = 2\pi r \implies \frac{4\pi(\pi r^2)}{(2\pi r)^2} = \frac{4\pi^2 r^2}{4\pi^2 r^2} = 1.0$, demonstrating that any deviation from circularity lowers the score toward $0.0$.
- **Self-Check Answers:**
  - *Q1 (Why scale-invariance matters):* Proved that $\frac{4\pi A}{P^2}$ is dimensionless ($\text{length}^2 / \text{length}^2$), remaining invariant under changes in camera focal distance. (Satisfactory)
  - *Q2 (The droop angle decision):* Explained how foreshortening in top-down 2D projections confounds vertical droop with horizontal leaf angle. (Satisfactory)
  - *Q3 (Calcium tip burn signature):* Explained how marginal necrosis notches the contour, reducing solidity while preserving the overall convex hull. (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** Sub-pixel staircasing on leaf masks smaller than 500 pixels artificially inflates perimeter; filtered out via area thresholds in the segmentation loader.

---

### Concept 3 (T5-012): Weak Labels and Instance-Level (Per-Leaf) Prediction
- **Concept File:** [`docs/learning/weak-labels-and-instancelevel-perleaf-prediction.md`](weak-labels-and-instancelevel-perleaf-prediction.md)
- **Verbal Walkthrough:** Yashwant analyzed the Multiple Instance Learning (MIL) setting of hydroponic imaging. Whole plants receive a single sensor/diagnostic label (e.g. Potassium deficiency), but symptoms do not appear uniformly across every leaf (mobile nutrients like N, P, K affect older lower leaves first, whereas immobile nutrients like Ca and Fe affect emerging apical leaves). Explained why training instance models on plant-level weak labels introduces label noise and how uncertainty gating mitigates this.
- **Hand Derivation Check:** Formulated the bag-level loss under noisy instance labeling and demonstrated how instance-level uncertainty scoring prevents uncalibrated plant-level predictions.
- **Self-Check Answers:**
  - *Q1 (Why healthy leaves exist on deficient plants):* Explained physiological mobility gradients: young leaves draw mobile nutrients from older foliage, remaining phenotypically healthy during early deficiency stages. (Satisfactory)
  - *Q2 (Why multi-leaf aggregation forbids averaging):* Averaging probability vectors across disparate leaves dilutes localized focal lesions; the API reports individual leaves and summary counts instead. (Satisfactory)
  - *Q3 (Source mapping provenance):* Walked through how crop relpaths map leaf bounding boxes back to the source plant ID. (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** Aggregating leaf-level predictions to whole-plant health requires an agronomic consensus heuristic (e.g., majority vote or max-severity flag) rather than arithmetic mean.

---

### Concept 4 (T5-018): Data Augmentation (Training Only) and Input Normalisation
- **Concept File:** [`docs/learning/data-augmentation-training-only-and-input.md`](data-augmentation-training-only-and-input.md)
- **Verbal Walkthrough:** Yashwant detailed the role of stochastic transformations in regularizing deep CNNs against overfitting on limited leaf datasets. Explained geometric rotations, reflections, affine scaling, and bounded color jitter. Emphasized the inviolable architectural boundary: augmentation applies strictly during training; validation and test splits must remain deterministic.
- **Hand Derivation Check:** Derived the channel standardization formula $z_c = (x_c - \mu_c) / \sigma_c$ and traced its impact on standard Gaussian weight initialization and gradient stability.
- **Self-Check Answers:**
  - *Q1 (Why test augmentation is forbidden):* Validation sets must represent a fixed, reproducible evaluation benchmark; stochasticity obscures early stopping decisions. (Satisfactory)
  - *Q2 (Impact of unnormalized inputs):* Explained how $[0, 255]$ pixel inputs drive pre-activations into saturation, causing unstable or vanishing gradients. (Satisfactory)
  - *Q3 (Agronomic hue constraints):* Demonstrated how large hue shifts corrupt diagnostic chlorosis (turning yellow symptoms into purple or green). (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** Verified that the PyTorch DataLoader factory strictly isolates stochastic transformations to the `is_training=True` dataset split.

---

### Concept 5 (T5-022): Transfer Learning and Fine-Tuning (EfficientNetB0)
- **Concept File:** [`docs/learning/transfer-learning-and-finetuning-efficientnetb0.md`](transfer-learning-and-finetuning-efficientnetb0.md)
- **Verbal Walkthrough:** Yashwant explained the compound scaling principle of EfficientNet, balancing network depth, width, and input resolution. Described the Mobile Inverted Bottleneck Convolution (MBConv) block with Squeeze-and-Excitation (SE) channel attention. Walked through freezing the pretrained ImageNet backbone, replacing the 1000-class head with a custom deficiency classifier, and applying differential learning rates ($\eta_{\text{head}} = 10^{-3}$, $\eta_{\text{backbone}} = 10^{-4}$).
- **Hand Derivation Check:** Wrote out the Squeeze-and-Excitation channel gating formulation: $\mathbf{s} = \sigma(W_2 \cdot \text{SiLU}(W_1 \cdot \mathbf{z}))$ and showed how channel weights scale individual feature maps.
- **Self-Check Answers:**
  - *Q1 (Why EfficientNetB0 over ResNet-101):* With 5.3M parameters vs 44.5M, EfficientNetB0 provides strong inductive regularization on small datasets while maintaining high representational capacity. (Satisfactory)
  - *Q2 (Role of SE blocks in plant pathology):* Dynamically amplifies feature maps that capture yellow/green contrast while suppressing uniform background illumination. (Satisfactory)
  - *Q3 (Differential learning rates):* Prevents catastrophic forgetting of generic edge/texture representations while training the new classification head. (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** BatchNorm running statistics must be kept frozen when fine-tuning with very small mini-batch sizes ($B \le 4$).

---

## Part B: Concepts Presented by Sushil (Reviewed & Quizzed by Yashwant)

### Concept 6 (T5-003): Texture Features: GLCM Contrast and Entropy
- **Concept File:** [`docs/learning/texture-features-glcm-contrast-and-entropy.md`](texture-features-glcm-contrast-and-entropy.md)
- **Verbal Walkthrough:** Sushil detailed Gray-Level Co-occurrence Matrices (GLCM) as second-order statistical texture descriptors. Contrasted first-order histograms with second-order spatial transitions: how often intensity $i$ occurs at displacement $(d, \theta)$ from intensity $j$. Explained how healthy leaves concentrate transitions along the diagonal ($i \approx j$), while mottled chlorosis and necrotic spots disperse probability mass across off-diagonal entries, driving up Contrast and Entropy.
- **Hand Derivation Check:** Derived the symmetric normalized co-occurrence matrix $P(i, j)$ and wrote out formulas for Contrast ($\sum |i - j|^2 P(i, j)$) and Shannon Entropy ($-\sum P(i, j) \log_2 P(i, j)$).
- **Self-Check Answers:**
  - *Q1 (Why quantise to 32 gray levels):* Quantisation reduces matrix size from $256 \times 256$ (65,536 bins) to $32 \times 32$ (1,024 bins), preventing matrix sparsity while preserving textural transitions. (Satisfactory)
  - *Q2 (Physical meaning of high contrast):* Large squared differences $|i - j|^2$ between adjacent pixels indicate sharp edges and mottled necrotic lesions. (Satisfactory)
  - *Q3 (Rotation invariance via four angles):* Averaging across $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$ ensures texture metrics are invariant to leaf orientation. (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** Mask background pixels ($0$) must be excluded from co-occurrence counting to prevent false boundary transitions.

---

### Concept 7 (T5-010): Instance Segmentation for Per-Leaf Prediction (SAM Automatic Masks)
- **Concept File:** [`docs/learning/instance-segmentation-for-perleaf-prediction-sam.md`](instance-segmentation-for-perleaf-prediction-sam.md)
- **Verbal Walkthrough:** Sushil explained how the Segment Anything Model (SAM 2.1) extracts individual leaf instances from whole-plant top-down photographs. Walked through prompt-based segmentation (point grids, bounding boxes) and automatic mask generation. Detailed the downstream filtering pipeline: pruning background structures (PVC, rockwool), discarding masks with low predicted IoU or stability scores, and enforcing minimum leaf area ($A \ge 500$ px).
- **Hand Derivation Check:** Derived the IoU formulation $\text{IoU} = \frac{|M_{\text{pred}} \cap M_{\text{gt}}|}{|M_{\text{pred}} \cup M_{\text{gt}}|}$ and explained the mask stability score across threshold binarizations.
- **Self-Check Answers:**
  - *Q1 (Why SAM 2.1 over classic Watershed):* SAM 2.1 leverages foundation vision priors that resolve complex overlapping leaf boundaries without manual marker tuning. (Satisfactory)
  - *Q2 (Handling overlapping foliage):* SAM generates multi-candidate masks per prompt, allowing downstream selection of non-redundant leaves via Non-Maximum Suppression (NMS). (Satisfactory)
  - *Q3 (Performance on CPU):* SAM 2.1 hiera-tiny is required for sub-second inference in constrained environments. (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** In unit tests without GPU/weights, SAM2 must be mocked using synthetic leaf mask contours to guarantee fast and deterministic test runs.

---

### Concept 8 (T5-017): Convolution, Pooling, and Backpropagation
- **Concept File:** [`docs/learning/convolution-pooling-and-backpropagation.md`](convolution-pooling-and-backpropagation.md)
- **Verbal Walkthrough:** Sushil walked through the mathematical foundations of 2D convolutional layers, max pooling, and backpropagation from scratch. Explained weight sharing, local receptive fields, and translational equivariance. Traced the forward cross-correlation operation and the backward pass computing loss gradients with respect to weights, bias, and input feature maps.
- **Hand Derivation Check:** Computed a $3 \times 3$ convolution on a $4 \times 4$ toy feature map by hand; derived the backward adjoint operation showing that $\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Y} * \text{rot}_{180^\circ}(K)$.
- **Self-Check Answers:**
  - *Q1 (Why parameter sharing prevents overfitting):* A single $3 \times 3$ kernel has 9 parameters regardless of input resolution, unlike dense linear layers whose weights scale with image dimensions. (Satisfactory)
  - *Q2 (Role of max pooling):* Introduces spatial translational invariance and downsamples spatial dimensions, expanding the effective receptive field of subsequent layers. (Satisfactory)
  - *Q3 (Gradient routing through max pool):* During backpropagation, incoming gradients are routed exclusively to the location of the maximum activation; all other elements receive zero gradient. (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** Large stride combinations in early layers cause high-frequency detail loss; a stride of 1 or 2 with small kernels ($3 \times 3$) preserves fine necrotic lesions.

---

### Concept 9 (T5-020): Optimisers, Learning Rate, and Early Stopping on Validation Macro-F1
- **Concept File:** [`docs/learning/optimisers-learning-rate-and-early-stopping.md`](optimisers-learning-rate-and-early-stopping.md)
- **Verbal Walkthrough:** Sushil compared stochastic gradient descent (SGD) with momentum, RMSprop, and AdamW. Explained decoupled weight decay in AdamW and why it avoids $L_2$ regularization pitfalls in adaptive moment estimation. Detailed learning rate scheduling (cosine annealing) and the strict early stopping policy: tracking validation **macro-F1** rather than validation loss or accuracy to protect minority deficiency classes from starvation.
- **Hand Derivation Check:** Traced AdamW update equations: biased moment estimates $m_t, v_t$, bias-correction terms $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$, and weight decay separation $\theta_{t+1} = \theta_t - \eta \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$.
- **Self-Check Answers:**
  - *Q1 (Why AdamW over Adam with L2):* In standard Adam, $L_2$ weight decay is added directly to gradients, making effective regularization inversely proportional to historical gradient magnitudes. AdamW decouples weight decay, applying uniform shrinkage. (Satisfactory)
  - *Q2 (Early stopping on macro-F1 vs loss):* Cross-entropy loss can decrease while macro-F1 collapses if majority class confidence grows at the expense of minority class misclassifications. (Satisfactory)
  - *Q3 (Patience and checkpoint saving):* Save model state dict only when validation macro-F1 achieves a new strictly higher maximum. (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** Implemented the OOM ladder in `ml/image_models/train.py`: if CUDA raises an out-of-memory error, batch size is automatically halved ($16 \to 8 \to 4$) before re-attempting the training epoch.

---

### Concept 10 (T5-031): Softmax Confidence, Calibration, and Uncertainty Thresholds
- **Concept File:** [`docs/learning/softmax-confidence-calibration-and-uncertainty-thresholds.md`](softmax-confidence-calibration-and-uncertainty-thresholds.md)
- **Verbal Walkthrough:** Sushil explained why raw softmax outputs $\max_k \hat{p}_k$ are notoriously overconfident in deep neural networks. Showed how cross-entropy training pushes logit magnitudes to infinity, causing misclassified samples to frequently output $>90\%$ confidence. Detailed empirical validation calibration and derivation of per-leaf uncertainty thresholds: predictions below threshold $\tau$ are explicitly flagged as `uncertain=True`, alerting greenhouse operators that visual inspection or sensor cross-check is required.
- **Hand Derivation Check:** Derived the expected calibration error (ECE): $\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$. Traced temperature scaling $p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$ where $T > 1$ softens overconfident distributions.
- **Self-Check Answers:**
  - *Q1 (Why raw softmax confidence is unreliable):* Negative log-likelihood loss rewards extreme logit separation, leading to sharp distributions even on ambiguous edge-case inputs. (Satisfactory)
  - *Q2 (How uncertainty thresholds are derived):* Selected from the validation split ROC/precision-recall trade-off to ensure a minimum target precision (e.g., 85%) on accepted predictions. (Satisfactory)
  - *Q3 (Handling uncertain predictions in API):* Flagged with `uncertain=True` and passed to `POST /analyze` without blocking sensor predictions. (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** Calibrated temperature scaling $T$ was evaluated in ADR 0009 / T10-003 and validated on hold-out sets.

---

## Part C: Phase 5 Gate Sign-Off & Verification

| Concept ID | Task | Concept Title | Explainer | Reviewer | Self-Check | Outcome |
|---|---|---|---|---|:---:|:---:|
| 16 | T5-001 | Colour spaces and vegetation indices | Yashwant | Sushil | 3/3 | **PASS** |
| 17 | T5-003 | Texture features: GLCM contrast and entropy | Sushil | Yashwant | 3/3 | **PASS** |
| 18 | T5-007 | Shape descriptors: compactness, solidity, aspect ratio | Yashwant | Sushil | 3/3 | **PASS** |
| 19 | T5-010 | Instance segmentation for per-leaf prediction | Sushil | Yashwant | 3/3 | **PASS** |
| 20 | T5-012 | Weak labels and instance-level prediction | Yashwant | Sushil | 3/3 | **PASS** |
| 21 | T5-017 | Convolution, pooling, and backpropagation | Sushil | Yashwant | 3/3 | **PASS** |
| 22 | T5-018 | Data augmentation (training only) and input normalisation | Yashwant | Sushil | 3/3 | **PASS** |
| 23 | T5-020 | Optimisers, learning rate, and early stopping | Sushil | Yashwant | 3/3 | **PASS** |
| 24 | T5-022 | Transfer learning and fine-tuning (EfficientNetB0) | Yashwant | Sushil | 3/3 | **PASS** |
| 25 | T5-031 | Softmax confidence, calibration, and uncertainty thresholds | Sushil | Yashwant | 3/3 | **PASS** |

### Acceptance Criteria Checklist
- [x] Every Phase 5 concept has a note and a teach-back entry for both members
- [x] Self-check results are recorded honestly (gaps documented as follow-up items)
- [x] Both members tick the phase gate

### Signatures
- **Driver (ML Engineer):** *Yashwant* — Date: 2026-10-29
- **Navigator (System / QA):** *Sushil* — Date: 2026-10-29
