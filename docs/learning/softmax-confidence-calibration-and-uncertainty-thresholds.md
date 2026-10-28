# Concept: Softmax Confidence, Calibration, and Uncertainty Thresholds

- **Concept File:** `docs/learning/softmax-confidence-calibration-and-uncertainty-thresholds.md`
- **Related Task:** T5-031
- **Driver:** Sushil
- **Author / Navigator:** Yashwant
- **Date:** 2026-10-28
- **Status:** Completed

---

## 1. Intuition

Modern deep convolutional neural networks (such as EfficientNetB0 or ResNet50) typically terminate with a linear projection layer producing unconstrained real-valued logits $z \in \mathbb{R}^K$, followed by a **softmax operator**:

$$
\sigma(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^K \exp(z_j)}
$$

Because $\sum_i \sigma(z)_i = 1$ and $\sigma(z)_i \in (0, 1)$, practitioners frequently treat $\max_i \sigma(z)_i$ as the model's "probability" of being correct. **This interpretation is fundamentally flawed.**

As demonstrated by Guo et al. (2017, *On Calibration of Modern Neural Networks*), modern deep architectures are notoriously **overconfident**: a network may output a softmax score of $0.98$ on a sample for which its empirical historical accuracy is only $72\%$. This overconfidence stems from:
1. **Depth and Capacity:** Deep residual networks easily minimize cross-entropy loss by pushing logit magnitudes $\|z\|$ to large positive values, driving softmax values toward $1.0$ even on borderline samples.
2. **Weight Decay and Absence of Regularization on Softmax Saturation:** Cross-entropy loss incentivizes unbounded logit separation, leading to saturated output layers.
3. **Foliar Label Inheritance in Hydroponics:** In automated agricultural image diagnosis, plants are induced with specific nutrient regimes (e.g. Nitrogen starvation). During dataset collection, all segmented leaves harvested from that plant inherit the label `NitrogenDeficiency`. However, young emergent leaves or lower leaves may exhibit zero visible chlorosis at early stress stages. An individual leaf that looks completely healthy may be labelled `NitrogenDeficiency`. Consequently, expecting an image model to produce a perfectly deterministic 1.0 probability on every isolated leaf ignores underlying biological ground-truth ambiguity.

To build safe agronomic AI, we must:
- Calibrate model probabilities so confidence scores align with true empirical accuracy.
- Establish explicit operational **uncertainty thresholds** (High, Moderate, Uncertain) derived strictly from validation data.
- Clearly state score semantics in metadata and UI to avoid misleading growers.

---

## 2. Mathematical Formulation & Calibration

### 2.1 The Definition of Perfect Calibration
Let $X \in \mathcal{X}$ be an input leaf image, $Y \in \{1, \dots, K\}$ be the true diagnosis label, $\hat{Y} = \arg\max_k \sigma(z(X))_k$ be the predicted class, and $\hat{P} = \max_k \sigma(z(X))_k$ be the associated confidence score.

A model is **perfectly calibrated** if its confidence matches the true probability of correctness for all confidence levels $p \in [0, 1]$:

$$
\mathbb{P}\left(\hat{Y} = Y \mid \hat{P} = p\right) = p, \quad \forall p \in [0, 1]
$$

For instance, among all leaves where the model outputs confidence $\hat{P} = 0.80$, exactly $80\%$ of them should be correctly classified.

### 2.2 Expected Calibration Error (ECE)
To measure calibration empirically, the continuous interval $[0, 1]$ is partitioned into $M$ equally spaced bins $B_1, B_2, \dots, B_M$. For our benchmark, $M=10$, with boundaries $[0.0, 0.1), [0.1, 0.2), \dots, [0.9, 1.0]$.

For each bin $B_m$:
- **Bin Accuracy:**
  $$
  \text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)
  $$
- **Bin Average Confidence:**
  $$
  \text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i
  $$

The **Expected Calibration Error (ECE)** is the weighted average absolute difference between accuracy and confidence across all bins:

$$
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

### 2.3 Post-Processing Calibration: Temperature Scaling
**Temperature scaling** is a single-parameter extension of Platt scaling applied directly to logits before the softmax layer. For unnormalized logits $z_i \in \mathbb{R}^K$:

$$
q_i(T) = \frac{\exp(z_{i, k} / T)}{\sum_{j=1}^K \exp(z_{i, j} / T)}
$$

Where $T > 0$ is the learned temperature parameter:
- If $T > 1$: Logits are softened, pulling overconfident softmax outputs downward toward a uniform distribution.
- If $T < 1$: Logits are sharpened.
- If $T = 1$: Standard raw softmax is recovered.

#### Fundamental Mathematical Property: Invariance of Model Predictions
Because scalar division by $T > 0$ is a strictly monotonic transformation:
$$
\arg\max_k \frac{z_{i, k}}{T} = \arg\max_k z_{i, k}
$$
**Temperature scaling does not alter class predictions, accuracy, precision, recall, or Macro-F1.** It exclusively recalibrates the confidence score magnitude to match true empirical frequency.

#### Optimization Objective (Fit on Validation Split Only)
In strict compliance with TECH_RULES Rule 3 (fit-on-train/val-only, never test), $T$ is optimized on the **isolated validation partition** by minimizing Negative Log-Likelihood (NLL / Cross-Entropy Loss):

$$
\min_{T > 0} \; -\frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \log \left( \frac{\exp(z_{i, y_i} / T)}{\sum_{j=1}^K \exp(z_{i, j} / T)} \right)
$$

---

## 3. Operational Uncertainty Thresholds

Even with temperature calibration, individual predictions carry varying levels of ambiguity. In hydroponic production, recommending aggressive chemical dosing (e.g. dumping $500\text{ mL}$ of concentrated phosphoric acid into a nutrient reservoir) based on a low-confidence or ambiguous diagnosis can destroy an entire crop canopy within hours.

We define three standardized operational bands derived from validation distributions:

```
[0.0 ──────────────── Moderate Threshold ────────────── High Threshold ─────────────── 1.0]
         Uncertain Band               Moderate Band                High Confidence Band
      (Human Review Flag)          (Secondary Confirm)            (Automated Guidance)
```

1. **High Confidence Band ($s \ge \theta_{\text{high}}$):**
   - Empirical validation accuracy $\ge 90\%$.
   - The symptom manifestation strongly aligns with canonical training exemplars. Automated dosing and adjustments may proceed.
2. **Moderate Confidence Band ($\theta_{\text{mod}} \le s < \theta_{\text{high}}$):**
   - Empirical validation accuracy $\ge 70\%$.
   - Plausible deficiency signs, but subtle co-occurring symptoms or early onset. Recommendation is flagged with a caution chip.
3. **Uncertain Band ($s < \theta_{\text{mod}}$):**
   - High entropy distribution across classes.
   - The platform abstains from automated action, shows an "Uncertain" badge, and prompts the grower for manual inspection or secondary sensor verification.

---

## 4. Empirical Implementation & Verification

In `scripts/derive_uncertainty_thresholds.py`:
- All validation leaf samples are ingested through the promoted `EfficientNetB0` models for Lettuce ($N=128$), Spinach ($N=2,745$), and Tomato ($N=5,034$).
- Logits $z$ and targets $y$ are collected under `torch.no_grad()`.
- $T$ is optimized using `scipy.optimize.minimize_scalar` within bounds $[0.05, 10.0]$.
- Pre-scaling ECE and post-scaling ECE are computed and verified.
- Thresholds are stored directly in each model's `metadata.json` under `uncertainty_thresholds` and `score_semantics`.
- Visual reliability diagrams are generated and saved to `reports/calibration/`.

