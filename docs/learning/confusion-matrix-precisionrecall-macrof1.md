# Concept: Multi-Class Confusion Matrix, Precision, Recall, and Macro-F1

- **Concept File:** `docs/learning/confusion-matrix-precisionrecall-macrof1.md`
- **Related Task:** T4-004
- **Driver:** Sushil
- **Author / Navigator:** Yashwant
- **Date:** 2026-10-26
- **Status:** Completed
- **Companion Notebook:** [`notebooks/03_metrics_from_scratch.ipynb`](../../notebooks/03_metrics_from_scratch.ipynb)

---

## 1. Intuition

In classical machine learning evaluation, **overall accuracy** (fraction of correct predictions out of all predictions) is the most intuitive metric. However, in **precision agriculture, environmental monitoring, and medical diagnosis**, accuracy is notoriously deceptive due to severe **class imbalance**.

In a hydroponic system, a crop operates within normal healthy physiological boundaries 95%+ of the time. Critical stress events—such as acute nutrient solution acidification (`pH_Low`), electrical conductivity toxicity (`EC_High_Toxicity`), or root rot inducing water temperatures (`Water_Temp_High`)—are comparatively rare anomalies. 

### The Accuracy Paradox
Suppose a commercial tomato greenhouse dataset contains 1,000 hourly sensor readings:
- 960 readings are healthy (`Normal`).
- 40 readings represent acute, root-killing acidity (`pH_Low`).

A completely defective or naive model that predicts `Normal` 100% of the time achieves **96.0% overall accuracy**. A manager glancing solely at accuracy would celebrate an ostensibly high-performing model. In reality, **every single crop failure was missed** ($0\%$ detection rate), leading to irreversible root death and complete crop loss.

To evaluate diagnostic systems honestly, we must inspect:
1. **Confusion Matrix:** The full contingency table separating actual conditions from predicted classifications.
2. **Precision (Quality of Alarms):** When the model sounds an alarm for a specific fault, how often is it actually that fault? (Minimizes false alarms and unnecessary chemical dosing).
3. **Recall / Sensitivity (Quantity of Detections):** Of all actual physical stress events, what fraction did the model successfully catch? (Minimizes missed failures).
4. **F1-Score:** The harmonic mean balancing precision and recall.
5. **Macro-Averaging:** Computing metrics independently per class and taking the unweighted arithmetic mean. Under macro-averaging, a rare failure category like `pH_Low` contributes identically ($1/K$) to the final score as the dominant `Normal` class, instantly unmasking lazy majority-class models.

---

## 2. Mathematical Formulation & Formal Definition

Let $K$ be the number of discrete classes in an ordered label vocabulary $\mathcal{C} = \{c_1, c_2, \dots, c_K\}$. Let $y_{\text{true}}, y_{\text{pred}} \in \{1, \dots, K\}^N$ be the ground truth and predicted class labels for $N$ samples.

### 2.1 Multi-Class Confusion Matrix
The confusion matrix $C \in \mathbb{N}_0^{K \times K}$ is defined such that element $C_{i, j}$ is the number of observations known to be in group $i$ and predicted to be in group $j$:

$$
C_{i, j} = \sum_{n=1}^N \mathbb{I}(y_{\text{true}, n} = c_i \land y_{\text{pred}, n} = c_j)
$$

Convention (standard scikit-learn convention):
- **Rows ($i$):** Actual true classes.
- **Columns ($j$):** Predicted classes.
- **Main Diagonal ($C_{i, i}$):** Correct predictions (True Positives for class $c_i$).

### 2.2 Per-Class Confusion Terms
For each class $c_i$:
- **True Positives ($TP_i$):** Correctly predicted instances of class $c_i$:
  $$TP_i = C_{i, i}$$
- **False Positives ($FP_i$):** Instances of other classes incorrectly predicted as class $c_i$:
  $$FP_i = \sum_{k \neq i} C_{k, i} = \text{ColumnSum}_i - C_{i, i}$$
- **False Negatives ($FN_i$):** Actual instances of class $c_i$ incorrectly predicted as another class:
  $$FN_i = \sum_{k \neq i} C_{i, k} = \text{RowSum}_i - C_{i, i}$$
- **True Negatives ($TN_i$):** Instances of other classes correctly predicted as not class $c_i$:
  $$TN_i = N - (TP_i + FP_i + FN_i)$$
- **Support ($S_i$):** Ground truth occurrences of class $c_i$:
  $$S_i = \sum_{j=1}^K C_{i, j} = TP_i + FN_i$$

### 2.3 Precision, Recall, and F1-Score
For each class $c_i$:

$$
\text{Precision}_i = \frac{TP_i}{TP_i + FP_i}
$$

$$
\text{Recall}_i = \frac{TP_i}{TP_i + FN_i} = \frac{TP_i}{S_i}
$$

$$
F1_i = 2 \times \frac{\text{Precision}_i \times \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i} = \frac{2 TP_i}{2 TP_i + FP_i + FN_i}
$$

### 2.4 Multi-Class Averaging Strategies

#### Macro-Averaging (Unweighted Arithmetic Mean)
Every class is weighted equally, regardless of its sample support $S_i$:

$$
\text{Macro-Precision} = \frac{1}{K} \sum_{i=1}^K \text{Precision}_i
$$

$$
\text{Macro-Recall} = \frac{1}{K} \sum_{i=1}^K \text{Recall}_i
$$

$$
\text{Macro-F1} = \frac{1}{K} \sum_{i=1}^K F1_i
$$

#### Weighted-Averaging (Support-Weighted Mean)
Each class metric is weighted by its prevalence in the ground truth:

$$
\text{Weighted-F1} = \sum_{i=1}^K \frac{S_i}{N} F1_i
$$

*Warning:* Weighted-F1 suffers from the same vulnerability as accuracy under severe imbalance, as majority classes dominate the weighted sum.

#### Micro-Averaging
Aggregates total $TP, FP, FN$ across all classes before computing metrics:

$$
\text{Micro-Precision} = \text{Micro-Recall} = \text{Micro-F1} = \text{Overall Accuracy} = \frac{\sum_{i=1}^K TP_i}{N}
$$

### 2.5 Zero-Division Handling
When a class is never predicted by a model ($TP_i + FP_i = 0$), the denominator for $\text{Precision}_i$ is zero. Similarly, when a class has no positive instances in the test partition ($S_i = TP_i + FN_i = 0$), $\text{Recall}_i$ has a zero denominator.

By standard convention (scikit-learn `zero_division=0.0`):
$$
\text{If } (TP_i + FP_i = 0) \implies \text{Precision}_i = 0.0
$$
$$
\text{If } (TP_i + FN_i = 0) \implies \text{Recall}_i = 0.0
$$
$$
\text{If } (\text{Precision}_i + \text{Recall}_i = 0.0) \implies F1_i = 0.0
$$

---

## 3. From-Scratch Implementation (NumPy / Pure Python)

Below is the standalone reference implementation implementing multi-class confusion matrix, per-class metrics, and macro-averaged metrics in pure NumPy:

```python
import numpy as np


def confusion_matrix_scratch(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    labels: list | None = None,
) -> tuple[np.ndarray, list]:
    """Compute multi-class confusion matrix in pure NumPy."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_pred must have identical length.")

    ordered_labels = list(labels) if labels is not None else sorted(set(y_t).union(set(y_p)))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(ordered_labels)}
    k = len(ordered_labels)
    cm = np.zeros((k, k), dtype=np.int64)

    for t, p in zip(y_t, y_p):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    return cm, ordered_labels


def calculate_metrics_scratch(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    labels: list | None = None,
    zero_division: float = 0.0,
) -> dict:
    """Compute accuracy, per-class precision/recall/F1/support, and macro metrics."""
    cm, ordered_labels = confusion_matrix_scratch(y_true, y_pred, labels=labels)
    k = len(ordered_labels)

    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)
    n = len(y_true)

    prec_denom = tp + fp
    precision = np.divide(tp, prec_denom, out=np.full(k, zero_division), where=prec_denom > 0)

    rec_denom = tp + fn
    recall = np.divide(tp, rec_denom, out=np.full(k, zero_division), where=rec_denom > 0)

    f1_denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, f1_denom, out=np.full(k, zero_division), where=f1_denom > 0)

    accuracy = float(tp.sum()) / float(n) if n > 0 else zero_division
    macro_precision = float(np.mean(precision)) if k > 0 else zero_division
    macro_recall = float(np.mean(recall)) if k > 0 else zero_division
    macro_f1 = float(np.mean(f1)) if k > 0 else zero_division

    per_class = {
        ordered_labels[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(k)
    }

    return {
        "confusion_matrix": cm,
        "labels": ordered_labels,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }
```

---

## 4. Production & Library Usage

In production scikit-learn pipelines, always pass `average="macro"` and `zero_division=0.0`:

```python
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Fixed class order must be enforced across all evaluation calls
CLASSES = ["Normal", "EC_High_Toxicity", "EC_Low_Deficiency", "pH_Imbalance", "Water_Temp_High"]

# 1. Confusion Matrix with explicit labels
cm = confusion_matrix(y_true, y_pred, labels=CLASSES)

# 2. Macro-F1 with explicit zero_division handling
macro_f1 = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0.0)

# 3. Comprehensive text report
report = classification_report(
    y_true,
    y_pred,
    labels=CLASSES,
    zero_division=0.0,
    digits=4,
)
```

**Key Parameters & Rationale:**
- `labels=CLASSES`: Crucial for multi-class problems. If a test fold happens to contain 0 instances of a rare class, omitting `labels` shrinks the matrix from $K \times K$ to $(K-1) \times (K-1)$, misaligning rows and corrupting tracking.
- `zero_division=0.0`: Avoids runtime `UndefinedMetricWarning` and prevents crashes or NaN propagations when a candidate model fails to predict a rare class.

---

## 5. Failure Modes, Edge Cases & Debugging

1. **Zero-Prediction Class ($FP = 0, TP = 0$):**
   - *Symptom:* Precision denominator is zero.
   - *Fix:* Must evaluate to `0.0`. If left unhandled, division by zero produces `NaN` or crashes hyperparameter search (`GridSearchCV`).
2. **Zero-Support Class in Split ($S_i = 0$):**
   - *Symptom:* Recall denominator is zero.
   - *Fix:* When a class has no true instances in a validation partition, its recall is set to `0.0` (or excluded from the macro average if evaluating conditional presence). In Hydroponic-Nutrition-AI, stratified or grouped partition manifests explicitly record support per class to ensure non-zero representation.
3. **Accuracy Masking Minority Faults:**
   - *Symptom:* Model shows 96% accuracy but high crop mortality.
   - *Diagnosis:* Compare accuracy against Macro-F1. A large gap ($\text{Accuracy} \gg \text{Macro-F1}$) is the hallmark diagnostic signature of minority class failure.
4. **Row vs Column Inversion:**
   - *Symptom:* Precision and recall appear transposed.
   - *Fix:* Standard scikit-learn confusion matrix maps actual rows to predicted columns ($C_{\text{true}, \text{pred}}$). Inverting this axes convention swaps precision and recall.

---

## 6. Where It Appears in This Project

- **Notebook Reference:** [`notebooks/03_metrics_from_scratch.ipynb`](../../notebooks/03_metrics_from_scratch.ipynb) — Executed empirical demonstration and verification against scikit-learn.
- **Evaluation Module:** [`ml/evaluation/metrics.py`](../../ml/evaluation/metrics.py) (Task T4-005) — Standard project evaluation function returning accuracy, macro metrics, per-class dictionaries, and confusion matrices.
- **Cross-Validation Objective:** Tasks T4-015, T4-017, T4-018 — Hyperparameter search optimizes `scoring="f1_macro"` rather than accuracy.
- **Sensor Validation ADR:** [`docs/decisions/0006-sensor-validation-protocol.md`](0006-sensor-validation-protocol.md) — Mandates Macro-F1 as the primary model selection criterion.

---

## 7. Self-Check Questions

1. **Q1: Why is overall accuracy an invalid primary metric for imbalanced crop stress classification?**  
   *Answer:* A trivial model predicting the majority healthy class (`Normal`) for 100% of samples can achieve high overall accuracy (e.g. 96%) while having 0% recall on critical stress events (`pH_Low`). It fails to diagnose any physical anomalies.

2. **Q2: In a confusion matrix, what do the rows and columns represent by convention?**  
   *Answer:* Rows represent the actual ground truth classes ($y_{\text{true}}$); columns represent the model's predicted classes ($y_{\text{pred}}$). Diagonal elements $C_{i, i}$ are True Positives.

3. **Q3: How does Macro-F1 differ from Weighted-F1 and Micro-F1?**  
   *Answer:* Macro-F1 calculates the F1-score independently for each class and takes an unweighted arithmetic average, giving equal importance to rare classes. Weighted-F1 weights each class F1 by its support $S_i / N$, which allows majority classes to overpower rare classes. Micro-F1 aggregates global TP, FP, FN and is mathematically identical to overall accuracy in multi-class classification.

4. **Q4: What happens mathematically when a classifier makes zero predictions for class $k$, and how does `zero_division=0.0` resolve it?**  
   *Answer:* When a class is never predicted, $TP_k = 0$ and $FP_k = 0$, making precision $0 / 0$ (undefined). Passing `zero_division=0.0` replaces the undefined operation with $0.0$, preventing runtime errors and ensuring the class correctly receives an F1-score of $0.0$.

5. **Q5: Why must the `labels` argument be explicitly provided to `confusion_matrix` and `precision_score` in production pipelines?**  
   *Answer:* If a validation batch or test fold is missing instances of a class, scikit-learn will infer classes only from the observed values, shrinking the matrix dimension from $K \times K$ to $(K-1) \times (K-1)$ and breaking downstream index alignment and tracking contracts.
