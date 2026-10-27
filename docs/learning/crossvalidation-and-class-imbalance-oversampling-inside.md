# Concept: Cross-Validation and Class Imbalance (Oversampling Inside Folds)

- **Concept File:** `docs/learning/crossvalidation-and-class-imbalance-oversampling-inside.md`
- **Related Task:** T4-007
- **Driver:** Yashwant
- **Author / Navigator:** Sushil
- **Date:** 2026-10-27
- **Status:** Completed

---

## 1. Intuition

In machine learning workflows with imbalanced class distributions (such as agricultural stress diagnostics where rare anomalies like `pH_Low` appear in less than 2% of telemetry rows), practitioners often use oversampling techniques (such as random oversampling, SMOTE, or class replication) to prevent the classifier from collapsing to the majority class.

However, a subtle yet catastrophic form of **data leakage** occurs when oversampling is performed on the entire dataset *prior* to cross-validation or hyperparameter search (e.g., calling `oversample(df)` and then passing the balanced dataframe to `cross_val_score` or `GridSearchCV`).

### The Pre-Fold Oversampling Flaw
1. When a minority class sample $x_i$ is replicated 25 times to balance the dataset before partitioning, all 25 copies are identical physical observations.
2. During $k$-fold cross-validation or train/val splitting, rows are partitioned randomly or even grouped by row index.
3. Inevitably, several identical copies of sample $x_i$ land in the **training fold**, while other exact copies of the exact same sample $x_i$ land in the **validation fold**.
4. The classifier memorizes sample $x_i$ during training and is then evaluated on the exact duplicate $x_i$ during validation.
5. Cross-validation reports inflated, near-perfect macro-F1 (e.g., 99.8%), concealing severe real-world failure. When deployed in production on unseen days, the model fails because it never learned generalizable decision boundaries.

### The Correct Protocol: Oversample Inside Each Fold Only
To ensure scientifically valid cross-validation:
- Cross-validation must begin with **unresampled, real observations**.
- Oversampling and preprocessing (scaling, imputation) must take place **strictly inside each cross-validation training fold**.
- The validation fold must remain an untouched, real, out-of-sample distribution.

```
       RAW DATASET (Unresampled, Real Rows)
                     │
         [GroupKFold / GuardedSplit]
         ┌───────────┴───────────┐
         ▼                       ▼
    Train Fold              Val Fold
   (Real Rows)            (Real Rows)
         │                       │
 [Oversample Train Only]         │  <-- NEVER TOUCH VAL FOLD
         │                       │
         ▼                       │
 Balanced Train Fold             │
         │                       │
   [Fit Model]                   │
         │                       │
         └───────────┬───────────┘
                     ▼
           [Evaluate on Val Fold]
                     ▼
          Honest, Unbiased Metrics
```

---

## 2. Mathematical Formulation & Formal Definition

### 2.1 Empirical Risk and Support Sets
Let $\mathcal{D} = \{(x_i, y_i, g_i)\}_{i=1}^N$ be the raw dataset where $x_i \in \mathbb{R}^d$ is the feature vector, $y_i \in \{1, \dots, C\}$ is the class label, and $g_i \in \mathcal{G}$ is the physical group identifier (e.g., calendar day or tank batch).

For a $K$-fold cross-validation scheme, let the sample indices be partitioned into $K$ disjoint validation index sets $\{V_k\}_{k=1}^K$ such that $\bigcup_{k=1}^K V_k = \{1, \dots, N\}$ and $V_k \cap V_j = \emptyset$ for $k \neq j$.

The corresponding training set for fold $k$ is $T_k = \{1, \dots, N\} \setminus V_k$.

### 2.2 The Zero Group Overlap Invariant
Under grouped cross-validation, physical group keys must be mutually disjoint between training and validation folds:

$$
\mathcal{G}(T_k) \cap \mathcal{G}(V_k) = \emptyset, \quad \forall k \in \{1, \dots, K\}
$$

Where $\mathcal{G}(S) = \{g_i : i \in S\}$. Any non-empty intersection $\mathcal{G}(T_k) \cap \mathcal{G}(V_k) \neq \emptyset$ violates temporal independence and constitutes leakage.

### 2.3 In-Fold Oversampling Operator
Let $\mathcal{O}_\theta: \mathcal{X} \times \mathcal{Y} \to \mathcal{X}' \times \mathcal{Y}'$ denote an oversampling operator parameterized by random seed $\theta$.

- **Flawed Workflow (Leaked):**
  $$
  (\mathcal{D}_{\text{bal}}, \cdot) = \mathcal{O}_\theta(\mathcal{D})
  $$
  $$
  \hat{R}_{\text{leaked}} = \frac{1}{K} \sum_{k=1}^K \mathcal{L}\left( f_{\text{train}(T_k^{\text{bal}})}, V_k^{\text{bal}} \right)
  $$
  Here, $\text{Support}(T_k^{\text{bal}}) \cap \text{Support}(V_k^{\text{bal}}) \neq \emptyset$.

- **Rigorous In-Fold Workflow (Unbiased):**
  $$
  T_k^{\text{bal}} = \mathcal{O}_{\theta_k}(T_k)
  $$
  $$
  \hat{R}_{\text{honest}} = \frac{1}{K} \sum_{k=1}^K \mathcal{L}\left( f_{\text{train}(T_k^{\text{bal}})}, V_k \right)
  $$
  Here, $V_k$ contains strictly real, unresampled rows, and no sample in $V_k$ has an identical twin in $T_k^{\text{bal}}$.

---

## 3. From-Scratch Implementation (NumPy / Pure Python)

A clean demonstration showing how in-fold oversampling is implemented using pure NumPy:

```python
import numpy as np


def oversample_in_fold_from_scratch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes inside the training fold only."""
    classes, counts = np.unique(y_train, return_counts=True)
    target_count = int(np.max(counts))
    rng = np.random.RandomState(seed)

    resampled_x = []
    resampled_y = []

    for cls, count in zip(classes, counts):
        cls_mask = y_train == cls
        cls_x = X_train[cls_mask]
        cls_y = y_train[cls_mask]

        if count < target_count:
            draw_idx = rng.choice(count, size=target_count - count, replace=True)
            extra_x = cls_x[draw_idx]
            extra_y = cls_y[draw_idx]
            cls_x = np.vstack([cls_x, extra_x])
            cls_y = np.concatenate([cls_y, extra_y])

        resampled_x.append(cls_x)
        resampled_y.append(cls_y)

    full_x = np.vstack(resampled_x)
    full_y = np.concatenate(resampled_y)

    # Deterministic shuffle
    perm = rng.permutation(len(full_x))
    return full_x[perm], full_y[perm]


if __name__ == "__main__":
    # Toy verification run: 10 Normal, 2 pH_Low
    X_train = np.arange(12).reshape(-1, 1)
    y_train = np.array(["Normal"] * 10 + ["pH_Low"] * 2)

    X_bal, y_bal = oversample_in_fold_from_scratch(X_train, y_train, seed=42)
    cls, cnts = np.unique(y_bal, return_counts=True)
    print("Balanced classes:", dict(zip(cls, cnts)))
    assert (cnts == 10).all(), "Every class must reach majority count (10)!"
    print("Toy verification PASSED.")
```

---

## 4. Production & Library Usage

In production code (`ml/evaluation/cv.py`), this principle is enforced through:

1. `GuardedGroupKFold`: Wraps scikit-learn's `GroupKFold` and asserts that no group ID appears in both folds.
2. `get_fold_data`: Takes raw data, extracts the fold slices, asserts zero group overlap, balances the training partition, and leaves the validation slice completely untouched.
3. `evaluate_grouped_cv`: Clones the estimator and scaler, fits both solely on the training fold, and predicts on the untouched validation fold.

```python
from ml.evaluation import GuardedGroupKFold, evaluate_grouped_cv
from sklearn.tree import DecisionTreeClassifier

# Run grouped 5-fold CV with in-fold oversampling
results = evaluate_grouped_cv(
    estimator=DecisionTreeClassifier(max_depth=4, random_state=42),
    X=X_raw_train,
    y=y_raw_train,
    groups=calendar_groups,
    labels=["Normal", "EC_High", "pH_Low", "WaterTemp_High"],
    n_splits=5,
    oversample_train=True,
    random_state=42,
)

print(f"Validation Macro-F1: {results['mean_macro_f1']:.4f} +/- {results['std_macro_f1']:.4f}")
```

---

## 5. Hydroponic Application Context & Decisions

In Hydroponic Nutrition AI:
1. **Lettuce Sensor Telemetry:** Real training pool contains 430 real rows across 41 calendar days. Minority class `pH_Low` has only 7 real rows. Naive pre-split oversampling replicates those 7 rows across folds, falsely showing 100% precision. In-fold oversampling ensures that when a day is held out, any evaluation of `pH_Low` tests true out-of-sample generalization.
2. **Spinach Sensor Telemetry:** 6-hour contiguous blocks per bench are grouped together. Oversampling is avoided because classes are natural, but grouped folds ensure time-block independence.
3. **Tomato Sensor Telemetry:** Calendar months are held out whole.
4. **ADR 0006 Compliance:** ADR 0006 mandates this exact in-fold oversampling protocol across Phase 4 model training tasks (T4-010 to T4-025).

---

## 6. Self-Check Questions & Answers

### Q1: What happens if you oversample a minority class prior to cross-validation?
**Answer:** The duplicated minority samples are distributed across both the training fold and the validation fold. The model memorizes the training duplicate and is evaluated on an identical copy in the validation fold, leading to severe duplicate-sample data leakage and wildly inflated validation scores.

### Q2: Why must feature scalers (e.g. `StandardScaler`) also be fit inside each cross-validation fold?
**Answer:** If `StandardScaler` is fit on the full dataset before splitting, the calculated mean ($\mu$) and standard deviation ($\sigma$) contain information about the validation fold's feature values. This causes distribution leakage. The scaler must be fit exclusively on $X_{\text{train\_fold}}$ and merely applied (`transform`) to $X_{\text{val\_fold}}$.

### Q3: How does `GuardedGroupKFold` prevent temporal autocorrelation leakage?
**Answer:** Sensor readings logged minutes apart share ambient conditions and nutrient concentrations. Standard random $K$-fold scatters adjacent readings across folds, enabling the model to interpolate. `GuardedGroupKFold` groups all readings from the same calendar day or batch into a single partition and asserts that $G(T_k) \cap G(V_k) = \emptyset$.

### Q4: In `ml/evaluation/cv.py`, what guarantees that the validation fold is never oversampled?
**Answer:** The `get_fold_data` function explicitly slices `X_val` and `y_val` directly from the raw dataset, applies zero modifications, and applies `oversample_minority_classes` exclusively to the training fold variables `X_train` and `y_train`.
