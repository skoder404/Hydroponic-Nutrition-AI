# Concept: Decision Tree Hyperparameters, Depth, and Overfitting

- **Concept File:** `docs/learning/decision-tree-hyperparameters-depth-and-overfitting.md`
- **Related Task:** T4-011
- **Driver:** Sushil
- **Author / Navigator:** Yashwant
- **Date:** 2026-10-27
- **Status:** Completed

---

## 1. Intuition

Decision trees are inherently **low-bias, high-variance** learning algorithms. Because they place no structural assumptions on the underlying data distribution (such as linearity or Gaussian noise), an unconstrained decision tree has infinite capacity: it will continuously split the feature space until every leaf contains a single sample, achieving 100% training accuracy.

In hydroponic nutrient monitoring, telemetry data is contaminated by physical noise:
- Sensor drift and ADC quantization errors.
- Micro-fluctuations in electrical conductivity (EC) during peristaltic dosing pump activation.
- Diurnal temperature swings and transient bubble artifacts across probe surfaces.

When a decision tree is allowed to grow to arbitrary depth (e.g., `max_depth=None`), it does not merely discover agronomic boundaries (e.g. "EC > 2.5 causes toxicity"). Instead, it begins isolating single anomalous data points with absurdly specific compound conditions:
> *If air_temp $\le 21.34^\circ\text{C}$ AND humidity $> 68.2\%$ AND pH $\le 5.41$ AND EC $= 1.83 \rightarrow$ class is Nitrogen Deficiency.*

Such narrow leaves memorize the exact transient state of a single sensor glitch on a single afternoon. When presented with unseen data on a different day, this rule is meaningless. To ensure models generalize, we regularize trees by constraining their capacity through explicit structural hyperparameters (pre-pruning) or cost-complexity pruning (post-pruning).

---

## 2. Mathematical Formulation & Formal Definition

### 2.1 The Bias-Variance Tradeoff in Decision Trees
Let $f_D(x)$ be the decision tree trained on a training dataset $D$ drawn from distribution $P(X, Y)$, predicting continuous label distribution $y$. The expected generalization error decomposes into:

$$
\mathbb{E}_D \left[ (y - f_D(x))^2 \right] = \text{Bias}^2(f(x)) + \text{Var}_D(f_D(x)) + \sigma_{\text{irreducible}}^2
$$

- **Shallow Trees (Low Depth / High Regularization):**
  - High Bias: Incapable of capturing complex non-linear interactions across multiple nutrients. Underfits.
  - Low Variance: Minor changes to the training set $D$ produce almost identical top-level splits.
- **Deep Trees (High Depth / No Regularization):**
  - Low Bias: Can approximate any piecewise-constant function on feature space.
  - Extreme Variance: Adding or modifying just 2–3 training points near a boundary can completely invert the tree structure below the root node.

### 2.2 Cost-Complexity Pruning (Minimal Cost-Complexity Criterion)
To balance tree size $|T|$ (number of terminal leaf nodes) and empirical misclassification cost $R(T)$, CART introduces a complexity parameter $\alpha \ge 0$:

$$
R_\alpha(T) = R(T) + \alpha |T|
$$

where:
- $R(T) = \sum_{m \in \text{leaves}(T)} \frac{N_m}{N} I(m)$ is the total weighted training impurity of tree $T$.
- $|T|$ is the number of terminal leaves in tree $T$.
- $\alpha$ is the regularization penalty per additional leaf node.

As $\alpha$ increases from $0$ to $\infty$, small branches that provide negligible reduction in training impurity relative to their structural complexity are sequentially collapsed into leaves (pruning).

### 2.3 Leaf Size and Splitting Constraints
Instead of post-pruning via $\alpha$, production pipelines predominantly apply **pre-pruning** constraints:
1. **Minimum Samples per Leaf ($N_{\text{min, leaf}}$):** A candidate split $\theta = (j, t)$ is valid if and only if both child nodes satisfy:
   
   $$
   N_L \ge N_{\text{min, leaf}} \quad \text{and} \quad N_R \ge N_{\text{min, leaf}}
   $$

2. **Minimum Samples to Split ($N_{\text{min, split}}$):** Node $m$ can only be split if:
   
   $$
   N_m \ge N_{\text{min, split}}
   $$

---

## 3. From-Scratch Implementation (NumPy / Scikit-Learn Experiment)

The following experiment evaluates the relationship between tree depth, training accuracy, and held-out validation accuracy to illustrate the exact onset of overfitting:

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def demonstrate_overfitting_curve():
    # Generate synthetic hydroponic sensor dataset with 10 features and noise
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=3,
        flip_y=0.10,  # 10% label noise (sensor miscalibration)
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    depths = range(1, 16)
    train_scores = []
    val_scores = []

    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        train_scores.append(dt.score(X_train, y_train))
        val_scores.append(dt.score(X_val, y_val))

    best_depth = depths[int(np.argmax(val_scores))]
    print(f"Optimal max_depth: {best_depth}")
    print(f"Train score at max_depth={best_depth}: {train_scores[best_depth-1]:.4f}")
    print(f"Validation score at max_depth={best_depth}: {val_scores[best_depth-1]:.4f}")
    print(f"Deepest tree (depth 15) Train: {train_scores[-1]:.4f}, Val: {val_scores[-1]:.4f}")

    assert train_scores[-1] > val_scores[-1] + 0.15, "Expected severe overfitting at depth 15"
    return best_depth


if __name__ == "__main__":
    best_d = demonstrate_overfitting_curve()
    print(f"[PASS] Successfully demonstrated bias-variance tradeoff (Optimal Depth = {best_d}).")
```

---

## 4. Production & Library Usage

In this project, `DecisionTreeClassifier` is configured with strict regularization to maintain interpretable white-box rules:

```python
from sklearn.tree import DecisionTreeClassifier, export_text

dt_reg = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,              # Restricts maximum rule depth to 5 hierarchical steps
    min_samples_leaf=5,       # Each leaf must represent at least 5 sensor observations
    min_samples_split=10,     # Prohibits splitting small clusters
    max_leaf_nodes=16,        # Explicit ceiling on total rules in system
    class_weight="balanced",  # Adjusts for rare nutrient deficiencies
    random_state=42,
)

dt_reg.fit(X_train, y_train)
exported_rules = export_text(dt_reg, feature_names=list(feature_cols))
```

**Key Hyperparameter Interaction Matrix:**
| Hyperparameter | Low Value Effect | High Value Effect | Project Recommended Range |
|---|---|---|---|
| `max_depth` | High bias (underfitting, simple rules) | High variance (memorizes sensor spikes) | `3` to `6` |
| `min_samples_leaf` | Overfitting (allows 1-sample leaves) | Smoothing (prevents micro-leaves) | `3` to `10` |
| `min_samples_split` | Splitting continues into micro-nodes | Early stopping of node expansion | `5` to `20` |
| `ccp_alpha` | No post-pruning ($\alpha=0$) | Aggressive tree pruning ($\alpha > 0.05$) | `0.0` to `0.01` |

---

## 5. Failure Modes, Edge Cases & Debugging

1. **The Single-Sample Leaf Trap:**
   - *Failure:* If `min_samples_leaf=1` and `max_depth=None`, an outlier reading (e.g. air temp 55°C due to probe drop) becomes a permanent rule.
   - *Mitigation:* Always enforce `min_samples_leaf >= 5`.
2. **False Confidence from Rule Simplicity:**
   - *Failure:* A very shallow tree (`max_depth=2`) looks elegant and clean to domain experts, but conflates compound nutrient stressors (e.g. Potassium deficiency masked by low EC).
   - *Mitigation:* Evaluate per-class recall and macro-F1, not just overall accuracy.
3. **Threshold Sensitivity to Calibration Drift:**
   - *Failure:* If pH split is $x_{\text{pH}} \le 5.75$, a 0.05 pH electrode calibration drift shifts thousands of predictions across the decision boundary.
   - *Mitigation:* In downstream scoring, apply soft probability thresholds and uncertainty intervals rather than hard step cutoffs.

---

## 6. Where It Appears in This Project

- **Trainer Script:** [`scripts/train_decision_tree.py`](../../scripts/train_decision_tree.py) — applies grid search over `max_depth` and `min_samples_leaf`.
- **Validation Tuning:** [`scripts/tune_sensor_models.py`](../../scripts/tune_sensor_models.py) — selects regularized hyperparameter sets using GroupKFold cross-validation.
- **Promoted Artifacts:**
  - `lettuce_sensor_decision_tree_v1.0.0` (`max_depth=5`, `criterion='entropy'`)
  - `spinach_sensor_decision_tree_v1.0.0` (`max_depth=4`, `criterion='gini'`)

---

## 7. Self-Check Questions (5 Questions)

1. **Q1:** Why does training accuracy monotonically increase as `max_depth` increases, while validation accuracy eventually declines?
   - *Answer:* Training accuracy increases because each split strictly cannot increase training loss; deeper trees subdivide space into finer cells. Validation accuracy declines once cells become smaller than the natural feature noise variance, fitting sample-specific noise rather than the true data-generating distribution.
2. **Q2:** If a decision tree has `max_depth=4`, what is the theoretical maximum number of leaf nodes it can create?
   - *Answer:* A complete binary tree of depth $D$ has at most $2^D$ leaves. For $D=4$, the maximum number of leaves is $2^4 = 16$.
3. **Q3:** How does setting `min_samples_leaf=10` act as an implicit regularizer?
   - *Answer:* It prevents the tree from creating any split that leaves fewer than 10 training instances in either child node. This forces all leaf predictions to be supported by at least 10 observations, preventing memorization of isolated noise points.
4. **Q4:** What is the role of the $\alpha$ parameter in Minimal Cost-Complexity Pruning ($R_\alpha(T) = R(T) + \alpha |T|$)?
   - *Answer:* $\alpha$ acts as a penalty weight on the number of leaves $|T|$. When $\alpha=0$, the tree is unpruned. As $\alpha$ increases, branches are pruned unless their reduction in impurity $R(T)$ exceeds $\alpha \cdot \Delta |T|$.
5. **Q5:** Why is tree pruning especially critical before deploying models to IoT microcontroller endpoints?
   - *Answer:* Unpruned trees have large binary footprints, consume excessive cache/RAM, and execute dozens of pointer lookups per sample. A regularized shallow tree executes in microsecond constant-time nested if-statements with zero risk of stack overflow.
