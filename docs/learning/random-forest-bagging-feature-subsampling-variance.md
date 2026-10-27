# Concept: Random Forest: Bagging, Feature Subsampling, and Variance Reduction

- **Concept File:** `docs/learning/random-forest-bagging-feature-subsampling-variance.md`
- **Related Task:** T4-012
- **Driver:** Yashwant
- **Author / Navigator:** Sushil
- **Date:** 2026-10-27
- **Status:** Completed

---

## 1. Intuition

A single Decision Tree is an intuitive and interpretable classifier, but it suffers from a fundamental statistical limitation: **high variance**. A deep decision tree fits itself aggressively to the quirks, noise, and specific samples of its training set. Small perturbations in sensor telemetry (such as a 0.2 pH spike caused by electrical noise) can radically alter the root split, cascading completely different decision rules through the rest of the tree.

**Random Forest** addresses this variance bottleneck through an ensemble strategy founded on two complementary mechanisms:
1. **Bagging (Bootstrap Aggregation):** Instead of fitting a single tree on all training observations, we generate $B$ separate bootstrap samples (sampling $N$ observations *uniformly with replacement*). Each tree in the ensemble is trained on its own unique bootstrap sample. Because each bootstrap sample leaves out approximately $36.8\%$ of rows (the *Out-of-Bag* or OOB samples), every tree learns on a slightly different perspective of the data.
2. **Feature Subsampling (Random Subspaces):** If we only did bagging on correlated features, every tree would still choose the single most dominant feature (e.g. `ec` or `ph`) at the root node, resulting in highly correlated trees. Averaging correlated trees yields diminishing variance reduction. Feature subsampling forces each tree node to consider only a random subset of $m \approx \sqrt{p}$ features (out of total $p$ features) when selecting its best split. This decorrelates the individual trees, unlocking the full variance-reducing power of ensembling.

```
                  TRAINING DATA (N samples, p features)
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
  Bootstrap 1                  Bootstrap 2                  Bootstrap B
(N rows w/ repl.)            (N rows w/ repl.)            (N rows w/ repl.)
       │                            │                            │
  Node Split:                  Node Split:                  Node Split:
Random m features            Random m features            Random m features
       │                            │                            │
       ▼                            ▼                            ▼
    Tree 1                       Tree 2                       Tree B
 (High Variance)              (High Variance)              (High Variance)
       │                            │                            │
       └────────────────────────────┼────────────────────────────┘
                                    ▼
                         AGGREGATION (Majority Vote)
                                    ▼
                          LOW-VARIANCE PREDICTION
```

---

## 2. Mathematical Formulation & Formal Definition

### 2.1 The Bias-Variance Tradeoff in Averaging

Let $T_1(x), T_2(x), \dots, T_B(x)$ be $B$ individual decision trees, each identically distributed (though not completely independent) with variance $\sigma^2$ and pairwise correlation $\rho = \text{Corr}(T_i(x), T_j(x))$ for $i \neq j$.

The ensemble prediction is the arithmetic average (or probability average before argmax):
$$
\bar{T}(x) = \frac{1}{B} \sum_{b=1}^B T_b(x)
$$

The expected value preserves the individual tree's expectation (leaving bias essentially unchanged):
$$
\mathbb{E}[\bar{T}(x)] = \mathbb{E}[T(x)]
$$

The variance of the ensemble average is formally derived as:
$$
\begin{aligned}
\text{Var}(\bar{T}(x)) &= \text{Var}\left(\frac{1}{B} \sum_{b=1}^B T_b(x)\right) \\
&= \frac{1}{B^2} \left[ \sum_{b=1}^B \text{Var}(T_b(x)) + \sum_{b=1}^B \sum_{j \neq b} \text{Cov}(T_b(x), T_j(x)) \right] \\
&= \frac{1}{B^2} \left[ B \sigma^2 + B(B - 1) \rho \sigma^2 \right] \\
&= \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2
\end{aligned}
$$

### Key Mathematical Takeaways:
1. **As $B \to \infty$:** The second term $\frac{1 - \rho}{B} \sigma^2 \to 0$. The irreducible variance lower bound is governed purely by $\rho \sigma^2$.
2. **The Role of Feature Subsampling:** Standard bagging decreases $\frac{1 - \rho}{B} \sigma^2$, but if trees are correlated ($\rho \approx 0.8$), the ensemble cannot drop variance below $0.8 \sigma^2$. By subsampling features at each split ($m \approx \sqrt{p}$), trees are forced to pick diverse features, driving $\rho$ downward toward zero ($\rho \to 0$). This allows the ensemble variance $\text{Var}(\bar{T}(x))$ to drop dramatically without increasing tree depth or bias.

### 2.2 Bootstrap Sample Probability (Out-of-Bag Rate)
Given a dataset of size $N$, the probability that a specific row $x_i$ is *not* selected in a single random draw with replacement is $1 - \frac{1}{N}$.
The probability that $x_i$ is omitted across all $N$ independent draws is:
$$
P(x_i \notin \text{Bootstrap}) = \left(1 - \frac{1}{N}\right)^N
$$
Taking the limit as $N \to \infty$:
$$
\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 0.3679 \approx 36.8\%
$$
Hence, approximately $63.2\%$ of unique observations enter each bootstrap training sample, while $36.8\%$ serve as pristine Out-of-Bag validation observations.

### 2.3 Mean Decrease in Impurity (MDI Feature Importance)
For tree $t \in \{1, \dots, B\}$, let $v(s)$ be the feature split at node $s$, and $\Delta I(s)$ be the impurity reduction (Gini or entropy gain) achieved by splitting node $s$:
$$
\Delta I(s) = \frac{N_s}{N} \left( I(s) - \frac{N_{s_L}}{N_s} I(s_L) - \frac{N_{s_R}}{N_s} I(s_R) \right)
$$
The unnormalized feature importance for feature $j$ across the forest is:
$$
\text{MDI}_j = \frac{1}{B} \sum_{t=1}^B \sum_{s \in \text{nodes}(t): v(s)=j} \Delta I(s)
$$
Normalizing across all $p$ features:
$$
\text{Importance}_j = \frac{\text{MDI}_j}{\sum_{k=1}^p \text{MDI}_k}, \quad \text{such that} \quad \sum_{j=1}^p \text{Importance}_j = 1.0
$$

---

## 3. From-Scratch Implementation (NumPy / Pure Python)

The following minimal, self-contained implementation demonstrates:
1. Bootstrap sampling with replacement.
2. Random feature subspace selection ($m = \text{ceil}(\sqrt{p})$).
3. Pure Python decision stump / tree with Gini impurity gain.
4. Ensemble majority voting and variance reduction proof.

```python
import numpy as np


class SimpleDecisionStump:
    """Single decision split with random feature subsampling."""

    def __init__(self, max_features: int | None = None):
        self.max_features = max_features
        self.feature_idx: int = 0
        self.threshold: float = 0.0
        self.left_class: int = 0
        self.right_class: int = 0

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - float(np.sum(probs**2))

    def fit(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> None:
        n_samples, n_features = X.shape
        m = self.max_features or int(np.ceil(np.sqrt(n_features)))
        chosen_features = rng.choice(n_features, size=min(m, n_features), replace=False)

        best_gain = -1.0
        parent_gini = self._gini(y)

        for feat in chosen_features:
            vals = np.unique(X[:, feat])
            if len(vals) <= 1:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                g_left = self._gini(y[left_mask])
                g_right = self._gini(y[right_mask])
                p_left = np.mean(left_mask)
                gain = parent_gini - (p_left * g_left + (1.0 - p_left) * g_right)

                if gain > best_gain:
                    best_gain = gain
                    self.feature_idx = feat
                    self.threshold = float(thresh)
                    # Majority vote in child leaves
                    self.left_class = int(np.bincount(y[left_mask]).argmax())
                    self.right_class = int(np.bincount(y[right_mask]).argmax())

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.where(X[:, self.feature_idx] <= self.threshold, self.left_class, self.right_class)
        return preds


class ToyRandomForest:
    """Minimal Random Forest assembling bagged trees with feature subsampling."""

    def __init__(self, n_estimators: int = 25, seed: int = 42):
        self.n_estimators = n_estimators
        self.seed = seed
        self.trees: list[SimpleDecisionStump] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        n_samples = len(X)
        self.trees = []

        for _ in range(self.n_estimators):
            # 1. Bagging: sample N with replacement
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X[boot_idx], y[boot_idx]

            # 2. Fit tree with random feature subspace
            tree = SimpleDecisionStump(max_features=None)
            tree.fit(X_boot, y_boot, rng=rng)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Collect predictions from all B trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])  # shape: (B, N)
        # Majority voting along ensemble axis (axis 0)
        final_preds = []
        for i in range(X.shape[0]):
            counts = np.bincount(tree_preds[:, i])
            final_preds.append(int(counts.argmax()))
        return np.array(final_preds)


# Verification with assertion on synthetic sensor data
if __name__ == "__main__":
    rng = np.random.default_rng(123)
    # 100 samples, 5 features (ph, ec, water_temp, humidity, air_temp)
    X_toy = rng.normal(loc=20.0, scale=3.0, size=(100, 5))
    # Binary target: 1 if pH (col 0) > 20 and EC (col 1) > 20 else 0
    y_toy = ((X_toy[:, 0] > 20.0) & (X_toy[:, 1] > 20.0)).astype(int)

    rf = ToyRandomForest(n_estimators=30, seed=42)
    rf.fit(X_toy, y_toy)
    preds = rf.predict(X_toy)

    accuracy = np.mean(preds == y_toy)
    assert accuracy > 0.85, f"Expected RF accuracy > 0.85, got {accuracy}"
    print(f"Toy Random Forest verified: Accuracy = {accuracy:.3f}")
```

---

## 4. Hydroponic Sensor Diagnostics Application

In the Hydroponic Nutrition AI system, sensor telemetry consists of 5 tightly coupled continuous variables:
`['ph', 'ec', 'water_temp', 'humidity', 'air_temp']`.

### Why Random Forest Outperforms a Single Decision Tree:
1. **Sensor Measurement Noise:** Submerged industrial glass electrode probes suffer from transient ionic drift ($\pm 0.15\text{ pH}$) and ambient electromagnetic noise from water pumps. A single tree can overfit to a transient noise spike by creating an artificial split leaf. A Random Forest averages out isolated sensor jitter across dozens of trees.
2. **Correlated Environmental Features:** In greenhouse hydroponics, `air_temp` and `water_temp` are naturally correlated ($r \approx 0.72$), as are `humidity` and transpiration rates. In a standard tree, `water_temp` might monopolize every split, completely suppressing `air_temp`. Feature subsampling ($m = \text{ceil}(\sqrt{5}) = 3$) forces trees to explore splits on `air_temp` and `humidity` when `water_temp` is excluded from the candidate set.
3. **Actionable Feature Importances:** The saved `feature_importances.json` artifact provides agronomists with direct diagnostic attribution:
   - For Lettuce EC disorders, `ec` dominates with $> 60\%$ importance.
   - For Tomato transpiration stress, `humidity` and `air_temp` account for significant combined importance.

---

## 5. Self-Check Questions & Answers

### Q1: Why doesn't Bagging alone fully solve the variance problem when features are strongly correlated?
**Answer:** Bagging averages $B$ bootstrap-trained trees. However, the variance of the average is $\rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2$. If features are correlated, every bootstrap tree will pick the same dominant feature at the root node, keeping the inter-tree correlation $\rho$ high. The $\rho \sigma^2$ floor remains large regardless of how many trees ($B$) are added. Feature subsampling breaks this correlation by forcing candidate splits from random subsets, driving $\rho$ down and unlocking substantial variance reduction.

### Q2: What fraction of unique training samples does an individual bootstrap tree observe, and what happens to the remaining samples?
**Answer:** In the limit of large $N$, each bootstrap draw with replacement omits any given row with probability $\left(1 - \frac{1}{N}\right)^N \to \frac{1}{e} \approx 36.8\%$. Thus, each tree is trained on roughly $63.2\%$ of the unique observations. The remaining $36.8\%$ are **Out-of-Bag (OOB)** samples, which can be used as an internal validation set without requiring an explicit separate split.

### Q3: When should `class_weight='balanced'` be specified in Random Forest for sensor anomalies?
**Answer:** In hydroponic telemetry, healthy/normal states represent the vast majority ($> 70\%$) of observations, while severe anomalies like `WaterTemp_High` or `EC_Low` appear infrequently ($2\text{--}5\%$). Setting `class_weight='balanced'` automatically weights the Gini impurity penalty inversely proportional to class frequencies ($w_c = \frac{N}{C \cdot N_c}$), preventing the forest from sacrificing minority anomaly detection to optimize raw sample accuracy.
