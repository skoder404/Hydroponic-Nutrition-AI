# Concept: KNN and Logistic Regression Baselines: Distance Metrics, Feature Scaling, and Regularization

- **Concept File:** `docs/learning/knn-and-logistic-regression-baselines-distance.md`
- **Related Task:** T4-014
- **Driver:** Yashwant
- **Author / Navigator:** Sushil
- **Date:** 2026-10-27
- **Status:** Completed

---

## 1. Intuition

Before deploying complex ensembles (Random Forests, Gradient Boosting) or deep networks in a production system, machine learning best practices demand establishing solid **baseline models**. Baselines answer a critical architectural question:
> *"Is the non-linear complexity and computational cost of an advanced model actually justified by measurable gains over simple linear and memory-based baselines?"*

This project introduces two foundational baselines across sensor diagnostics:
1. **$k$-Nearest Neighbors (KNN):** A non-parametric, instance-based model that classifies an unseen sensor query by taking a majority vote (or distance-weighted vote) among its $k$ closest neighbors in feature space.
2. **Logistic Regression (Multinomial):** A parametric, linear model that learns a linear decision hyperplane per class using cross-entropy loss and regularized weight decay ($L_2$ Ridge or $L_1$ Lasso).

### Why Distance-Based Models are Fragile to Feature Scale
A model that calculates Euclidean or Minkowski distances between points treats all axes as dimensionless numbers in geometric space. 

In hydroponics, our 5 telemetry features operate on drastically different physical units:
- `humidity` fluctuates between $30.0$ and $85.0$ (range $\approx 55.0$).
- `air_temp` fluctuates between $18.0$ and $34.0$ (range $\approx 16.0$).
- `water_temp` fluctuates between $18.0$ and $28.0$ (range $\approx 10.0$).
- `ph` fluctuates between $5.5$ and $7.5$ (range $\approx 2.0$).
- `ec` fluctuates between $1.0$ and $3.5$ (range $\approx 2.5$).

When computing the distance $d(u, v) = \sqrt{\sum (u_k - v_k)^2}$, the squared difference in `humidity` ($50^2 = 2500$) completely dwarfs a lethal spike in `ec` ($1.5^2 = 2.25$). As a consequence, unscaled KNN finds neighbors that share similar humidity, completely ignoring whether the water solution is toxic or depleted. Standardizing inputs with `StandardScaler` ($\mu = 0, \sigma = 1$) equalizes feature variances and restores multi-dimensional diagnostic sensitivity.

---

## 2. Comparison Table Across Five Model Families (Lettuce Sensor Diagnostics)

Per Acceptance Criterion 2, the table below records the validation performance of all five model families on the held-out validation split (`processed_data/sensor/lettuce/val.csv`):

| Model Family | Pipeline Architecture | Preprocessing | Validation Macro-F1 | Validation Accuracy | Inference Characteristics |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Decision Tree** | `DecisionTreeClassifier` (depth=6, min_leaf=2) | None (Scale-Invariant) | **0.7125** | 0.7760 | Ultra-fast ($O(\text{depth})$), human-interpretable rules |
| **Random Forest** | `RandomForestClassifier` (trees=100, balanced) | None (Scale-Invariant) | **0.7842** | **0.8480** | Robust to noise, variance reduction, feature importances |
| **SVM (RBF Kernel)** | `SVC(kernel='rbf', C=1.0, probability=True)` | `StandardScaler` + `SimpleImputer` | **0.8151** | **0.8400** | Max-margin boundary, high separation on scaled spaces |
| **KNN (Baseline)** | `KNeighborsClassifier(n_neighbors=5)` | `StandardScaler` + `SimpleImputer` | **0.4728** | 0.5840 | Memory-based ($O(N \cdot d)$ inference), pure local topology |
| **Logistic Regression** | `LogisticRegression(C=1.0, multi_class='multinomial')` | `StandardScaler` + `SimpleImputer` | **0.5416** | 0.6320 | Fast linear baseline, well-calibrated softmax probabilities |

### Key Takeaways from the Comparison:
1. **Tree Ensembles and SVM dominate:** Both Random Forest ($0.784$) and Scaled SVM ($0.815$) significantly outperform linear Logistic Regression ($0.542$) and KNN ($0.473$), proving that the physical hydroponic solution boundaries (coupling pH, EC, and temperature) possess non-linear interaction structures.
2. **Logistic Regression confirms non-linearity:** The linear baseline plateaus at $\approx 0.54$ Macro-F1 because single hyperplanes cannot isolate multi-dimensional rule deviations simultaneously.
3. **KNN suffers from high dimensionality and sparsity:** Even with scaling, KNN struggles with rare anomaly classes because sparse minority points lack sufficient local density in 5-dimensional continuous space.

---

## 3. Mathematical Formulation & Formal Definition

### 3.1 KNN Distance Metric Sensitivity
For vectors $x, x' \in \mathbb{R}^p$, the general Minkowski distance of order $q$ is:
$$
D_q(x, x') = \left( \sum_{k=1}^p |x_k - x'_k|^q \right)^{\frac{1}{q}}
$$
For Euclidean distance ($q = 2$):
$$
D_2(x, x')^2 = \sum_{k=1}^p (x_k - x'_k)^2
$$
If feature $k$ has scale $\sigma_k$, its expected squared difference scales as $\mathbb{E}[(x_k - x'_k)^2] \propto \sigma_k^2$. 
Features with $\sigma_k \gg \sigma_j$ dominate the metric tensor:
$$
\lim_{\sigma_1 \gg \sigma_{2..p}} D_2(x, x') \approx |x_1 - x'_1|
$$
Standardization replaces $x_k$ with $z_k = \frac{x_k - \mu_k}{\sigma_k}$, ensuring $\text{Var}(z_k) = 1$ for all $k \in \{1, \dots, p\}$.

### 3.2 Multinomial Logistic Regression and Regularization
For $C$ classes and $p$ features, Multinomial Logistic Regression models class posterior probabilities via the Softmax function:
$$
P(y = c \mid x) = \frac{\exp(w_c^T x + b_c)}{\sum_{j=1}^C \exp(w_j^T x + b_j)}
$$
The regularized negative log-likelihood objective with $L_2$ weight penalty (Ridge) is:
$$
\mathcal{L}(W, b) = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C \mathbb{I}(y_i = c) \log P(y_i = c \mid x_i) + \frac{1}{2C_{\text{reg}}} \sum_{c=1}^C \|w_c\|_2^2
$$
Where:
- $C_{\text{reg}} > 0$ is the inverse regularization strength (smaller values penalize large weights more severely).
- **Why scaling is required for regularized logistic regression:** The penalty term $\sum_j w_{c,j}^2$ penalizes all coefficients uniformly. If feature $x_1$ has a tiny scale (e.g. EC), its corresponding weight $w_{c,1}$ must be large to influence the logits. The regularization term will unfairly suppress $w_{c,1}$ over weights corresponding to large-scale features, biasing the learned hyperplane away from critical small-scale signals.

---

## 4. From-Scratch Implementation (NumPy / Pure Python)

The following minimal implementation demonstrates:
1. Pure NumPy KNN classifier.
2. Proof of distance distortion under unscaled versus scaled inputs.
3. Nearest neighbor voting with assertions.

```python
import numpy as np


class ScaledKNNFromScratch:
    """Minimal k-Nearest Neighbors classifier demonstrating scale sensitivity."""

    def __init__(self, n_neighbors: int = 3, scale: bool = True):
        self.k = n_neighbors
        self.scale = scale
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> ScaledKNNFromScratch:
        if self.scale:
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            self.std_[self.std_ == 0.0] = 1.0
            self.X_train_ = (X - self.mean_) / self.std_
        else:
            self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.X_train_ is not None and self.y_train_ is not None
        if self.scale:
            assert self.mean_ is not None and self.std_ is not None
            X_query = (X - self.mean_) / self.std_
        else:
            X_query = X

        preds = []
        for x in X_query:
            # Compute Euclidean distances to all training points
            dists = np.sqrt(np.sum((self.X_train_ - x) ** 2, axis=1))
            # Find indices of k smallest distances
            k_indices = np.argsort(dists)[: self.k]
            k_labels = self.y_train_[k_indices]
            # Majority vote
            counts = np.bincount(k_labels)
            preds.append(int(counts.argmax()))
        return np.array(preds)


# Verification with assertion on disparate-scale sensor telemetry
if __name__ == "__main__":
    rng = np.random.default_rng(101)
    N = 60
    # Feature 0: humidity (scale 40..80), Feature 1: EC (scale 1.0..3.0)
    # Ground truth rule: Class 1 if EC > 2.2 (EC_High), else Class 0
    humidity = rng.uniform(40.0, 80.0, size=N)
    ec = rng.uniform(1.0, 3.0, size=N)
    X = np.column_stack([humidity, ec])
    y = (ec > 2.2).astype(int)

    # 1. Without scaling: humidity noise destroys EC-based classification
    knn_unscaled = ScaledKNNFromScratch(n_neighbors=3, scale=False).fit(X, y)
    acc_unscaled = np.mean(knn_unscaled.predict(X) == y)

    # 2. With scaling: EC signal is recovered and accurately classified
    knn_scaled = ScaledKNNFromScratch(n_neighbors=3, scale=True).fit(X, y)
    acc_scaled = np.mean(knn_scaled.predict(X) == y)

    assert acc_scaled > acc_unscaled, f"Scaled ({acc_scaled:.2f}) must outperform unscaled ({acc_unscaled:.2f})"
    print(f"KNN Scale Verification: Unscaled Acc = {acc_unscaled:.2%}, Scaled Acc = {acc_scaled:.2%}")
```

---

## 5. Self-Check Questions & Answers

### Q1: Why are Decision Trees and Random Forests invariant to monotonic feature scaling, while KNN and SVM are extremely sensitive?
**Answer:** Decision trees split each feature independently based on rank order ($\mathbb{I}(x_j \le \theta)$), never comparing values across different dimensions. Multiplying a feature by 1000 simply shifts the split threshold without altering the partition or Gini gain. In contrast, KNN and SVM compute Euclidean norms across all dimensions simultaneously ($\sum (x_j - x'_j)^2$); features with large numerical variance dominate the metric and dictate neighbor selection.

### Q2: What is the primary operational drawback of deploying KNN in a high-frequency real-time edge sensor pipeline?
**Answer:** KNN is a lazy, instance-based learner with zero training time but $O(N \cdot p)$ inference latency per telemetry reading, where $N$ is the size of the training dataset. As telemetry grows into tens of thousands of rows, evaluating pairwise distances to every historical sample on an embedded IoT controller becomes computationally prohibitive and memory-intensive, unlike parametric models (SVM, Logistic Regression) or tree ensembles whose inference complexity is independent of $N$.

### Q3: Why does unscaled data distort $L_2$ regularization in Logistic Regression?
**Answer:** The $L_2$ penalty $\frac{1}{2C} \sum_j w_j^2$ applies an identical quadratic cost to all feature weights. When a feature has a very small numeric scale (such as EC in mS/cm), its weight $w_j$ must naturally be large to produce an equivalent change in the log-odds. The regularization penalty heavily penalizes this large weight, unfairly shrinking it and preventing the model from utilizing critical signals with small physical magnitudes.
