# Concept: Decision Trees: Gini Impurity and Greedy Splits

- **Concept File:** `docs/learning/decision-trees-gini-impurity-and-greedy.md`
- **Related Task:** T4-003
- **Driver:** Yashwant
- **Author / Navigator:** Sushil
- **Date:** 2026-10-27
- **Status:** Completed

---

## 1. Intuition

A decision tree is a hierarchical, non-parametric supervised learning algorithm that partitions feature space into orthogonal hyper-rectangles through a sequence of recursive binary decisions. 

In a hydroponic greenhouse, an agronomist diagnosing plant health does not process 10 environmental sensor parameters simultaneously via complex non-linear manifolds. Instead, they apply a mental flow chart:
1. *Is the electrical conductivity (EC) above 2.5 mS/cm?* If yes $\rightarrow$ inspect for nutrient salt toxicity.
2. *If no, is the pH below 5.5?* If yes $\rightarrow$ acid imbalance.
3. *If no, is water temperature above 24°C?* If yes $\rightarrow$ root hypoxia hazard.

A decision tree automates the synthesis of this exact hierarchical diagnostic flowchart. At each internal node, the algorithm must select:
1. **Which feature** to split on (e.g., pH vs. EC vs. water temperature).
2. **What numerical threshold** to evaluate ($x_j \le \theta$).

Because evaluating all possible tree architectures simultaneously is an NP-complete combinatorial problem, decision trees employ a **top-down greedy heuristic** (the CART algorithm). At every node, it evaluates every candidate feature and split threshold, choosing the single split that maximizes the reduction in class mixture—measured by **Gini impurity**.

A leaf node is "pure" (Gini = 0.0) if all samples belonging to it share the exact same diagnostic class (e.g., all 100% "Normal"). A leaf node is maximally impure if samples are distributed uniformly across all classes.

---

## 2. Mathematical Formulation & Formal Definition

### 2.1 Gini Impurity Definition
Let node $m$ contain a set of $N_m$ training instances $Q_m$, distributed across $K$ discrete classes $k \in \{1, 2, \dots, K\}$.
The empirical probability of class $k$ in node $m$ is:

$$
p_{m,k} = \frac{1}{N_m} \sum_{i \in Q_m} \mathbb{I}(y_i = k)
$$

The **Gini Impurity** $I_G(m)$ of node $m$ measures the probability that a randomly chosen element from the node would be incorrectly labeled if it were randomly labeled according to the class distribution in the node:

$$
I_G(m) = \sum_{k=1}^K p_{m,k} (1 - p_{m,k}) = 1 - \sum_{k=1}^K p_{m,k}^2
$$

**Properties:**
- **Minimum Value:** $I_G(m) = 0.0$ when all samples belong to a single class (maximum purity: $\exists k \text{ s.t. } p_{m,k} = 1.0$).
- **Maximum Value:** $I_G(m) = 1 - \frac{1}{K}$ when samples are uniformly distributed across all $K$ classes ($p_{m,k} = \frac{1}{K}$ for all $k$). For binary classification ($K=2$), maximum impurity is $0.5$.

### 2.2 Split Evaluation and Impurity Reduction (Gini Gain)
A candidate split $\theta = (j, t)$ partitions the dataset $Q_m$ on feature $j$ at threshold $t$ into left and right child subsets:

$$
Q_m^{\text{left}}(\theta) = \{ (x, y) \in Q_m \mid x_j \le t \}
$$

$$
Q_m^{\text{right}}(\theta) = \{ (x, y) \in Q_m \mid x_j > t \}
$$

Let $N_L = |Q_m^{\text{left}}|$ and $N_R = |Q_m^{\text{right}}|$ such that $N_m = N_L + N_R$.
The weighted post-split impurity $G(Q_m, \theta)$ is:

$$
G(Q_m, \theta) = \frac{N_L}{N_m} I_G(Q_m^{\text{left}}) + \frac{N_R}{N_m} I_G(Q_m^{\text{right}})
$$

The greedy split criterion selects the optimal feature $j^*$ and threshold $t^*$ that minimizes weighted post-split impurity, thereby maximizing impurity reduction (Gini gain):

$$
\theta^* = \arg\min_{\theta} G(Q_m, \theta) \equiv \arg\max_{\theta} \left[ I_G(Q_m) - G(Q_m, \theta) \right]
$$

### 2.3 Candidate Split Thresholds
For a continuous numeric feature $x_j$, instances are sorted in ascending order: $x_{(1), j} \le x_{(2), j} \le \dots \le x_{(N), j}$.
Candidate thresholds $t_i$ are placed strictly at the midpoints between distinct adjacent values:

$$
t_i = \frac{x_{(i), j} + x_{(i+1), j}}{2} \quad \text{for } x_{(i), j} \neq x_{(i+1), j}
$$

---

## 3. From-Scratch Implementation (NumPy / Pure Python)

The following minimal, pure NumPy module implements Gini impurity and an exhaustive 1D greedy threshold search, and verifies that its chosen threshold matches `scikit-learn`'s `DecisionTreeClassifier(max_depth=1)`:

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def compute_gini(y: np.ndarray) -> float:
    """Compute Gini impurity for an array of discrete integer labels."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return float(1.0 - np.sum(probabilities**2))


def find_best_split_1d(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float]:
    """Find greedy best split threshold for a single numeric feature.

    Returns:
        (best_threshold, best_cost, gini_gain)
    """
    n_samples = len(y)
    parent_gini = compute_gini(y)

    # Sort values
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    best_cost = float("inf")
    best_threshold = None

    # Evaluate candidate midpoints between adjacent distinct values
    for i in range(len(x_sorted) - 1):
        if x_sorted[i] == x_sorted[i + 1]:
            continue

        threshold = (x_sorted[i] + x_sorted[i + 1]) / 2.0
        left_mask = x_sorted <= threshold
        y_left = y_sorted[left_mask]
        y_right = y_sorted[~left_mask]

        n_left = len(y_left)
        n_right = len(y_right)
        cost = (n_left / n_samples) * compute_gini(y_left) + (
            n_right / n_samples
        ) * compute_gini(y_right)

        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    gini_gain = parent_gini - (best_cost if best_threshold is not None else 0.0)
    return best_threshold, best_cost, gini_gain


if __name__ == "__main__":
    # Hand-worked test case: pH values vs Plant Health (0: Healthy, 1: pH Imbalance)
    # Samples: pH in [5.4, 5.7, 5.8, 6.5, 6.8, 7.2]
    # Labels:  [1, 1, 0, 0, 0, 1]
    X_toy = np.array([5.4, 5.7, 5.8, 6.5, 6.8, 7.2]).reshape(-1, 1)
    y_toy = np.array([1, 1, 0, 0, 0, 1])

    # Scratch computation
    scratch_thresh, scratch_cost, gain = find_best_split_1d(
        X_toy.ravel(), y_toy
    )

    # Scikit-learn computation
    clf = DecisionTreeClassifier(max_depth=1, criterion="gini", random_state=42)
    clf.fit(X_toy, y_toy)
    sklearn_thresh = clf.tree_.threshold[0]

    print(f"Scratch Threshold:  {scratch_thresh:.4f} (Cost: {scratch_cost:.4f})")
    print(f"Sklearn Threshold:  {sklearn_thresh:.4f}")
    assert np.isclose(scratch_thresh, sklearn_thresh), "Threshold mismatch!"
    print("[PASS] Scratch Gini implementation equals scikit-learn threshold.")
```

---

## 4. Production & Library Usage

In production, decision tree training and rule extraction are implemented via `scikit-learn`:

```python
from sklearn.tree import DecisionTreeClassifier, export_text

# Instantiate with parsimonious constraints
clf = DecisionTreeClassifier(
    criterion="gini",  # Or 'entropy' (information gain)
    max_depth=5,  # Prevents memorization of noise
    min_samples_leaf=5,  # Ensures leaf decisions generalize
    min_samples_split=10,  # Minimum node size to consider splits
    class_weight="balanced",  # Adjusts weights inversely proportional to class frequencies
    random_state=42,  # Ensures deterministic split selection on ties
)

clf.fit(X_train, y_train)

# Export white-box rules for agronomist interpretability
tree_rules = export_text(clf, feature_names=list(feature_names))
print(tree_rules)
```

**Key Hyperparameters & Trade-offs:**
- `criterion='gini'`: Computationally faster than entropy because it avoids logarithmic operations ($\log_2(p)$) while yielding virtually identical split choices in over 98% of cases.
- `max_depth`: Controls tree complexity. Depth $D$ yields up to $2^D$ leaves. Small depth ($D \le 5$) guarantees transparent human interpretability; excessive depth leads to high variance and severe overfitting.
- `min_samples_leaf`: Guards against creating isolated micro-leaves that memorize outlier sensor spikes.

---

## 5. Failure Modes, Edge Cases & Debugging

1. **Greedy Horizon Effect (Myopia):**
   - *Failure:* Greedy splitting selects the best instantaneous single-variable split. It cannot look ahead 2 steps. Two interacting features (like an XOR problem: e.g. EC normal and pH normal individually, but their combination indicates nutrient lockout) may look completely uninformative at the root node, resulting in suboptimal trees.
   - *Mitigation:* Explicit interaction features or ensemble methods (Random Forests).
2. **Axis-Aligned Splitting on Correlated Diagonal Features:**
   - *Failure:* Decision tree splits are strictly orthogonal ($x_j \le t$). If two sensor variables correlate along a diagonal manifold (e.g., water temperature and dissolved oxygen), the tree must construct a complex "staircase" of splits, consuming depth and overfitting.
   - *Mitigation:* Apply PCA or feature scaling/rotation prior to training if oblique boundaries are expected.
3. **Class Imbalance Distortion:**
   - *Failure:* With extreme imbalance (e.g. 98% Normal, 2% Toxicity), a split that isolates a small cluster of majority class items can achieve a lower weighted Gini cost than a split that partially isolates the minority class.
   - *Mitigation:* Pass `class_weight='balanced'` and evaluate splits using Macro-F1 rather than accuracy.

---

## 6. Where It Appears in This Project

- **Exploratory & From-Scratch Notebook:** [`notebooks/02_gini_from_scratch.ipynb`](../../notebooks/02_gini_from_scratch.ipynb) — implements Gini calculation on 2-feature lettuce telemetry.
- **Model Training Pipeline:** [`scripts/train_decision_tree.py`](../../scripts/train_decision_tree.py) — trains candidate DT models for all crops and exports rule manifests.
- **Promoted Sensor Models:**
  - Lettuce: [`models/sensor/lettuce/v1.0.0/lettuce_sensor_decision_tree_v1.0.0.joblib`](../../models/sensor/lettuce/v1.0.0/lettuce_sensor_decision_tree_v1.0.0.joblib)
  - Spinach: [`models/sensor/spinach/v1.0.0/spinach_sensor_decision_tree_v1.0.0.joblib`](../../models/sensor/spinach/v1.0.0/spinach_sensor_decision_tree_v1.0.0.joblib)

---

## 7. Self-Check Questions (5 Questions)

1. **Q1:** What is the numerical range of Gini impurity for a 10-class problem (such as tomato sensor classification), and what state corresponds to the maximum?
   - *Answer:* The range is $[0, 0.90]$. Minimum is $0.0$ when the node contains only one class. Maximum is $1 - \frac{1}{10} = 0.90$, which occurs when instances are distributed uniformly (10% per class).
2. **Q2:** Why do candidate threshold searches only test midpoints between distinct adjacent feature values rather than every value in the domain?
   - *Answer:* Any threshold $t \in [x_{(i)}, x_{(i+1)})$ partitions the dataset into the exact same left and right subsets. Choosing the midpoint $t = \frac{x_{(i)} + x_{(i+1)}}{2}$ maximizes margin separation between the adjacent data points.
3. **Q3:** If a parent node has Gini impurity of 0.48 and a proposed split creates children with impurities 0.20 and 0.30 of equal sample sizes, what is the Gini gain?
   - *Answer:* Post-split weighted impurity is $0.5(0.20) + 0.5(0.30) = 0.25$. The Gini gain is $0.48 - 0.25 = 0.23$.
4. **Q4:** How does Gini impurity differ from Information Gain (Shannon Entropy)?
   - *Answer:* Gini uses variance of class indicators ($1 - \sum p_k^2$) whereas entropy uses logarithmic information measure ($-\sum p_k \log_2 p_k$). Entropy penalizes mixed leaves slightly more heavily, but Gini is computationally more efficient and produces nearly identical splits.
5. **Q5:** Why can an unconstrained decision tree always achieve 100% training accuracy on consistent data, and why is this disastrous in hydroponic monitoring?
   - *Answer:* An unconstrained tree can recursively split until every training instance occupies its own leaf (samples=1, Gini=0). In hydroponics, it memorizes high-frequency sensor noise, daily battery fluctuations, or measurement error rather than physiological deficiency rules, collapsing on unseen days.
