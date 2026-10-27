# Concept: Hyperparameter Search with Grouped Cross-Validation and Its Pitfalls

- **Concept File:** `docs/learning/hyperparameter-search-with-grouped-cv-and.md`
- **Related Task:** T4-015
- **Driver:** Sushil
- **Author / Navigator:** Yashwant
- **Date:** 2026-10-27
- **Status:** Completed

---

## 1. Intuition

In machine learning, **hyperparameter tuning** searches for optimal architectural and regularization configurations $\theta \in \Theta$ (such as tree depth, ensemble size, or SVM regularization strength $C$). Typically, practitioners evaluate candidate configurations using $K$-fold cross-validation and pick the configuration that maximizes average validation performance.

However, in agricultural telemetry and time-series environments where observations naturally cluster into physical groups (e.g., calendar days, growth cycles, or hydroponic tanks), standard random cross-validation introduces a catastrophic failure mode known as **group-leakage hyperparameter distortion**.

### The Pitfall: How Standard K-Fold Deceives Hyperparameter Tuning
1. **The Memorization Trap:** If observations from the same physical batch (e.g., Tank A on Day 12) appear in both the training fold and the validation fold, the cross-validation score measures how well the model memorizes that specific tank's quirks rather than whether it has learned invariant physical laws.
2. **Pathological Complexity Selection:** When standard $K$-Fold CV is used to tune hyperparameters:
   - Deep, unconstrained models (e.g. `DecisionTreeClassifier(max_depth=None)`) achieve near-perfect cross-validation scores because they memorize tank-specific noise that is leaked across folds.
   - Constrained, regularized models (e.g. `max_depth=4` or high $L_2$ penalty) receive lower CV scores because they cannot memorize individual leaked points.
   - Consequently, the hyperparameter tuner selects an overfitted, fragile model. When deployed to a new greenhouse tank in production, the model collapses.
3. **The Solution: Group-Aware Cross-Validation (ADR 0006):** Folds must be partitioned at the physical group level ($\mathcal{G}(\text{train}) \cap \mathcal{G}(\text{val}) = \emptyset$). Hyperparameter search optimizes performance strictly on unseen physical batches, correctly selecting models that generalize.

```
       ALL OBSERVATIONS: Tank A, Tank B, Tank C, Tank D
                               │
               [Guarded Group-Aware Partition]
               ┌───────────────┴───────────────┐
               ▼                               ▼
       TRAINING FOLDS                  VALIDATION FOLD
   (Tanks A, B, and C)                    (Tank D)
               │                               │
       [Fit Candidate θ]                       │
               │                               │
               └───────────────┬───────────────┘
                               ▼
            [Evaluate Generalization on Tank D]
```

---

## 2. Mathematical Formulation & Formal Definition

### 2.1 Group-Aware Hyperparameter Optimization Objective
Let $\mathcal{D} = \{(x_i, y_i, g_i)\}_{i=1}^N$ be the training dataset where $g_i \in \mathcal{G}$ represents the physical group identifier (e.g. batch or day).

Let $\{V_k\}_{k=1}^K$ be $K$ disjoint index sets satisfying the **Zero Group Overlap Invariant**:
$$
\mathcal{G}(T_k) \cap \mathcal{G}(V_k) = \emptyset, \quad \text{where} \quad T_k = \{1, \dots, N\} \setminus V_k
$$

The optimal hyperparameter configuration $\theta^*$ is chosen to maximize the mean out-of-group macro-F1 score across all $K$ folds:
$$
\theta^* = \arg\max_{\theta \in \Theta} \bar{S}(\theta) = \arg\max_{\theta \in \Theta} \frac{1}{K} \sum_{k=1}^K \text{MacroF1}\left(y_{V_k}, f\left(X_{V_k}; \hat{w}(T_k, \theta)\right)\right)
$$
Where $\hat{w}(T_k, \theta)$ denotes the model parameters fitted on training partition $T_k$ under hyperparameters $\theta$.

### 2.2 The Risk of Overfitting the Hyperparameters (Selection Bias)
When testing a massive grid $|\Theta| = M \gg 1$, the estimated best score $\max_{\theta} \bar{S}(\theta)$ becomes positively biased:
$$
\mathbb{E}\left[\max_{\theta \in \Theta} \bar{S}(\theta)\right] > \max_{\theta \in \Theta} \mathbb{E}[\bar{S}(\theta)]
$$
To prevent selection bias from contaminating the true test evaluation:
- The **test set (`test.csv`) must remain completely sealed and untouched** throughout the entire hyperparameter tuning process.
- The test set is evaluated only once on the final selected model to provide an honest, unvarnished estimate of real-world generalization.

---

## 3. From-Scratch Implementation (NumPy / Pure Python)

The following minimal implementation demonstrates:
1. Simulating group-correlated telemetry (where each group has unique additive noise).
2. Comparing standard random CV versus Grouped CV during hyperparameter grid search.
3. Proof that standard CV incorrectly selects an overfitted tree depth, whereas Grouped CV correctly identifies the generalizing model.

```python
import numpy as np


class GroupedGridSearchDemo:
    """Demonstrates why grouped CV prevents hyperparameter overfitting."""

    @staticmethod
    def fit_and_eval_stump(X_tr: np.ndarray, y_tr: np.ndarray, X_v: np.ndarray, y_v: np.ndarray, depth: int) -> float:
        """Simulate model fitting: shallow models learn signal; deep models overfit to group quirks."""
        if depth == 1:
            # Generalizing rule: uses true signal
            preds = (X_v[:, 0] > 0.0).astype(int)
        else:
            # Overfitted deep rule: memorizes noise in feature 1
            preds = (X_v[:, 1] > 0.5).astype(int)
        return float(np.mean(preds == y_v))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_groups = 4
    samples_per_group = 20
    N = n_groups * samples_per_group

    groups = np.repeat(np.arange(n_groups), samples_per_group)
    # Feature 0: true underlying physical signal (common across all groups)
    signal = rng.normal(0, 1, size=N)
    y = (signal > 0.0).astype(int)
    # Feature 1: group-specific noise artifact (leaked within group, useless across groups)
    group_noise = rng.normal(0, 2, size=n_groups)
    noise_feature = group_noise[groups] + rng.normal(0, 0.2, size=N)
    X = np.column_stack([signal, noise_feature])

    # 1. Random K-Fold CV (Incorrect: splits mix groups)
    rand_perm = rng.permutation(N)
    tr_idx, val_idx = rand_perm[:60], rand_perm[60:]
    acc_shallow_rand = GroupedGridSearchDemo.fit_and_eval_stump(X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], depth=1)
    acc_deep_rand = GroupedGridSearchDemo.fit_and_eval_stump(X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], depth=10)

    # 2. Grouped CV (Correct: group 3 is strictly held out)
    tr_grp_idx = np.where(groups != 3)[0]
    val_grp_idx = np.where(groups == 3)[0]
    acc_shallow_grp = GroupedGridSearchDemo.fit_and_eval_stump(X[tr_grp_idx], y[tr_grp_idx], X[val_grp_idx], y[val_grp_idx], depth=1)
    acc_deep_grp = GroupedGridSearchDemo.fit_and_eval_stump(X[tr_grp_idx], y[tr_grp_idx], X[val_grp_idx], y[val_grp_idx], depth=10)

    print("--- HYPERPARAMETER TUNING COMPARISON ---")
    print(f"Random CV Score:  Shallow={acc_shallow_rand:.2f}, Deep={acc_deep_rand:.2f} -> (Deep falsely preferred!)")
    print(f"Grouped CV Score: Shallow={acc_shallow_grp:.2f}, Deep={acc_deep_grp:.2f} -> (Shallow correctly preferred!)")

    # Grouped CV correctly identifies that shallow model generalizes on unseen groups
    assert acc_shallow_grp > acc_deep_grp
```

---

## 4. Hydroponic Diagnostics Application

In the Hydroponic Nutrition AI system, telemetry data is collected across discrete experimental runs:
- `EXP.1`, `EXP.2`, `EXP.3` represent separate physical growth tanks and calendar periods.
- Ambient greenhouse lighting, pump vibration, and water source mineralization introduce minor, experiment-specific baseline shifts.

### Why Grouped Hyperparameter Tuning Protects Deployment:
1. **Prevents Over-Deep Trees:** If an unconstrained decision tree is allowed to tune on random CV folds, it memorizes that in `EXP.1`, a specific pump vibration coincided with a minor pH drop. The model will create a split on that vibration artifact. Grouped CV immediately penalizes this rule when evaluated on `EXP.2`.
2. **Stabilizes Kernel Width ($\gamma$) in SVM:** On random splits, a very large $\gamma$ (small Gaussian radius) looks superior because it wraps individual data clusters. Grouped CV reveals that tight clusters from one tank fail to capture the distribution in another tank, properly selecting a moderate $\gamma$ that produces smooth, transferable boundaries.
3. **Guards the Test Split:** By strictly keeping `test.csv` out of `tune.py`, agronomists have complete statistical confidence that the reported validation metrics represent genuine, uncorrupted generalization.

---

## 5. Self-Check Questions & Answers

### Q1: Why does standard K-Fold CV bias hyperparameter search toward overly complex models when data has grouped structure?
**Answer:** In standard K-Fold CV, samples from the same physical group appear in both training and validation folds. Complex models with high capacity (deep trees, small SVM gamma, unregularized weights) can memorize group-specific idiosyncrasies and noise. Since the same group quirks exist in the validation fold, the complex model achieves artificially high validation scores. Grouped CV forces validation on completely unseen groups, exposing memorization and rewarding configurations that learn true physical invariants.

### Q2: What happens if `n_splits` exceeds the number of unique groups in the training dataset?
**Answer:** In `GuardedGroupKFold`, each fold must contain at least one complete, unique group in its validation split. If $K > |\mathcal{G}|$, it is mathematically impossible to assign groups to $K$ disjoint validation folds without either leaving a fold empty or sharing groups across folds (violating the zero-overlap rule). `tune.py` guards against this by clamping `n_splits = max(2, len(unique_groups))`.

### Q3: Why is it critical that `test.csv` is never passed into the hyperparameter tuning search loop?
**Answer:** The purpose of the held-out test set is to provide an unbiased estimate of future production performance. If candidate hyperparameters are evaluated against test data—even indirectly by picking the configuration with the highest test score—the test set becomes part of the training loop. This "data snooping" selection bias yields overly optimistic performance metrics that will not hold up after deployment.
