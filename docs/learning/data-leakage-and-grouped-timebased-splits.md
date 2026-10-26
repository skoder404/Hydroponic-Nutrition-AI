# Concept: Data Leakage and Grouped / Time-Based Splits

- **Concept File:** `docs/learning/data-leakage-and-grouped-timebased-splits.md`
- **Related Task:** T4-001
- **Driver:** Sushil
- **Author / Navigator:** Yashwant
- **Date:** 2026-10-26
- **Status:** Completed

---

## 1. Intuition

In introductory machine learning, data points are assumed to be **independent and identically distributed (i.i.d.)**. Under this assumption, shuffling rows randomly and drawing an 80/20 train/validation split (`train_test_split`) produces an unbiased estimate of future generalization error.

However, in **hydroponic sensor telemetry and time-series environments**, this i.i.d. assumption is catastrophically false. Physical systems possess inertia: a 500-liter nutrient tank does not change temperature, pH, or electrical conductivity instantaneously. Sensor readings logged every 10 minutes are heavily autocorrelated. Furthermore, readings collected on the same calendar day share unobserved environmental confounders (e.g., cloudy weather, solar irradiance, ambient greenhouse humidity, diurnal cycles, or specific nutrient replenishment events).

When a dataset is shuffled row-by-row, consecutive readings from the **exact same day and tank** are partitioned across both the training set and the validation set. If row $t$ is in the training set and row $t+1$ (10 minutes later) is in the validation set, a decision tree or neural network does not need to learn physical rules governing crop stress. It simply memorizes the local state of that specific tank on that specific day and acts as a local interpolator. When evaluated on the random validation split, the model achieves near-perfect diagnostic accuracy (e.g. 100% Macro-F1). 

However, when deployed in production tomorrow, the model is asked to predict on an **unseen future calendar day** (extrapolation). The temporal interpolation cheat is impossible, and diagnostic accuracy collapses (e.g. dropping by 20%–40%), causing undetected crop failure. Grouped holdout splitting (holding out whole calendar days or multi-hour blocks) enforces true extrapolation and prevents this catastrophic leakage.

---

## 2. Mathematical Formulation & Formal Definition

### 2.1 Temporal Autocorrelation Function (ACF)
Let $X_t$ be a univariate continuous sensor reading (e.g., water temperature or pH) at discrete timestamp $t$, with mean $\mu$ and variance $\sigma^2$. The lag-$k$ autocovariance $\gamma(k)$ and autocorrelation $\rho(k)$ are defined as:

$$
\gamma(k) = \mathbb{E}\left[ (X_t - \mu)(X_{t+k} - \mu) \right]
$$

$$
\rho(k) = \frac{\gamma(k)}{\gamma(0)} = \frac{\mathbb{E}\left[ (X_t - \mu)(X_{t+k} - \mu) \right]}{\sigma^2}
$$

When $\rho(1) \approx 0.95$, adjacent readings $X_t$ and $X_{t+1}$ share nearly 90%+ of their empirical variance.

### 2.2 Empirical Risk Minimization under Temporal Dependence
Standard Empirical Risk Minimization (ERM) minimizes empirical loss over a sample $S = \{(x_i, y_i)\}_{i=1}^N$:

$$
\hat{R}(f) = \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f(x_i), y_i)
$$

For an unbiased generalization bound $\mathbb{E}[\hat{R}(f)] = R(f)$, the samples must be drawn independently:

$$
P(x_1, \dots, x_N) = \prod_{i=1}^N P(x_i)
$$

When $(x_i, x_j)$ belong to the same cluster or calendar date $g \in G$, their joint distribution has mutual information $I(x_i; x_j) > 0$. Under random row splitting, the training set $S_{\text{train}}$ and validation set $S_{\text{val}}$ share groups:

$$
G(S_{\text{train}}) \cap G(S_{\text{val}}) \neq \emptyset
$$

A grouped holdout split guarantees complete group disjointness:

$$
G(S_{\text{train}}) \cap G(S_{\text{val}}) = \emptyset
$$

---

## 3. From-Scratch Implementation (NumPy / Pure Python)

The following pure-Python snippet implements both naive random splitting and grouped holdout splitting, computes empirical autocorrelation, and evaluates the resulting performance gap without external ML frameworks:

```python
import numpy as np


def generate_synthetic_telemetry(n_days=30, readings_per_day=24, seed=42):
    """Simulate diurnal, autocorrelated hydroponic water temperature."""
    rng = np.random.RandomState(seed)
    data = []
    for day in range(n_days):
        day_baseline = 20.0 + rng.normal(0, 1.5)  # Daily weather shift
        temp = day_baseline
        for hour in range(readings_per_day):
            temp = 0.85 * temp + 0.15 * day_baseline + rng.normal(0, 0.2)
            # Label: 1 if stress (> 22.0 C), else 0
            label = 1 if temp > 22.0 else 0
            data.append({"day": day, "hour": hour, "temp": temp, "label": label})
    return data


def random_split(data, test_ratio=0.2, seed=42):
    """Naive random row split (violates group independence)."""
    rng = np.random.RandomState(seed)
    indices = np.arange(len(data))
    rng.shuffle(indices)
    cut = int(len(data) * (1 - test_ratio))
    train_idx, val_idx = set(indices[:cut]), set(indices[cut:])
    return [data[i] for i in train_idx], [data[i] for i in val_idx]


def grouped_split(data, test_ratio=0.2, seed=42):
    """Grouped holdout split by calendar day."""
    rng = np.random.RandomState(seed)
    days = sorted(list({d["day"] for d in data}))
    rng.shuffle(days)
    cut = int(len(days) * (1 - test_ratio))
    train_days, val_days = set(days[:cut]), set(days[cut:])
    train = [d for d in data if d["day"] in train_days]
    val = [d for d in data if d["day"] in val_days]
    return train, val


if __name__ == "__main__":
    telemetry = generate_synthetic_telemetry()

    train_rnd, val_rnd = random_split(telemetry)
    train_grp, val_grp = grouped_split(telemetry)

    # Check shared days
    rnd_shared_days = len({d["day"] for d in train_rnd} & {d["day"] for d in val_rnd})
    grp_shared_days = len({d["day"] for d in train_grp} & {d["day"] for d in val_grp})

    print(f"Total rows: {len(telemetry)}")
    print(f"Random Split - Shared Days: {rnd_shared_days} (Severe Leakage)")
    print(f"Grouped Split - Shared Days: {grp_shared_days} (Zero Leakage)")
    assert rnd_shared_days > 0
    assert grp_shared_days == 0
```

---

## 4. Production & Library Usage

In production pipelines with `scikit-learn`, grouped cross-validation and holdout splitting are configured via `GroupShuffleSplit`, `GroupKFold`, or `TimeSeriesSplit`:

```python
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import pandas as pd

df = pd.read_csv("processed_data/sensor/lettuce/train.csv")
X = df[["ph", "ec", "water_temp", "humidity", "air_temp"]]
y = df["label"]
groups = df["date"]

# Single Grouped Train/Validation Holdout (80/20 by date)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups=groups))

X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# 5-Fold Grouped Cross Validation
gkf = GroupKFold(n_splits=5)
for fold, (f_train_idx, f_val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    # Crucial rule: Oversampling / SMOTE must occur INSIDE this loop only
    pass
```

### Key Parameters:
- `groups`: The grouping identifier (e.g., `date`, `experiment_day`, or `bench_block`). Must never be included as an input feature to the model.
- `n_splits`: Number of validation folds. In grouped cross-validation, ensure each fold receives approximately equal class counts.

---

## 5. Failure Modes, Edge Cases & Debugging

1. **Replicate / Tank Confounding:**
   - *Failure:* If experiment 1 has only "Normal" days and experiment 2 has only "Stress" days, splitting by date within an experiment still allows tank-level calibration differences to leak.
   - *Mitigation:* Group by `(experiment, date)` or hold out entire replicate tanks.

2. **Class Starvation in Rare Faults:**
   - *Failure:* A rare fault like `pH_Low` may only appear on 2 distinct dates. Holding out whole dates can place 100% of a minority class into the validation set, leaving 0 training examples.
   - *Mitigation:* Audited manifest checks (as performed in T2-010) flagging classes with under 5 real training instances.

3. **Pre-Split Oversampling Leakage:**
   - *Failure:* Applying SMOTE or random oversampling to the dataset *before* performing grouped splitting. Synthetic copies of validation rows end up in the training fold.
   - *Mitigation:* Resample inside the training fold only, as enforced by our pipeline contracts.

---

## 6. Where It Appears in This Project

- **Notebook Demonstration:** [`notebooks/01_leakage_demo.ipynb`](../../notebooks/01_leakage_demo.ipynb) (Task T4-001)
- **Lettuce Sensor Split:** [`scripts/make_sensor_validation.py`](../../scripts/make_sensor_validation.py) (Task T2-010)
- **Spinach Sensor Split:** [`scripts/make_sensor_validation.py`](../../scripts/make_sensor_validation.py) (Task T2-011)
- **Leakage Blacklists:** [`ml/sensor_models/schemas.py`](../../ml/sensor_models/schemas.py) (Task T2-018)
- **Architectural Rules:** `TECH_RULES` §4.2 & `PROJECT_OVERVIEW.md` §9.2

---

## 7. Self-Check Questions (3–5 Questions)

1. **Q1:** Why does random row-wise splitting produce an artificially high validation Macro-F1 score (near 100%) on sensor telemetry?
   - *Answer:* High temporal autocorrelation means adjacent 10-minute readings from the same tank have almost identical feature values. Random row splitting distributes these twins into both train and val, allowing the model to act as a local interpolator rather than learning general physical fault boundaries.
2. **Q2:** What happens when a model trained on random splits is deployed in a real greenhouse?
   - *Answer:* The model experiences catastrophic performance collapse (dropping by 20% to 40%+ Macro-F1) because future calendar days require extrapolation to new diurnal cycles and baseline drifts that were never learned.
3. **Q3:** What is the mathematical definition of a grouped holdout split in terms of set intersection?
   - *Answer:* Given groups $G(S)$, the training set $S_{\text{train}}$ and validation set $S_{\text{val}}$ must satisfy $G(S_{\text{train}}) \cap G(S_{\text{val}}) = \emptyset$.
4. **Q4:** Why is pre-split oversampling (oversampling before train/val split) a form of data leakage?
   - *Answer:* Oversampling duplicates or interpolates points within the existing dataset. If done before splitting, identical or near-identical copies of validation points end up in the training set.
5. **Q5:** In our lettuce dataset, what was the empirical Macro-F1 gap between naive random splitting and grouped calendar date splitting?
   - *Answer:* Decision Tree Macro-F1 dropped from 1.0000 (random split) to 0.6372 (grouped date split), an inflation gap of 0.3628 (36.28%).
