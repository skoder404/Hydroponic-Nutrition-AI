# Concept: Optimisers, Learning Rate, and Early Stopping on Validation Macro-F1

- **Concept File:** `docs/learning/optimisers-learning-rate-and-early-stopping.md`
- **Related Task:** T5-020
- **Driver:** Sushil
- **Author / Navigator:** Yashwant Vadhan & Sushil
- **Date:** 2026-10-28
- **Status:** Completed

---

## 1. Intuition

Training deep convolutional neural networks is fundamentally an empirical optimization problem over a high-dimensional, non-convex loss landscape. The optimizer governs how weight parameters $\theta \in \mathbb{R}^D$ step across this surface based on stochastic gradient estimates computed on mini-batches of leaf imagery.

If the **learning rate** $\eta$ is set too large, weight updates overshoot narrow valleys, causing the loss to oscillate or diverge to `NaN`. Conversely, if $\eta$ is set too small, parameter updates stall in shallow plateaus or saddle points, requiring excessive training epochs and risking convergence to poor local minima.

Modern adaptive optimizers—most notably **AdamW** (Adam with decoupled Weight Decay)—maintain running exponentially decayed estimates of past gradients (momentum) and squared gradients (uncentered variance). This adaptively scales coordinate-wise step sizes, accelerating progress along flat directions while damping oscillations in steep canyons.

Finally, neural networks possess millions of parameters and can trivially memorize training samples (achieving zero training error while failing completely on unseen crops). To prevent overfitting, we implement **Early Stopping**. Crucially, in agronomic disease and nutrient classification where healthy leaves heavily outnumber rare deficiencies, stopping on *validation accuracy* produces degenerate models that ignore minority classes. We strictly stop on **Validation Macro-F1** ($\text{Macro-F1}_{\text{val}}$), halting training when minority class sensitivity peaks.

---

## 2. Mathematical Formulation & Formal Definition

### 2.1 The AdamW Optimization Update Rules

Let $\mathcal{L}(\theta)$ be the cross-entropy objective on mini-batch $B_t$, and $g_t = \nabla_\theta \mathcal{L}_t(\theta_{t-1})$ be the objective gradient at step $t$.

1. **Biased First and Second Moment Estimates:**
   $$
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
   $$
   $$
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   $$
   where standard hyperparameters are $\beta_1 = 0.9$ (momentum) and $\beta_2 = 0.999$ (second-moment scale).

2. **Bias Correction:**
   Because $m_0 = 0$ and $v_0 = 0$, the initial estimates are biased towards zero:
   $$
   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
   $$

3. **Decoupled Weight Decay & Parameter Update (Loshchilov & Hutter, 2019):**
   In standard Adam with L2 regularization, the weight penalty $\lambda \theta$ is folded into $g_t$, which inadvertently scales the weight decay by $\frac{1}{\sqrt{\hat{v}_t}}$. AdamW decouples weight decay directly from gradient scaling:
   $$
   \theta_t = \theta_{t-1} - \eta_t \lambda \theta_{t-1} - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
   $$
   where $\lambda$ is the weight decay coefficient (e.g. $10^{-4}$), $\eta_t$ is the scheduled learning rate, and $\epsilon \approx 10^{-8}$ prevents division by zero.

### 2.2 Validation Macro-F1 Early Stopping Metric

Let $C$ be the number of target classes in the crop taxonomy. For each class $c \in \{1, \dots, C\}$ evaluated on the holdout validation partition $\mathcal{D}_{\text{val}}$:
$$
\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}, \quad \text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}
$$
$$
\text{F1}_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c + \epsilon}
$$
The unweighted Macro-F1 score is:
$$
\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^C \text{F1}_c
$$

**Stopping Condition:**
Let $S_k = \text{Macro-F1}^{(k)}_{\text{val}}$ be the validation Macro-F1 at epoch $k$.
We maintain the champion score $S^* = \max_{j \le k} S_j$.
If $S_k < S^* + \delta$ for $P$ consecutive epochs (where $P$ is patience, e.g. 5), training halts early, and the checkpoint corresponding to $S^*$ is restored.

---

## 3. From-Scratch Implementation (NumPy / Pure Python)

```python
import numpy as np


class AdamWFromScratch:
    """Minimal from-scratch implementation of AdamW optimizer."""

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            # 1. Update first moment
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            # 2. Update second moment
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g**2)

            # 3. Bias correction
            m_hat = self.m[i] / (1.0 - self.beta1**self.t)
            v_hat = self.v[i] / (1.0 - self.beta2**self.t)

            # 4. Decoupled weight decay update
            p -= self.lr * self.weight_decay * p
            # 5. Adaptive gradient update
            p -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))


class EarlyStoppingTracker:
    """Monitors validation Macro-F1 and signals when patience runs out."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float = -float("inf")
        self.wait_count: int = 0
        self.should_stop: bool = False

    def update(self, val_macro_f1: float) -> bool:
        """Returns True if a new best checkpoint was achieved."""
        if val_macro_f1 > self.best_score + self.min_delta:
            self.best_score = val_macro_f1
            self.wait_count = 0
            return True

        self.wait_count += 1
        if self.wait_count >= self.patience:
            self.should_stop = True
        return False


if __name__ == "__main__":
    # Toy verification test
    p = np.array([2.0, -3.0], dtype=np.float64)
    optimizer = AdamWFromScratch([p], lr=0.1, weight_decay=0.01)

    # Minimize f(p) = p^2 (optimal at 0.0)
    for _ in range(50):
        grad = 2.0 * p
        optimizer.step([grad])

    assert np.all(np.abs(p) < 0.05), f"Failed convergence: {p}"
    print("AdamW from scratch test passed! Optimal parameters:", p)

    # Test Early Stopping Tracker
    tracker = EarlyStoppingTracker(patience=3)
    scores = [0.40, 0.45, 0.50, 0.49, 0.48, 0.47]
    stops = []
    for s in scores:
        is_best = tracker.update(s)
        stops.append(tracker.should_stop)

    assert tracker.should_stop is True
    print("Early stopping test passed! Halted when patience exhausted.")
```

---

## 4. Production & Library Usage

In `ml/image_models/train_image.py`:

```python
import torch
from torch import nn, optim
from ml.image_models.custom_cnn import CustomCNN

model = CustomCNN(crop="lettuce")
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)
```

**Key Hyperparameters & Trade-offs:**
- `learning_rate = 1e-3`: Standard initial learning rate for AdamW on scratch CNNs.
- `weight_decay = 1e-4`: Regularizes large weights without distorting gradient direction.
- `early_stopping_patience = 5`: Permits temporary validation fluctuations across 5 epochs before terminating.

---

## 5. Failure Modes, Edge Cases & Debugging

1. **Overfitting to Majority Classes (Accuracy Masking):**
   - *Symptom:* Overall accuracy remains steady at 85%, but the model has zero recall on iron or nitrogen chlorosis.
   - *Mitigation:* Early stopping evaluates exclusively on **Macro-F1**, where failing any single deficiency class severely degrades the criterion score.
2. **CUDA Out-of-Memory (OOM) on Large Batches:**
   - *Symptom:* `torch.cuda.OutOfMemoryError` aborts training mid-epoch.
   - *Mitigation:* DeviceManager catches CUDA OOM, runs garbage collection, halves the batch size, and falls back to CPU if memory is constrained.
3. **Data Leakage from Test Holdout:**
   - *Symptom:* Unrealistically high validation scores that collapse on field imagery.
   - *Mitigation:* The training loop never opens or references `test.csv`; model selection is strictly confined to `val.csv`.

---

## 6. Where It Appears in This Project

- **File / Module:** [`ml/image_models/train_image.py`](../../ml/image_models/train_image.py)
- **Role in Pipeline:** Generic PyTorch training loop powering Custom CNN and transfer learning classifiers across lettuce, spinach, and tomato crops.
- **Relevant Contracts / Rules:** Canonical Overview §18.1, §18.2; TECH_RULES Rule 17 (holdout isolation).

---

## 7. Self-Check Questions (3–5 Questions)

1. **Why does AdamW decouple weight decay instead of using standard L2 regularization?**
   - *Answer:* Standard Adam folds weight decay into the gradient vector $g_t$, which causes weights associated with large historical gradients to experience attenuated decay, while weights with small historical gradients decay too rapidly. Decoupling ensures every weight decays proportionally to its magnitude independent of gradient variance.
2. **Why is validation Macro-F1 used for early stopping instead of validation cross-entropy loss or validation accuracy?**
   - *Answer:* Validation loss can continue decreasing as the model becomes overconfident on easy/majority samples while misclassifying minority deficiency classes. Macro-F1 directly penalizes poor sensitivity on rare conditions.
3. **What is an "overfit test" and why is it conducted before full training runs?**
   - *Answer:* An overfit test trains on a tiny fixed subset (e.g. 20 samples) without regularization. Reaching near-zero loss proves that gradients are flowing properly, the loss and optimizer are implemented correctly, and the model has sufficient representational capacity.
4. **How does the training loop maintain Device Honesty?**
   - *Answer:* It queries `DeviceManager`, verifies tensor execution on the actual compute hardware, and records `device_actually_used` in the output metadata and evaluation report.
5. **What happens when a CUDA OOM occurs during a forward/backward pass?**
   - *Answer:* The OOM ladder cleans VRAM via `gc.collect()` and `torch.cuda.empty_cache()`, retries with half the batch size, and if unresolvable, gracefully falls back to CPU while logging the hardware switch.
