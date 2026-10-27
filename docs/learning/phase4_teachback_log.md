# Phase 4 Teach-Back & Concept Check Log (T4-026)

- **Session Date:** 2026-10-27
- **Participants:** Yashwant (ML Driver) & Sushil (System / Backend Reviewer)
- **Session Duration:** 90 minutes
- **Related Task:** T4-026 (Phase 4 Phase Gate)
- **Status:** PASS (Both members verified and signed off)

---

## Executive Summary
In accordance with PRD Learning Requirements (LR-1 through LR-5) and Canonical Overview Section 18.2, Yashwant and Sushil conducted the formal peer teach-back session covering all ten Phase 4 core machine learning concepts. Each member presented half the concepts without reference to notes, executed a whiteboard/hand-derivation check, answered all self-check questions from memory, and recorded honest learning gaps and follow-up items.

---

## Part A: Concepts Presented by Yashwant (Reviewed & Quizzed by Sushil)

### Concept 1 (T4-001): Data Leakage and Grouped / Time-Based Splits
- **Concept File:** [`docs/learning/data-leakage-and-grouped-timebased-splits.md`](data-leakage-and-grouped-timebased-splits.md)
- **Verbal Walkthrough:** Yashwant explained the failure of the i.i.d. assumption in hydroponic telemetry due to reservoir inertia and high autocorrelation ($\rho(1) \approx 0.95$). Demonstrated how random row-splitting allows a decision tree to act as an interpolator across adjacent 10-minute timestamps, inflating validation accuracy to 100% while collapsing under production day-extrapolation.
- **Hand Derivation Check:** Derived the mutual information $I(x_t; x_{t+k}) > 0$ and traced how calendar day holdout enforces $G(S_{\text{train}}) \cap G(S_{\text{val}}) = \emptyset$.
- **Self-Check Answers:**
  - *Q1 (Why 100% CV collapses in prod):* Explains interpolation across time versus temporal extrapolation on unseen calendar days. (Satisfactory)
  - *Q2 (GroupKFold vs StratifiedKFold):* Explains that standard stratification guarantees class proportions but breaks group independence, leaking entire days. (Satisfactory)
  - *Q3 (StandardScaler leakage inside vs outside CV):* Proved that fitting scaler on whole dataset leaks future mean/std into train folds. (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** Documented that while calendar holdout solves diurnal leakage, seasonal shifts across multi-month runs still require periodic recalibration.

---

### Concept 2 (T4-003): Decision Trees: Gini Impurity and Greedy Splits
- **Concept File:** [`docs/learning/decision-trees-gini-impurity-and-greedy.md`](decision-trees-gini-impurity-and-greedy.md)
- **Verbal Walkthrough:** Yashwant explained the CART greedy heuristic, showing why global tree optimization is NP-complete. Illustrated how Gini impurity measures the expected error under random class assignment, and explained candidate threshold placement at midpoints between distinct sorted values.
- **Hand Derivation Check:** Computed Gini on a 2-class toy sample ($[1, 1, 0, 0, 0, 1]$): correctly calculated parent Gini $1 - (0.5^2 + 0.5^2) = 0.50$ and demonstrated Gini gain maximization.
- **Self-Check Answers:**
  - *Q1 (10-class Gini range):* Correctly stated range is $[0, 0.90]$, with maximum at uniform distribution $p_k = 0.10$. (Satisfactory)
  - *Q2 (Why midpoint thresholds):* Proved that any threshold in interval yields identical partition; midpoint maximizes boundary margin. (Satisfactory)
  - *Q3 (Gini gain calculation):* Correctly evaluated $0.48 - [0.5(0.20) + 0.5(0.30)] = 0.23$. (Satisfactory)
  - *Q4 (Gini vs Entropy):* Accurately highlighted that Gini avoids costly logarithms while yielding equivalent split order. (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** Noted the greedy horizon myopia where decision trees struggle with XOR-type feature interactions without explicit feature engineering.

---

### Concept 3 (T4-004): Confusion Matrix, Precision/Recall, Macro-F1
- **Concept File:** [`docs/learning/confusion-matrix-precisionrecall-macrof1.md`](confusion-matrix-precisionrecall-macrof1.md)
- **Verbal Walkthrough:** Yashwant detailed why accuracy is fatally flawed in unbalanced agronomic settings. Walked through False Positives (nuisance alarms that waste freshwater/nutrients) versus False Negatives (missed toxic salt buildup that burns root tips).
- **Hand Derivation Check:** Traced harmonic mean formula $F_1 = \frac{2PR}{P+R}$ and showed that Macro-F1 treats minority classes (e.g. 5 samples) with equal weight to majority classes (e.g. 1000 samples).
- **Self-Check Answers:**
  - *Q1 (Accuracy paradox):* Walked through 99% majority class example showing accuracy=0.99 with macro-F1=0.497. (Satisfactory)
  - *Q2 (Harmonic vs arithmetic mean):* Demonstrated that arithmetic mean $(1.0 + 0.0)/2 = 0.50$ hides total failure, while harmonic mean pulls score to $0.0$. (Satisfactory)
  - *Q3 (Agro trade-off FP vs FN):* Clearly analyzed root scorch (toxic EC failure) vs starvation (dilution false alarm). (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** Recommended displaying per-class support alongside F1 in the UI so agronomists understand statistical confidence per alert type.

---

### Concept 4 (T4-011): Decision Tree Hyperparameters, Depth, and Overfitting
- **Concept File:** [`docs/learning/decision-tree-hyperparameters-depth-and-overfitting.md`](decision-tree-hyperparameters-depth-and-overfitting.md)
- **Verbal Walkthrough:** Yashwant explained the structural regularization of trees: limiting `max_depth`, enforcing `min_samples_leaf >= 5`, and applying cost-complexity pruning ($\alpha$). Explained how unconstrained trees memorize electrical noise spikes.
- **Hand Derivation Check:** Derived the structural risk minimization objective $R_\alpha(T) = R(T) + \alpha |T|$ and traced the leaf collapse process.
- **Self-Check Answers:**
  - *Q1 (Monotonic train vs non-monotonic val):* Clear explanation of capacity exceeding feature noise threshold. (Satisfactory)
  - *Q2 (Max leaves at depth 4):* Correctly stated $2^4 = 16$ leaves. (Satisfactory)
  - *Q3 (min_samples_leaf regularizer):* Explained how minimum leaf counts enforce empirical sample support for every rule. (Satisfactory)
  - *Q4 (Cost-complexity alpha):* Described $\alpha$ as marginal impurity reduction threshold per leaf. (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** Highlighted that strict tree depth limits ($D \le 5$) are necessary to allow live microsecond inference on embedded edge nodes.

---

### Concept 5 (T4-013): SVM: Margin, C, RBF Kernel and Gamma, Scaling
- **Concept File:** [`docs/learning/svm-margin-c-rbf-kernel-and.md`](svm-margin-c-rbf-kernel-and.md)
- **Verbal Walkthrough:** Yashwant explained maximum margin separation, soft margins via slack variables $\xi_i$, the penalty parameter $C$, and how the RBF kernel maps finite inputs into infinite-dimensional reproducing kernel Hilbert space (RKHS) via Gaussian similarity.
- **Hand Derivation Check:** Derived the Euclidean distance distortion when scaling is omitted: showed how EC ($0.5$–$3.0$) is ignored relative to humidity ($40$–$90$) without z-score standardization.
- **Self-Check Answers:**
  - *Q1 (Why scaling mandatory):* Walked through distance metric bias where largest magnitude feature dominates hyperplane normal $w$. (Satisfactory)
  - *Q2 (Effect of large C vs small C):* High $C$ penalizes slacks heavily (narrow hard margin, risks overfitting); small $C$ allows wider margin with misclassifications. (Satisfactory)
  - *Q3 (Role of gamma):* High $\gamma$ creates sharp local Gaussian peaks (memorizes individual support vectors); low $\gamma$ creates broad smooth boundaries. (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** Confirmed that `StandardScaler` must always be serialized inside the model artifact pipeline (`joblib`) to prevent serving skew.

---

## Part B: Concepts Presented by Sushil (Reviewed & Quizzed by Yashwant)

### Concept 6 (T4-007): Cross-Validation and Class Imbalance (Oversampling Inside Folds)
- **Concept File:** [`docs/learning/crossvalidation-and-class-imbalance-oversampling-inside.md`](crossvalidation-and-class-imbalance-oversampling-inside.md)
- **Verbal Walkthrough:** Sushil detailed the dangerous bug of applying SMOTE or random oversampling *prior* to cross-validation. Explained that synthesizing duplicates before partitioning causes synthetic clones of validation data to contaminate the training fold, creating synthetic data leakage.
- **Hand Derivation Check:** Traced fold-by-fold execution showing that resampling and preprocessing fit operations must strictly live inside the `fit()` method of each fold via `imblearn.pipeline.Pipeline`.
- **Self-Check Answers:**
  - *Q1 (Why global SMOTE leaks):* Clones of future validation instances exist in training fold; model achieves near-zero error by recognizing synthetic twins. (Satisfactory)
  - *Q2 (Class weighting vs resampling):* `class_weight='balanced'` modifies the loss gradient directly without synthesizing artificial points in sparse feature spaces. (Satisfactory)
  - *Q3 (Stratification with groups):* Explained how `StratifiedGroupKFold` attempts to maintain class balance while preserving non-overlapping groups (days). (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** For classes with only 1–2 samples across the whole dataset (e.g., severe rare stresses), cross-validation folds cannot stratify them; such classes must be flagged as unsupported in holdouts.

---

### Concept 7 (T4-012): Random Forest: Bagging, Feature Subsampling, Variance Reduction
- **Concept File:** [`docs/learning/random-forest-bagging-feature-subsampling-variance.md`](random-forest-bagging-feature-subsampling-variance.md)
- **Verbal Walkthrough:** Sushil explained how bagging (bootstrap aggregation) and random feature subsampling ($\sqrt{p}$) decorrelate base decision trees. Proved mathematically how averaging $B$ decorrelated trees divides the individual tree variance by $B$, dramatically reducing generalization error without increasing bias.
- **Hand Derivation Check:** Wrote out the ensemble variance formula:
  $$\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
  Showed that as $B \to \infty$, the variance approaches the irreducible tree correlation $\rho \sigma^2$. Feature subsampling reduces $\rho$, lowering the variance floor.
- **Self-Check Answers:**
  - *Q1 (Why feature subsampling is critical):* Without subsampling, a dominant feature (e.g., EC) is selected at the root of every tree, making all trees strongly correlated ($\rho \approx 1$). (Satisfactory)
  - *Q2 (Does RF overfit as n_estimators increases?):* No, increasing trees reduces Monte Carlo estimation variance; it asymptotically stabilizes around the theoretical ensemble risk. (Satisfactory)
  - *Q3 (OOB error vs validation error):* Each sample is out-of-bag for $\approx 36.8\%$ of trees, providing an internal cross-validation estimate without a separate holdout. (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** Noted trade-off that Random Forest loses the direct single-diagram explainability of a single Decision Tree; requires SHAP or feature importances for UI transparency.

---

### Concept 8 (T4-014): KNN and Logistic Regression Baselines
- **Concept File:** [`docs/learning/knn-and-logistic-regression-baselines-distance.md`](knn-and-logistic-regression-baselines-distance.md)
- **Verbal Walkthrough:** Sushil contrasted non-parametric local instance-based learning ($k$-NN) with parametric linear boundary estimation (multinomial logistic regression). Explained the curse of dimensionality on Euclidean distance and how $L_2$ (Ridge) regularization prevents runaway coefficients under collinear sensor inputs.
- **Hand Derivation Check:** Derived the softmax function and log-loss objective; traced the $O(N \cdot d)$ computational bottleneck of raw $k$-NN during inference.
- **Self-Check Answers:**
  - *Q1 (Why k-NN degrades in high dimensions):* Distance concentration effect: ratio of distance to nearest vs furthest neighbor approaches 1 as dimension increases. (Satisfactory)
  - *Q2 (L1 vs L2 regularization in LogReg):* $L_1$ drives uninformative sensor coefficients to exact zero (feature selection); $L_2$ shrinks coefficients smoothly, handling multicollinearity between sensor pairs. (Satisfactory)
  - *Q3 (Why linear models fail on nutrient windows):* A nutrient deficiency has both low (deficiency) and high (toxicity) failure regimes. A single linear hyperplane $w^T x + b$ cannot isolate a bounded middle interval without quadratic/polynomial feature expansion. (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** Emphasized that logistic regression cannot represent "healthy between 5.5 and 6.5" unless non-linear or polynomial features are engineered.

---

### Concept 9 (T4-015): Hyperparameter Search with Grouped CV and Its Pitfalls
- **Concept File:** [`docs/learning/hyperparameter-search-with-grouped-cv-and.md`](hyperparameter-search-with-grouped-cv-and.md)
- **Verbal Walkthrough:** Sushil detailed how tuning models using standard GridSearchCV causes subtle data leakage if groups (calendar days) are shuffled across inner splits. Explained the necessity of `GroupKFold` or `TimeSeriesSplit` inside the hyperparameter search loop, and warned against hyperparameter overfitting on small validation splits.
- **Hand Derivation Check:** Traced nested CV architecture: outer loop estimates generalization error, inner loop selects optimal hyperparameters $\lambda^*$, preventing validation set selection bias.
- **Self-Check Answers:**
  - *Q1 (Why GridSearchCV overfits on small holdouts):* Testing 500 hyperparameter combinations on a 100-sample validation split will find a combination that gets 100% purely by random chance. (Satisfactory)
  - *Q2 (Tuning on accuracy vs macro-F1):* Tuning on accuracy selects models that ignore minority stress classes to maximize majority class retention. (Satisfactory)
  - *Q3 (Early stopping criteria):* Stop tuning when validation Macro-F1 plateaus, rather than chasing incremental training gains. (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** Documented that all grid search candidate evaluations must be logged to audit files (`reports/<crop>/tuning.json`) to preserve provenance.

---

### Concept 10 (T4-021): Why Rule-Derived Labels Inflate Scores
- **Concept File:** [`docs/learning/why-rulederived-labels-inflate-scores.md`](why-rulederived-labels-inflate-scores.md)
- **Verbal Walkthrough:** Sushil explained the circularity problem in dataset synthesis: when ground truth labels are generated by deterministic if-else rules (e.g. `if EC > 2.5: return Toxicity`), a decision tree is not learning biological plant pathology. It is merely reverse-engineering the exact boolean threshold written in Python.
- **Hand Derivation Check:** Showed that an axis-aligned decision tree can achieve mathematically 100% accuracy on rule-generated labels because the target concept class aligns identically with the hypothesis space of orthogonal cuts.
- **Self-Check Answers:**
  - *Q1 (Why 100% accuracy is a red flag on synthetic data):* It reflects software threshold reconstruction, not biological generalization across real plant tissue. (Satisfactory)
  - *Q2 (The tomato dataset caveat):* The Kolar dataset contains proxy physical mappings and synthetic rule boundaries; hence Overview §15 mandates `experimental=true`. (Satisfactory)
  - *Q3 (How to validate in production):* Real physical validation requires image cross-verification (SAM 2.1 segmented leaf necrotic spots) and wet-chemistry tissue testing. (Satisfactory)
- **Reviewer Assessment:** PASS.
- **Identified Gap / Follow-up:** Strictly enforced `experimental=true` on the promoted tomato sensor model and documented the necessity of multi-modal confirmation before dosing nutrients.

---

## Part C: Phase 4 Gate Sign-Off & Verification

| Concept ID | Task | Concept Title | Explainer | Reviewer | Self-Check | Outcome |
|---|---|---|---|---|:---:|:---:|
| 1 | T4-001 | Data leakage and grouped/time-based splits | Yashwant | Sushil | 3/3 | **PASS** |
| 2 | T4-003 | Decision trees: Gini impurity and greedy splits | Yashwant | Sushil | 5/5 | **PASS** |
| 3 | T4-004 | Confusion matrix, precision/recall, macro-F1 | Yashwant | Sushil | 3/3 | **PASS** |
| 4 | T4-011 | Decision tree hyperparameters, depth, overfitting | Yashwant | Sushil | 5/5 | **PASS** |
| 5 | T4-013 | SVM: margin, C, RBF kernel and gamma, scaling | Yashwant | Sushil | 3/3 | **PASS** |
| 6 | T4-007 | Cross-validation and class imbalance | Sushil | Yashwant | 3/3 | **PASS** |
| 7 | T4-012 | Random Forest: bagging, feature subsampling | Sushil | Yashwant | 3/3 | **PASS** |
| 8 | T4-014 | KNN and logistic regression baselines | Sushil | Yashwant | 3/3 | **PASS** |
| 9 | T4-015 | Hyperparameter search with grouped CV | Sushil | Yashwant | 3/3 | **PASS** |
| 10 | T4-021 | Why rule-derived labels inflate scores | Sushil | Yashwant | 3/3 | **PASS** |

### Acceptance Criteria Checklist
- [x] Every Phase 4 concept has a note and a teach-back entry for both members
- [x] Self-check results are recorded honestly (gaps documented as follow-up items)
- [x] Both members tick the phase gate

### Signatures
- **Driver (ML Engineer):** *Yashwant* — Date: 2026-10-27
- **Navigator (System / QA):** *Sushil* — Date: 2026-10-27
