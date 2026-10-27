# Machine Learning Learning Progress & Tracker

Tracks mastery, concept notes, teach-backs, and self-checks across all machine learning concepts in the project per PRD §Learning Requirements (LR-1 to LR-5).

## Rules of Engagement
1. **Navigator Writes Concept Note:** The navigator of the implementation task writes the concept note in docs/learning/ using _TEMPLATE_concept.md.
2. **Teach-Back Required:** The other teammate delivers a verbal walkthrough and answers the 3–5 self-check questions using _TEMPLATE_teachback.md.
3. **Equal Mastery:** Both teammates must complete both checks (concept note written/reviewed, teach-back done, self-check passed) before closing the project phase.

---

## Concept Tracking Table

| # | Task | Concept | Concept Note File | Yashwant Note | Yashwant Teach-Back | Yashwant Self-Check | Yashwant Date | Sushil Note | Sushil Teach-Back | Sushil Self-Check | Sushil Date |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | T1-007 | Foundation-model segmentation: prompts, masks, and failure modes (SAM 2.1) | `docs/learning/foundationmodel-segmentation-prompts-masks-and-failure.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 2 | T2-005 | Train/test splits, oversampling, and day-based (grouped) separation | `docs/learning/traintest-splits-oversampling-and-daybased-grouped.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 3 | T2-010 | Hold-out validation design for time-correlated data | `docs/learning/holdout-validation-design-for-timecorrelated-data.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 4 | T2-014 | Near-duplicate detection and why duplicates across splits inflate image accuracy | `docs/learning/nearduplicate-detection-and-why-duplicates-across.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 5 | T4-001 | Data leakage and grouped / time-based splits | `docs/learning/data-leakage-and-grouped-timebased-splits.md` | [x] | [ ] | [ ] | 2026-10-26 | [ ] | [ ] | [ ] | - |
| 6 | T4-003 | Decision trees: Gini impurity and greedy splits | `docs/learning/decision-trees-gini-impurity-and-greedy.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 7 | T4-004 | Confusion matrix, precision/recall, macro-F1 | `docs/learning/confusion-matrix-precisionrecall-macrof1.md` | [x] | [ ] | [ ] | 2026-10-26 | [ ] | [ ] | [ ] | - |

| 8 | T4-007 | Cross-validation and class imbalance (oversampling inside folds) | `docs/learning/crossvalidation-and-class-imbalance-oversampling-inside.md` | [ ] | [ ] | [ ] | - | [x] | [ ] | [ ] | 2026-10-27 |
| 9 | T4-011 | Decision tree hyperparameters, depth, and overfitting | `docs/learning/decision-tree-hyperparameters-depth-and-overfitting.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 10 | T4-012 | Random Forest: bagging, feature subsampling, variance reduction | `docs/learning/random-forest-bagging-feature-subsampling-variance.md` | [ ] | [ ] | [ ] | - | [x] | [ ] | [ ] | 2026-10-27 |
| 11 | T4-013 | SVM: margin, C, RBF kernel and gamma, why scaling matters | `docs/learning/svm-margin-c-rbf-kernel-and.md` | [x] | [ ] | [ ] | 2026-10-27 | [ ] | [ ] | [ ] | - |
| 12 | T4-014 | KNN and logistic regression baselines (distance, regularisation) | `docs/learning/knn-and-logistic-regression-baselines-distance.md` | [ ] | [ ] | [ ] | - | [x] | [ ] | [ ] | 2026-10-27 |
| 13 | T4-015 | Hyperparameter search with grouped CV and its pitfalls | `docs/learning/hyperparameter-search-with-grouped-cv-and.md` | [x] | [ ] | [ ] | 2026-10-27 | [ ] | [ ] | [ ] | - |
| 14 | T4-020 | Train/serve skew and feature availability (live-compatible models) | `docs/learning/trainserve-skew-and-feature-availability-livecompatible.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 15 | T4-021 | Why rule-derived labels inflate scores | `docs/learning/why-rulederived-labels-inflate-scores.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 16 | T5-001 | Colour spaces (BGR/RGB/HSV) and vegetation indices (ExG, ExR) | `docs/learning/colour-spaces-bgrrgbhsv-and-vegetation-indices.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 17 | T5-003 | Texture features: GLCM contrast and entropy | `docs/learning/texture-features-glcm-contrast-and-entropy.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 18 | T5-007 | Shape descriptors: compactness, solidity, aspect ratio | `docs/learning/shape-descriptors-compactness-solidity-aspect-ratio.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 19 | T5-010 | Instance segmentation for per-leaf prediction (SAM automatic masks) | `docs/learning/instance-segmentation-for-perleaf-prediction-sam.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 20 | T5-012 | Weak labels and instance-level (per-leaf) prediction | `docs/learning/weak-labels-and-instancelevel-perleaf-prediction.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 21 | T5-017 | Convolution, pooling, and backpropagation | `docs/learning/convolution-pooling-and-backpropagation.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 22 | T5-018 | Data augmentation (training only) and input normalisation | `docs/learning/data-augmentation-training-only-and-input.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 23 | T5-020 | Optimisers, learning rate, and early stopping on validation macro-F1 | `docs/learning/optimisers-learning-rate-and-early-stopping.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 24 | T5-022 | Transfer learning and fine-tuning (EfficientNetB0) | `docs/learning/transfer-learning-and-finetuning-efficientnetb0.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 25 | T5-031 | Softmax confidence, calibration, and uncertainty thresholds | `docs/learning/softmax-confidence-calibration-and-uncertainty-thresholds.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |
| 26 | T10-003 | Probability calibration (temperature scaling, reliability diagrams) | `docs/learning/probability-calibration-temperature-scaling-reliability-diagrams.md` | [ ] | [ ] | [ ] | - | [ ] | [ ] | [ ] | - |

---

## Summary Statistics
- **Total Unique Concepts:** 26
- **Yashwant Completed:** 0 / 26
- **Sushil Completed:** 0 / 26
