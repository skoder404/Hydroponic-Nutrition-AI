# PRD — Hydroponic Nutrition AI

> **Status:** Draft v0.2 · **Date:** 2026-10-20 · **Owners:** Yashwant, Sushil Kumar M A
> **Source of truth:** [`PROJECT_OVERVIEW.md`](./PROJECT_OVERVIEW.md). This PRD *derives from* it and never overrides it. Where this PRD and the overview disagree, the overview wins until both are deliberately updated.
> **Truth labels** (same as the overview): **Confirmed**, **Planned default**, **Must verify in repository**, **Limitation**.
> Items marked `⚠️ UNCLEAR` are open questions; they are collected in [Assumptions & Open Questions](#assumptions--constraints).

---

## Executive Summary

| Field | Value |
|---|---|
| **Project name** | Hydroponic Nutrition AI |
| **Problem statement** | Hydroponic growers (and campus lab users) must judge crop condition from leaf appearance and water/air readings. This is slow, subjective, and error-prone, and existing ML demos usually hide uncertainty and overclaim accuracy. |
| **Solution overview** | A web-based decision-support tool for **lettuce, spinach, and tomato**. It runs image-based classification (custom CNN, EfficientNetB0, OpenCV features + classical ML), SAM 2.1 segmentation, and sensor-based condition classification (Decision Tree, Random Forest, SVM, others). It shows image and sensor results *separately*, with explicit uncertainty, and produces conservative advisory recommendations (Gemini with a deterministic rule-based fallback). |
| **Target audience** | (1) Student growers / campus hydroponics operators, (2) course evaluators and reviewers, (3) the two developers-as-learners. |
| **Secondary goal (project-level)** | **Equal ML learning for both team members.** Every ML concept implemented must be understood and explainable by *both* Yashwant and Sushil (see [Learning Requirements](#learning-requirements-lr)). |

**Non-negotiable product principles** (from the overview §1, §3, §19, §28):

1. A model output is a **prediction**, not a diagnosis.
2. Rule-derived sensor labels are **agreement with project rules**, not observed plant-health outcomes.
3. Image and sensor outputs are **not paired** and are never merged into one score without a validated mapping.
4. Missing data is **missing**, never zero. Historical data is **never shown as live**.
5. Recommendations are **advisory**; the app never actuates pumps or dosing.

---

## Resolved Decisions (2026-10-20)

| ID | Decision | Source |
|---|---|---|
| D1 | **No validation set exists for any crop.** Create one from the source CSV that was used for the train/test split: grouped hold-out (lettuce: calendar day; spinach: six-hour block; tomato: calendar month), made **before** oversampling, with the test partition left untouched (hash-guarded). Image data uses a 70/15/15 split. | Team |
| D2 | **No application/ML code exists yet.** The repo holds `image_data/` and `sensor_data/`; `sensor_data/` already includes per-crop preparation scripts and READMEs (`build_lettuce_dataset`, `generate_7class_dataset.py`, `build_tomato_dataset.py`) plus `*_full/_train/_test.csv`. These planning docs are added to the repo first. | Team |
| D3 | "MLM" = **Machine Learning Methods** (course project). | Team |
| D4 | **All three crops are trained with all model families:** images = custom CNN, EfficientNetB0, OpenCV features + RF/SVM; sensors = Decision Tree, Random Forest, SVM (+ KNN/Logistic Regression baselines). | Team |
| D5 | Datasets are unmodified copies from the pre-existing sources in the overview. Licences/versions still to be recorded. | Team |
| D6 | **Campus server:** sensor readings and the camera feed come from a server reachable at a given IP address. Protocol, endpoints and payloads are **not yet known** and will be probed on the campus network (task in `todo.md`). The IP is kept in `.env`, never committed. | Team |
| D7 | **UI:** earthy, nature-based, React, purposeful dynamic animation, clean and neat (see `DESIGN.md`). | Team |
| D9 | **Leaf-level image models.** All image models (OpenCV features + RF/SVM, custom CNN, EfficientNetB0) work on **individual leaves**. A whole-plant photo is split into leaves with SAM 2.1 and **each leaf gets its own class and confidence**; the plant summary is only a count of leaves and per-class counts, never an averaged or merged score. Lettuce leaf crops are generated from the whole-plant images and inherit the plant-level label (weak labels, stated as a limitation). Spinach and tomato CNNs/features use the leaf images as provided. | Team |
| D10 | **Facts verified from the sensor scripts and READMEs (2026-10-21).** Lettuce: 726 real labelled readings, 7 classes (incl. `Multiple_Stress`), test = every 4th calendar day per experiment, train oversampled to 240 per class, expected performance = grouped 5-fold CV macro-F1 0.837 ± 0.075 (single split 0.994 is one favourable split). Spinach: 1,400 rows, 7 classes, every non-Normal row breaks exactly one rule, test = whole six-hour blocks, balanced by drawing (no oversampling), TDS is derived. Tomato: 8 features, 10 sensor classes in default `primary` mode (label = most severe fault), test = random whole calendar months, **no validation partition**, Kolar source described by its own README as looking simulated, water temperature synthetic, soil-to-hydroponic mappings are proxy transforms. | Scripts + READMEs |
| D11 | **The campus system grows tomato only.** The Live Monitor and live inference default to tomato. Tomato sensor results stay experimental. If the campus server does not supply nitrogen/phosphorus/potassium, a live-compatible tomato variant is trained on the features it does supply. The raw Kolar source file is kept outside git with its checksum, build command, and the recorded severity scales. | Team |
| D8 | **On-disk layout is the source of truth for class labels** (`image_data/…` folder names). Observed difference from Overview §14: tomato has an extra class `Tomato_Powdery_Mildew`, giving **11** tomato classes (Healthy + 10 unhealthy); spinach has 6; lettuce has 4. The overview must be updated deliberately (task in `todo.md`). | Folder listing |

**Still open:** which sensor fields the campus server provides for tomato (N/P/K?); the real tomato row count and class count in the CSVs (overview says 44,141 rows, README estimates about 33,850); whether the raw sources (EXP_*.xlsx workbooks, Meng greenhouse CSV, Kolar CSV) are available (only needed to re-run scripts or recompute tomato severity scales); whether any class is too rare to appear in train, validation and test after the hold-out; SAM 2.1 leaf-separation quality (spike); the campus server interface (probe); the course deadline.

---

## Goals & Objectives

### Business / Project Goals
- **BG-1** Deliver a working, demonstrable end-to-end system (upload/enter → validate → infer → explain → recommend → save) for all three crops.
- **BG-2** Produce a *fair*, reproducible comparison of model families per crop (Custom CNN vs EfficientNetB0 vs OpenCV+classical ML; DT vs RF vs SVM vs others) with honest reporting.
- **BG-3** Produce a repository that a reviewer can clone, understand, and reproduce, with documented dataset provenance, split manifests, model registry, and tests.
- **BG-4** Demonstrate methodological rigor: leakage prevention, validation-only model selection, untouched test set, calibrated/declared uncertainty semantics.

### User Goals
- **UG-1** Grower: get a clear, plain-language read of crop condition and what to check next.
- **UG-2** Grower: know *how much to trust* each result and *what was not assessed*.
- **UG-3** Reviewer: inspect evidence (metrics, confusion matrices, limitations) without reading code.
- **UG-4** Developer-learner: understand *why* each algorithm behaves as it does, not just call `.fit()`.

### Success Metrics (KPIs)

> Per overview §2.3, **no production accuracy is promised** before final evaluation. KPIs below are *process, correctness, and completeness* metrics. Numeric latency figures are **planned targets to be measured**, not claims.

| ID | KPI | Target | How measured |
|---|---|---|---|
| K1 | Crop × task model coverage with full registry metadata (§24.1) | 100% of *promoted* models | Registry validation script |
| K2 | Reporting completeness per evaluated model (§18.2 checklist) | 100% of promoted models | Evaluation report lint |
| K3 | Leakage tests passing (§27.3) | 100% pass in CI | `pytest tests/data` |
| K4 | Model selection uses validation macro-F1 only; test set touched once at final eval | 0 violations | Evaluation-run audit log |
| K5 | Sensor inference API p95 latency (CPU, local) | ≤ 300 ms *(planned target; measure)* | Load script |
| K6 | Image inference end-to-end p95 latency incl. segmentation (CPU fallback) | ≤ 30 s *(planned target; SAM 2.1 is heavy; measure)* | Timing logs |
| K7 | Invalid/missing/stale input handled with a structured error (no silent zeros) | 100% of tested cases | Integration tests |
| K8 | Learning goal: each ML concept has a completed concept note + teach-back by *both* members | 100% of concepts | `docs/learning/` checklist |
| K9 | Backend unit+integration coverage | ≥ 70% line coverage *(planned)* | `pytest --cov` |

---

## User Personas

### P1 — Aarav, Campus Hydroponics Operator (primary)
- **Description:** Student/technician who tends lettuce/spinach/tomato in a campus NFT/hydroponic setup. Comfortable with a phone camera and a handheld pH/EC meter; not an ML person.
- **Pain points:** Can't tell nutrient deficiency vs disease at a glance; sensors are sometimes offline; afraid of over-trusting a black box.
- **Goals:** Quick check (photo and/or readings), plain-language result, know when to re-measure or ask an expert.

### P2 — Dr. Meera, Course Evaluator / Reviewer
- **Description:** Faculty member assessing ML methodology.
- **Pain points:** Projects that report inflated accuracy, leak data, or hide that labels are synthetic.
- **Goals:** See comparison tables, confusion matrices, split protocols, limitations, and reproducibility evidence quickly.

### P3 — Yashwant & Sushil, Developer-Learners
- **Description:** Two B.Tech AI&DS students building and learning together.
- **Pain points:** Risk that one member "does the ML" while the other does plumbing, leaving unequal understanding; risk of copy-pasting library calls without understanding.
- **Goals:** Equal depth on every concept; a plan that forces explain-back and rotates ownership.

---

## User Stories

### M1 — Crop & Mode Selection
- As a grower, I want to pick lettuce, spinach, or tomato so that the right models and input rules are used.
- As a grower, I want to choose image, sensors, or both so that I can use whatever data I have.

### M2 — Sensor Analysis
- As a grower, I want to enter readings with units so that the system can classify the condition.
- As a grower, I want a clear error naming the bad/missing field so that I can fix it instead of getting a wrong answer.
- As a grower, I want tomato sensor results marked *experimental/manual* unless compatibility is validated so that I'm not misled.

### M3 — Image Analysis
- As a grower, I want to upload a plant photo so that I get a class (healthy/unhealthy/<class>) with confidence.
- As a grower, I want a warning and a retry path if segmentation fails so that I never get a prediction from a bad mask.
- As a grower, I want low-confidence results labelled **uncertain** so that I don't act strongly on weak evidence.

### M4 — Evidence Presentation
- As a grower, I want image and sensor results side by side but separate so that I'm not shown a fake combined score.
- As a reviewer, I want to see which model version, preprocessing, and timestamp produced a result.

### M5 — Recommendations
- As a grower, I want conservative next-step suggestions that cite the evidence used.
- As a grower, I want the app to still give a safe suggestion if Gemini is down (rule-based fallback).
- As a grower, I want "insufficient information" instead of an invented fix or dosage.

### M6 — History
- As a grower, I want to save and revisit records with their true observation timestamp so that I can track changes.
- As a grower, I want old readings never labelled "live".

### M7 — Campus/Device Integration
- As an operator, I want the app to work with manual entry if hardware is unavailable.
- As an operator, I want stale/disconnected/malformed device data reported as *unavailable*.

### M8 — ML Lab (internal / reviewer-facing)
- As a developer, I want reproducible training scripts that write model artifacts + metadata to a registry.
- As a reviewer, I want evaluation reports with all required metrics and limitations.
- As a developer, I want explicit promotion and rollback of models.

### M9 — Learning Track (project-level)
- As Yashwant, I want to understand every algorithm we implement as deeply as Sushil does (and vice-versa).
- As a team, we want rotating driver/navigator roles, concept notes, and teach-back checks per concept.

---

## Functional Requirements

> Route names are **proposed** (Planned default) because overview §22 says exact routes must come from actual code. They are labelled `PROPOSED`.

### FR-1 Crop & Mode Selection
- **Description:** Select crop ∈ {lettuce, spinach, tomato}; mode ∈ {image, sensors, both}.
- **Inputs:** `crop`, `mode`.
- **Outputs:** Form configured with the crop's input contract (image type, sensor schema).
- **User actions:** Select crop; select mode.
- **Validation:** Unknown crop/mode → `422` structured error. Mode `both` allows partial success (one modality failing doesn't block the other).
- **Edge cases:** Switching crop mid-form clears crop-specific fields; mode `sensors` for tomato shows experimental banner (FR-3).

### FR-2 Sensor Classification (per crop)
- **Description:** Validate and classify sensor readings using the crop's exact ordered schema.
- **Inputs (ordered schemas — Confirmed in overview):**
  - Lettuce: `ph, ec, water_temp, humidity, air_temp`
  - Spinach: `air_temperature_C, relative_humidity_percent, water_temperature_C, ph, ec_mS_cm`
  - Tomato: `air_temperature_C, relative_humidity_percent, water_temperature_C, ph, ec_mS_cm, nitrogen_mg_L, phosphorus_mg_L, potassium_mg_L` (confirmed from `build_tomato_dataset.py`; labels are the most severe fault in `primary` mode)
- **Outputs:** predicted class, display label, confidence + defined semantics, model/rule version, timestamp, warnings, `experimental` flag.
- **User actions:** Enter values (with units) or receive adapter payload.
- **Validation:** Field names/order; numeric type; unit; plausible range (e.g. lettuce report blanked pH 2.0 and EC 26.7 as impossible); no unexpected fields; stale timestamp check for device data.
- **Edge cases:** Missing field → reject (or explicitly trained missing-data path, `⚠️ UNCLEAR` which; default = reject). Never zero-fill. Lettuce 7-class model excludes `WaterTemp_Low` and `EC_Low` (rare) — out-of-domain inputs trigger an "outside documented classes" note. Tomato: live device values not assumed compatible (Overview §15.7).
- **Classes:** see overview §11.5, §13.5, §15.5 — class names preserved verbatim in metadata.

### FR-3 Tomato Sensor Experimental Mode
- **Description:** Tomato sensor inference is flagged *experimental/manual-assessment* until a documented, validated mapping from real device readings to the transformed training features exists.
- **Outputs:** Prominent limitation banner; `experimental=true`.
- **Edge cases:** If mapping is later validated, flag flips via registry metadata, not code edit.

### FR-4 Image Classification
- **Description:** Classify a plant image per crop using the promoted model.
- **Inputs:** image file (JPEG/PNG; size limit — `⚠️ UNCLEAR`, default 10 MB), `crop`, `input_mode` (`leaf` = the photo is one leaf, `plant` = whole-plant photo to split into leaves; always explicit), optional `model_family`.
- **Outputs:** class (lettuce 4, spinach 6, tomato 11 classes; folder names under `image_data/` are the source labels), display label (`healthy` / `unhealthy/<original class>`), confidence, uncertainty flag, model version, preprocessing metadata.
- **Validation:** MIME + magic-byte check, decodability, size; filename never used as path.
- **Edge cases:** Missing `input_mode` → reject. Plant mode with no valid leaf → `SEGMENTATION_FAILED` (retryable), not a prediction. Leaves that fail validation are listed as *not assessed* with reasons. Low confidence per leaf (threshold derived from validation data) → `uncertain`. Invalid/unsegmentable image → *input-processing failure*, **not** a disease prediction.
- **Input contract (D9):** every image model consumes one leaf. Lettuce leaf crops come from whole-plant images (weak plant-level labels); spinach and tomato use the provided leaf images.
- **Outputs (image):** `leaves[]` (leaf id, bbox, class, display label, confidence, uncertain), `rejected_leaves[]`, and `summary` (leaf count, per-class counts; no averaged confidence).

### FR-5 Segmentation (SAM 2.1)
- **Description:** Split a plant photo into individual leaves and produce a validated mask per leaf; also used to build lettuce training crops and to mask leaf images for feature extraction.
- **Outputs:** mask (or reference), mask-quality verdict, segmentation config/version.
- **Validation:** mask exists, non-empty, plausible area/geometry, not background.
- **Edge cases:** Failed/poor masks → warning + **retry** path; never silent fallthrough. Overlapping duplicate masks are merged or removed by a documented rule; the number of leaves is capped. Leaf count and rejected leaves are always reported. GPU OOM → smaller batch/CPU fallback or clear error.

### FR-6 OpenCV Feature Extraction + Classical ML
- **Description:** Extract ordered, versioned features and classify with RF/SVM etc.
- **Feature contracts (Confirmed in overview §16):**
  - Lettuce (7): `exg_mean, exr_mean, yellow_pixel_ratio, brown_pixel_ratio, mean_hue, mean_saturation, contour_compactness`
  - Spinach (6): `exg_mean, exr_mean, yellow_pixel_ratio, brown_pixel_ratio, relative_leaf_area, glcm_contrast`
  - Tomato (5 candidate, **conditional**): `droop_angle, glcm_contrast, glcm_entropy, bounding_aspect_ratio, solidity`
- **Validation:** NaN/Inf check; count/name/order vs artifact metadata; BGR→RGB handling; safe zero-perimeter handling for compactness.
- **Edge cases:** `droop_angle` retained only if a defensible, reproducible calculation exists; otherwise removed/replaced, and the **entire** tomato feature set is regenerated and dependent models retrained (Overview §16.3). A failed feature is an error, never `0`.

### FR-7 Evidence Presentation (No Fusion)
- **Description:** Render image result and sensor result in separate panels.
- **Rules:** No combined severity score. Mismatch alerts only with a validated mapping (none exists by default → never raise "contradiction").
- **Edge cases:** One modality unavailable → show the other and state what was not assessed.

### FR-8 Recommendations
- **Description:** Generate conservative advice with cited evidence.
- **Inputs:** validated predictions + confidence + warnings.
- **Outputs:** structured recommendation: `summary`, `evidence[]`, `suggested_checks[]`, `uncertainty_note`, `source` ∈ {`gemini`,`rules`}, `rules_version`.
- **Validation:** Gemini output parsed against a schema; treated as untrusted; no dosage/concentration instructions; failure/timeout/invalid → rule-based fallback.
- **Edge cases:** Low confidence → recommend re-measure/inspect, not corrective action. Insufficient info → explicit "cannot recommend confidently".

### FR-9 History
- **Description:** Persist and list records (PostgreSQL).
- **Stored per record:** crop, modality, observed timestamp+timezone, provenance (`live|manual|imported`), inputs (or safe reference), prediction, confidence, model version, preprocessing/rule version, recommendation + evidence.
- **Validation:** DB failure → do **not** claim saved.
- **Edge cases:** Old records show their real timestamp; never "live"/"current" unless freshly acquired and validated. Pagination on list.

### FR-10 Campus/Device Adapter
- **Description:** Adapter to the campus server (reachable at a configured IP) that fetches the latest sensor readings and camera frames, mapping them to each crop's model schema. Manual entry always remains available.
- **Edge cases:** No device → manual entry; stale/disconnected/malformed → *unavailable*; distinguish `missing | disconnected | invalid | stale`. Hardware action never assumed executed.
- Camera frames fetched from the server are treated as `live` image inputs with a capture timestamp; they enter the normal image pipeline (validation, segmentation, classification). Warn that camera images may differ from the training-image distribution.
- `⚠️ UNCLEAR:` protocol, endpoints, auth, payload format, field names/units, update rate, and camera format (snapshot / MJPEG / RTSP) are unknown until the server is probed. Development proceeds against a **mock adapter**; the real adapter is built from the probe results.

### FR-11 ML Lab (Training, Evaluation, Registry)
- **Description:** Scripts to train, evaluate, and register models.
- **Outputs:** artifact + metadata (§24.1 list), evaluation report (§18.2 list), manifests.
- **Validation:** Feature schema compatibility checked at startup and at inference; missing/corrupt artifact → clear service error, never silently load another model.
- **Edge cases:** Promotion is explicit (not "newest file wins"); previous known-good retained for rollback.

---

## Learning Requirements (LR)

| ID | Requirement |
|---|---|
| LR-1 | Every ML concept (see list below) has a **concept note** in `docs/learning/` written by the *navigator* (not the driver) of the related implementation tasks. |
| LR-2 | Every concept has a **teach-back**: the other member explains it from scratch; both must answer the concept's **self-check questions** before the phase is closed. |
| LR-3 | Driver/navigator roles **alternate** by task; the todo balances drive counts across both members. |
| LR-4 | Each member implements from scratch (NumPy/pure-Python **or** hand-derivation) at least the "core intuition" piece for: decision-tree splitting (Gini), a confusion matrix + macro-F1, and a convolution/feature-map demo, *before* using library versions. `⚠️ UNCLEAR:` confirm whether course rules allow/expect this. |
| LR-5 | Both members must be able to explain *why* the headline sensor-model scores (reported 0.994 test accuracy for tree models) do **not** imply real-world biological accuracy. |

**Concepts covered:** train/val/test and leakage; stratification & grouped/time-aware CV; class imbalance & oversampling; decision trees & Gini/entropy; random forests & bagging; SVM, margins & kernels; feature scaling; KNN & logistic regression baselines; macro-F1 / precision / recall / confusion matrix; hyperparameter search (GridSearchCV); CNN fundamentals (convolution, pooling, backprop); transfer learning & fine-tuning (EfficientNetB0); data augmentation; colour spaces (BGR/RGB/HSV), vegetation indices (ExG/ExR); texture (GLCM); shape features (compactness, solidity, aspect ratio); segmentation with foundation models (SAM 2.1); softmax, calibration, and uncertainty; model registry & reproducibility.

---

## Non-Functional Requirements

### Performance *(planned targets; measure before claiming)*
- Sensor inference p95 ≤ 300 ms on laptop CPU.
- Image inference: target ≤ 30 s CPU fallback incl. segmentation; materially faster on RTX 3050 when runtime supports it. Report device actually used.
- History list endpoint paginated (default 20, max 100).

### Scalability
- Expected load: single-digit concurrent users (campus demo). Design: stateless FastAPI workers; local-file model store behind an interface so cloud storage can be swapped later (not assumed now).

### Security
- No authentication in v1 (explicit scope decision). Therefore: strict server-side validation, upload limits, safe file handling, rate limiting, CORS allow-list, secrets only via env, Gemini output treated as untrusted, model loading restricted to configured paths. If auth is added later, a new threat model is required (Overview §23).

### Accessibility
- Target WCAG 2.1 AA: contrast ≥ 4.5:1 body text, keyboard-operable flows, labelled inputs, non-colour-only status (uncertainty and errors must also use text/icons).

### Reliability
- No formal uptime SLA (course project). Required: graceful degradation (Gemini → rules; GPU → CPU; device → manual; DB failure → honest error).

### Maintainability
- Typed Python (type hints + `mypy`/`pyright` — pick one at setup), `ruff` lint/format, ESLint+Prettier for frontend, conventional commits, docs updated with every contract change, reproducible seeds and manifests.

---

## Assumptions & Constraints

**Constraints (Confirmed):** Next.js + React · FastAPI · PostgreSQL · PyTorch · scikit-learn · SAM 2.1 · Gemini API · local file storage · no auth · HP Victus 16 / RTX 3050 with CPU fallback · `.env` + `.env.example`.

**Assumptions (Planned default):**
- A1 Team = 2 people (Yashwant, Sushil). `⚠️ UNCLEAR:` deadline/end date still not given.
- A2 "MLM" = **Machine Learning Methods** (confirmed).
- A3 The repository currently holds datasets only (`image_data/`, `sensor_data/`); no code exists.
- A4 All datasets are accessible locally; licenses are verified before any redistribution.
- A5 Python/Node versions are chosen and pinned at setup after verification (not fixed here).
- A6 SAM 2.1 checkpoint size is chosen based on measured VRAM/CPU capacity (RTX 3050 has limited VRAM).

**Open questions:** see *Still open* under Resolved Decisions. Additional: are there course-mandated algorithms or from-scratch implementations (LR-4)?

---

## MVP Scope

### Must Have
- Dataset audit + manifests + split manifests (all crops), including the newly created validation sets (D1).
- Campus server adapter, Live Monitor screen, and camera-frame analysis, with mock adapter and manual fallback.
- Earthy, animated, accessible React UI per `DESIGN.md`.
- Sensor classifiers for lettuce, spinach, tomato (DT, RF, SVM minimum) with leakage-safe evaluation and registry entries.
- OpenCV feature extractors (lettuce 7, spinach 6, tomato per `droop_angle` decision) + RF/SVM.
- Custom CNN + EfficientNetB0 (PyTorch) for image classification for **all three crops** (D4).
- SAM 2.1 segmentation with mask validation + retry.
- FastAPI endpoints: predict (sensor, image), history, health, model info.
- Next.js UI: crop/mode selection, input, separate result panels, uncertainty display, limitations.
- Deterministic rule-based recommendations.
- PostgreSQL history with timestamps/provenance.
- Layered tests (unit, integration, data/leakage).
- Equal-learning artifacts (concept notes + teach-backs).

### Should Have
- Gemini recommendation layer with schema validation + fallback.
- Probability calibration experiments (validation-fit).
- Grouped CV for lettuce by calendar date.
- Model registry CLI for promote/rollback.
- CI pipeline (lint, test).

### Nice To Have
- Grad-CAM-style explanations for CNN.
- Experiment tracking (e.g., MLflow, `⚠️ UNCLEAR` whether desired).
- Dark mode, i18n (Tamil) UI.
- Export history to CSV.

---

## Future Enhancements
- Paired image+sensor data collection protocol → validated fusion (Overview §19.9).
- Real campus sensor integration with validated tomato feature mapping.
- Cloud object storage and authentication (new threat model).
- Per-cultivar/lighting domain adaptation and out-of-distribution detection.
- Mobile-first PWA.
