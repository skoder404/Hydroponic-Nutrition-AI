# ROADMAP — Hydroponic Nutrition AI

> **Status:** Draft v0.1 · **Date:** 2026-10-20 · **Owners:** Yashwant, Sushil Kumar M A
> Derived from `PRD.md`, `DESIGN.md`, `TECH_RULES.md`, `PROJECT_OVERVIEW.md`. Task-level detail lives in [`todo.md`](./todo.md); this file groups it into milestones with acceptance criteria and risks.
> **No calendar dates are given** because the course deadline has not been shared. Effort figures come from the planning estimates in `todo.md` (not measurements).

---

## Overview

Delivery is split into five milestones. M0-M3 form the **MVP** (161 tasks, about 153 estimated hours of task effort across both teammates); M4 adds extensions (7 tasks, about 7 h). Models come before the application on purpose: the API serves *promoted* models, so M1 gates M2.

**Two parallel streams are possible after M0** (both members still attend every concept session): (a) ML work for M1, and (b) frontend foundations for M2 (tokens, motion system, shell), plus the campus server probe, which needs the campus network.

**Order of crops inside M1:** the campus system grows **tomato only**, so tomato is the live crop. Tomato sensor work (`tomato_run`) has no dependency on the other crops and can go first. The image pipeline is built once on lettuce and reused for spinach and tomato in the current task order; swap the order if you prefer to build it on tomato first.

| Milestone | Todo phases | Tasks | Est. effort (h) |
|---|---|---|---|
| M0 Foundation & Data Readiness | 1, 2, 3 | 47 | 37 |
| M1 MVP Core: Models (sensor + image) | 4, 5 | 60 | 66 |
| M2 MVP Core: Application (API + UI + live data) | 6, 7 | 37 | 34 |
| M3 MVP Polish & Release | 8, 9 | 17 | 16 |
| M4 Growth Features | 10 | 7 | 7 |
| **Total** | 1-10 | **168** | **160** |

*Example conversion (illustrative, not a commitment):* if each of you spends 8 hours a week on the project, 160 h of task effort is roughly 10 weeks; at 12 hours each it is roughly 7 weeks. Review time and rework are not included.

---

## Milestone Structure

### Milestone 0: Foundation & Data Readiness
- **Todo phases:** 1 (Setup), 2 (Data layer), 3 (Security baseline)
- **Objective:** Turn a datasets-only folder into a reproducible project: audited data, preserved test sets, **new validation sets**, split manifests, label registry, database schema, tooling, CI, and the learning scaffold. De-risk the two unknowns early: SAM 2.1 on your hardware and the campus server interface.
- **Tasks / Features:**
  - Data inventory, repo structure, `.gitignore`, `.env.example`, pinned environment, device report (T1-001, T1-002, T1-003, T1-004, T1-005, T1-006).
  - Feasibility spikes: SAM 2.1 (T1-007), Gemini (T1-008), **campus server probe** (T1-016).
  - Linting, CI, contribution rules, learning scaffold (T1-012, T1-013, T1-014, T1-015).
  - Provenance sheets, sensor audits, **grouped validation sets** (T2-010, T2-011, T2-012), image audit/duplicates/splits, label registry (21 image classes), overview errata, split-hash guard, manifests, DB schema.
  - Security baseline without login (T3-001 to T3-008).
- **Dependencies:** Datasets present in `image_data/` and `sensor_data/`; access to the campus network for the probe; GPU drivers for the spike (CPU fallback acceptable).
- **Estimated Complexity:** Medium
- **Acceptance Criteria:**
  - [ ] Inventory shows what every sensor file is (raw / prepared / split-source); unknowns are listed, not guessed.
  - [ ] Test partitions are untouched (hash guard passes) and validation sets exist for all three sensor datasets; image splits are 70/15/15 with duplicates handled.
  - [ ] SAM 2.1 numbers are measured on your machines, and the campus server interface is documented (or each unknown is explicitly marked).
  - [ ] `ml/labels.py` matches the folder names on disk; `PROJECT_OVERVIEW.md` is updated deliberately.
  - [ ] CI runs lint and tests on every push to `main`; no secrets or IP addresses are committed.
  - [ ] Learning tracker lists every concept for both members.

### Milestone 1: MVP Core: Models
- **Todo phases:** 4 (Sensor models), 5 (Image pipeline)
- **Objective:** Train, compare, select, evaluate once, and promote models for **all three crops and all model families**, with honest reporting and uncertainty handling.
- **Features (Must Have):**
  - Sensors: Decision Tree, Random Forest, SVM, KNN, Logistic Regression per crop; selection by validation macro-F1; run-once test evaluation; tomato marked experimental.
  - Images (all leaf-level, per-leaf confidence): OpenCV features (lettuce 7, spinach 6, tomato per the `droop_angle` decision) + RF/SVM; custom CNN and EfficientNetB0 (PyTorch); SAM 2.1 leaf extraction with per-leaf mask validation; lettuce leaf crops generated from whole-plant images with source mapping and a manual label-noise review.
  - Evaluation tooling, registry with promote/rollback, uncertainty thresholds from validation data.
  - Concept notes, from-scratch notebooks, and teach-backs (T4-026, T5-034).
- **Dependencies:** M0 complete; T1-007 outcome decides segmentation approach.
- **Estimated Complexity:** High
- **Acceptance Criteria:**
  - [ ] A promoted sensor model exists for each crop and a promoted image model (or feature model) exists for each crop, all with complete registry metadata.
  - [ ] Every report states data source, split, metric and limitations; validation and test results are never mixed; the test log shows each model evaluated once.
  - [ ] Leakage tests (once written) show no source-image overlap, no group overlap, and no blacklisted features.
  - [ ] Leaf extraction rejects bad masks and reports rejected leaves; `droop_angle` decision is recorded and dependent features/models are consistent with it.
  - [ ] Both members have completed teach-backs and self-checks for every concept in M1.

### Milestone 2: MVP Core: Application (API, UI, live data)
- **Todo phases:** 6 (Backend), 7 (Frontend)
- **Objective:** A working end-to-end app: choose crop, give data (manual, upload, or campus live), get separate image and sensor results with uncertainty, a rule-based recommendation, and saved history, in the earthy, animated, clean UI.
- **Features (Must Have):**
  - FastAPI: `/predict/sensor`, `/predict/image`, `/analyze`, `/history`, `/models`, `/device/*`, error envelope, timeouts, OpenAPI snapshot.
  - Campus server adapter (built from the probe ADR), camera-frame analysis, mock adapter, manual fallback.
  - Rule-based recommendations (versioned, no dosage).
  - React/Next.js UI: S1-S8 per `DESIGN.md`, motion system with reduced-motion support, confidence tiers, honest states.
- **Dependencies:** M1 promoted models; T1-016 ADR for the real adapter (mock adapter lets work proceed).
- **Estimated Complexity:** High
- **Acceptance Criteria:**
  - [ ] A user can complete crop to input to result to save to history for each crop with the mock adapter; the same flow works against the real campus server when reachable.
  - [ ] Missing, stale, or invalid inputs produce structured errors; nothing is zero-filled; history never shows old data as live.
  - [ ] Image and sensor results are shown separately with no merged score; tomato sensor results are labelled experimental.
  - [ ] Segmentation failure shows a retry path instead of a prediction.
  - [ ] UI follows `DESIGN.md` tokens; all animations respect `prefers-reduced-motion`.

### Milestone 3: MVP Polish & Release
- **Todo phases:** 8 (Testing), 9 (Deployment)
- **Objective:** Prove it works and make it repeatable: layered tests, accessibility, containers, CI, backups, a runbook, and the `v0.1.0` tag.
- **Features (Should Have within the MVP):** unit/integration/contract/data/E2E tests, coverage report, accessibility pass, Docker compose, HTTPS proxy sample, monitoring counters, demo runbook.
- **Dependencies:** M2.
- **Estimated Complexity:** Medium
- **Acceptance Criteria:**
  - [ ] Tests that were run are reported honestly (including coverage and gaps); CI is green.
  - [ ] No critical automated accessibility violations; main flow works by keyboard.
  - [ ] A teammate can run the demo from the runbook alone, including pointing the app at the campus server.
  - [ ] README claims match verified features; `v0.1.0` tagged only after the release checklist passes.

### Milestone 4: Growth Features
- **Todo phase:** 10
- **Objective:** Extensions after the MVP is safe: Gemini recommendations with strict sanitising, calibration study, cross-crop comparison, CSV export, final report, and a closing quiz.
- **Features (Should / Nice To Have):** Gemini provider + sanitiser (rules stay the fallback), probability calibration, cross-crop report, history CSV export, final report and concept quiz.
- **Dependencies:** M3 release.
- **Estimated Complexity:** Medium
- **Acceptance Criteria:**
  - [ ] Gemini output never reaches the UI without schema validation and sanitising; fallback counters increment on failure.
  - [ ] Calibration is adopted only if validation evidence supports it.
  - [ ] Final report states source, split and metric for every number.
  - [ ] Both members score and review the final concept quiz; weak concepts have follow-up plans.

---

## Learning Checkpoints

| Checkpoint | When | Evidence |
|---|---|---|
| Concept notes + teach-back for sensor ML | End of M1 sensor part (T4-026) | `docs/learning/PROGRESS.md` ticks for both members |
| Concept notes + teach-back for image ML | End of M1 (T5-034) | Same |
| Final quiz and retrospective | End of M4 (T10-007) | Scores and follow-ups recorded |

Rule: driver/navigator roles alternate; the navigator writes each concept note. The balance table in `todo.md` is reviewed at each checkpoint.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Preparation scripts/READMEs disagree with the overview or the CSVs (thresholds, priorities, split rules) | Medium | High | Audit against the scripts; record every discrepancy; mark anything not stated as 'not stated' and ask; keep tomato sensor results experimental. |
| Campus server protocol unknown or unreachable off campus | High | Medium | Probe early; mock adapter keeps development moving; manual entry always works. |
| Tomato live readings lack N/P/K or differ from the proxy-transformed training data | High | High | Probe field list; live-compatible subset model; keep tomato experimental; state limitations in the UI. |
| Camera images differ from the training images (lighting, angle, background) | High | Medium | Warn in the UI; treat as live input with uncertainty; collect sample frames for later evaluation. |
| SAM 2.1 too heavy for the RTX 3050 or campus CPU | Medium | High | Measure in the spike; smaller checkpoint; any alternative is an explicit team decision, never silent. |
| Lettuce leaf crops inherit the whole-plant label (weak labels); leaf separation fails on touching leaves | High | Medium | Manual crop label review, per-leaf confidence with a limitation note, rejected leaves reported, validator tuned on the mask review. |
| Validation hold-out leaves rare classes missing from a partition | Medium | Medium | Report class coverage; ask before dropping or merging classes. |
| All 3 crops x all model families exceeds the time available | High | High | Lettuce first end to end, then spinach, then tomato; cut optional ablations before evaluation rigor; Phase 10 optional. |
| High scores on rule-derived labels read as real-world accuracy | Medium | High | Limitations in every report and UI; teach-back on why trees recover rules. |
| Duplicate or sibling-crop leakage across image splits | Medium | High | Duplicate report, whole-image splitting, crop inheritance, data tests. |
| `droop_angle` not defensible, forcing tomato feature regeneration | Medium | Medium | Decide early; schema versioning; mark dependent models incompatible until retrained. |
| Unequal learning between teammates | Medium | High | Strict alternation, navigator-written notes, teach-backs, balance review. |
| Dataset licence or redistribution terms unclear | Medium | Medium | Record 'unverified' honestly; never commit datasets; check before sharing artifacts. |
| Gemini unavailable, rate limited, or unsafe output | Medium | Medium | Rules are the default; schema validation; sanitiser; counters. |

---

## Open Questions (still to answer)

1. Which readings does the campus server provide for tomato (does it include nitrogen, phosphorus, potassium)? This decides whether the eight-feature tomato model can run live or the live-compatible variant is needed. (resolved by the probe task)
2. What is the campus server's protocol, authentication, payload format and camera format? (resolved by the probe task)
3. Which SAM 2.1 checkpoint and prompting approach separate lettuce leaves well enough? (the spike answers this)
4. Are any classes too rare to appear in train, validation and test after the grouped hold-out? (the validation-split tasks report this)
5. What is the course deadline? It determines how much of M4 and how many optional ablations are realistic.
6. Do the re-downloaded Kolar file and the tomato script reproduce the shipped 44k CSVs, and what are the real tomato row and class counts (overview: 44,141 rows; README estimate: about 33,850)? (the reproduction task answers this). The lettuce and spinach raw sources are not needed for this project.