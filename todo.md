# todo.md — Hydroponic Nutrition AI
> Generated from: PRD.md · DESIGN.md · TECH_RULES.md · PROJECT_OVERVIEW.md
> *(ROADMAP.md was not requested; milestones are derived from the PRD MVP scope and the phase gates below.)*
> Target: MVP-first execution by two learners (Yashwant, Sushil)
> Last Updated: 2026-10-20

**168 tasks · 10 phases.** Phases 1-9 deliver the MVP; Phase 10 holds Should Have extensions and starts only after the first release (T9-006).

---

## How to use this file

1. **Work in ID order within a phase**, and never start a task whose dependencies are unchecked. IDs follow `T<phase>-<number>`.
2. **Task format:** Description, Dependencies, Acceptance Criteria (checkboxes), Estimated Effort, Assigned To (role), plus two project-specific lines: **Driver / Navigator** and, for ML concepts, **Learning Focus**.
3. **Roles** extend the usual list with `ML` (data/model work) because this is an ML-first project.
4. **Tick a task only when its acceptance criteria are demonstrated** by code, tests, or committed artifacts (Overview §27.4 / §30.21). Update this file in the same commit that completes the task.
5. **`⚠️ UNCLEAR:`** marks a decision the source documents do not settle. Resolve these (see [Open questions](#open-questions-collected-from-tasks)) before starting that task, rather than guessing.
6. **Accuracy rules:** read `PROJECT_OVERVIEW.md` first; never invent formulas, thresholds, feature orders, split counts, endpoints, or test results; inspect the repo before claiming something exists.

### Pairing protocol (equal learning)

- **Driver** implements the task and commits it straight to `main` (`git pull --rebase` first, push right after). **Navigator** reviews the pushed commit(s) and, on every task with a *Learning Focus*, **writes the concept note** at `docs/learning/<concept>.md` using the template from T1-015 (explaining it is how the navigator learns it).
- Driver and navigator **alternate**: concept tasks and non-concept tasks are each alternated strictly in order, so each person drives about half of the work *and* about half of the concepts (see the balance table in the appendix).
- Parallel streams (e.g. frontend while the other person trains models) are allowed, **but both attend every concept session**.
- Each ML phase ends with a **teach-back** task. A phase is closed only when **both** members have ticked the phase gate in `docs/learning/PROGRESS.md`.
- The *from-scratch* notebooks (Gini split, metrics, convolution) are done **before** the library versions, by the person named as driver, then explained by the navigator.

### Phase overview

| Phase | Title | Tasks |
|---|---|---|
| 1 | [Project Setup & Tooling](#phase-1-project-setup--tooling) | 16 |
| 2 | [Database & Data Layer](#phase-2-database--data-layer) | 23 |
| 3 | [Security Baseline (no login in v1)](#phase-3-security-baseline-no-login-in-v1) | 8 |
| 4 | [ML Core: Sensor Models](#phase-4-ml-core-sensor-models) | 26 |
| 5 | [ML Core: Image Pipeline](#phase-5-ml-core-image-pipeline) | 34 |
| 6 | [Core Backend: APIs & Services](#phase-6-core-backend-apis--services) | 16 |
| 7 | [Core Frontend: UI & Interactions](#phase-7-core-frontend-ui--interactions) | 21 |
| 8 | [Testing](#phase-8-testing) | 11 |
| 9 | [Deployment & DevOps](#phase-9-deployment--devops) | 6 |
| 10 | [Should Have Features](#phase-10-should-have-features) | 7 |

> **Decisions applied (2026-10-20):** no validation sets existed, so Phase 2 creates them as grouped hold-outs from the split-source CSVs; the repo is greenfield (datasets only); all three crops are trained with all model families; campus server integration is part of the MVP (probe first, mock adapter meanwhile); the UI follows the earthy/animated direction in DESIGN.md.

> **Structure note (deviation from the default task-generator layout):** the default phases are web-app oriented (Database, Auth, Backend, Frontend). For this project the authentication phase is replaced by a *Security Baseline* because login is out of scope in v1, and two ML phases (4 and 5) are inserted before the backend because the API depends on promoted models.

---

## Phase 1: Project Setup & Tooling
*Goal: Reproducible dev environment, audited repository, folder structure, CI skeleton, and the learning scaffold.*

#### T1-001: Inventory the data and scripts on disk (read-only)
**Description:** The repository currently holds `image_data/` (Lettuce_Images, Spinach_Images, Tomato_Images) and `sensor_data/`, which per crop contains a preparation script (`build_lettuce_dataset` and `README_lettuce_dataset`, both without file extensions; `generate_7class_dataset.py` and `README_7class_dataset.md`; `build_tomato_dataset.py` and `README_tomato_dataset.md`) and `*_full.csv`, `*_train.csv`, `*_test.csv` files. Write `docs/decisions/0001-data-inventory.md`: for every folder the file counts, formats and sizes; for every sensor file its columns and row count; and which script produced which CSV. The two extension-less files are a Python script and a Markdown README (confirmed from their contents); decide in a deliberate rename commit whether to add `.py`/`.md`, and note that the lettuce README refers to `build_dataset.py` and `train_model.py`, which are not in the repo. Do not modify, move, or rename anything. No application or ML code exists yet.
**Dependencies:** None
**Acceptance Criteria:**

- [x] Every data folder has counts, formats and sizes recorded
- [x] Every file in `sensor_data/` is listed with columns and row count and mapped to the script that produced it
- [x] Nothing was modified, moved, or renamed

**Estimated Effort:** 45 min
**Assigned To:** Full-stack
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T1-002: Create or adapt the monorepo folder structure
**Description:** Using the audit, create only the folders that are missing from the TECH_RULES layout (`frontend/`, `backend/app/{api,schemas,services,database,core}`, `ml/{image_models,sensor_models,segmentation,features,fusion,recommendations,evaluation,registry}`, `processed_data/{splits,features,sensor,manifests}`, `models/{image,sensor,manifests}`, `reports/`, `notebooks/`, `scripts/`, `tests/{unit,integration,contracts,data}`, `docs/{learning,datasets,decisions}`) with `.gitkeep`. Leave `image_data/` and `sensor_data/` exactly where they are (they are the raw, read-only data). Never move or duplicate an existing equivalent folder; log deviations in the inventory doc. Add `ml/fusion/README.md` stating that no fusion model may be trained on unpaired image/sensor data.
**Dependencies:** T1-001
**Acceptance Criteria:**

- [x] No duplicate or parallel folders were created
- [x] Tree matches TECH_RULES or each deviation is logged in the audit doc
- [x] `ml/fusion/README.md` exists with the no-fusion rule (Overview §19)

**Estimated Effort:** 30 min
**Assigned To:** Full-stack
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T1-003: Add .gitignore and .gitattributes
**Description:** Ignore `.env` (while keeping `.env.example` tracked), temporary uploads (`uploads/`), masks (`masks/`), outputs (`outputs/`), virtualenvs (`.venv/`, `venv/`), `node_modules/`, `.next/`, caches, and notebook checkpoints. Keep `image_data/`, `sensor_data/`, `models/`, `processed_data/`, manifests, and reports tracked in the repository. `.gitattributes` normalises line endings and strips notebook outputs on commit (choose a tool, e.g. nbstripout).
**Dependencies:** T1-002
**Acceptance Criteria:**

- [x] `.env` is ignored while `.env.example` is tracked
- [x] `image_data/` and `sensor_data/` remain tracked in the repository
- [x] `models/` and `processed_data/` directories remain trackable
- [x] `.gitattributes` normalises line endings and configures notebook output stripping

**Estimated Effort:** 30 min
**Assigned To:** DevOps
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T1-004: Add .env.example and typed settings loader
**Description:** Create `.env.example` with placeholders only for `DATABASE_URL`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `MODEL_REGISTRY_PATH`, `IMAGE_DATA_PATH`, `SENSOR_DATA_PATH`, `PROCESSED_DATA_PATH`, `CAMPUS_SERVER_URL`, `CAMPUS_CAMERA_URL`, `CAMPUS_POLL_INTERVAL_S`, `CAMPUS_MAX_AGE_S`, `CAMPUS_REQUEST_TIMEOUT_S`, `ALLOWED_ORIGINS`, `MAX_UPLOAD_MB`, `DEVICE_ADAPTER`, `SAM2_CHECKPOINT`, `SAM2_CONFIG`, `FORCE_CPU`. Implement `backend/app/core/config.py` (pydantic-settings style) that loads them, redacts secrets in `repr`, and treats `GEMINI_API_KEY` as optional because the rule fallback must work without it.
**Dependencies:** T1-002, T1-003
**Acceptance Criteria:**

- [x] `.env.example` contains placeholders only
- [x] App settings load with no `GEMINI_API_KEY` set
- [x] Printing the settings object never reveals a secret value

**Estimated Effort:** 30 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T1-005: Verify and pin Python / ML dependencies
**Description:** Choose the Python version and package tool (`uv` or `pip-tools`) only after verifying that a CUDA-enabled PyTorch/torchvision build works on the RTX 3050 and that a CPU-only build installs on a machine without a GPU. Create separate pinned requirement files (backend, ml, dev) and record every pinned version (PyTorch, torchvision, scikit-learn, scikit-image, OpenCV, FastAPI, Pydantic, SQLAlchemy, Alembic) in `docs/decisions/0002-runtime-versions.md`. ⚠️ UNCLEAR: Overview §25 lists both Windows 11 and Ubuntu; confirm which OS each teammate uses.
**Dependencies:** T1-002
**Acceptance Criteria:**

- [x] A fresh virtualenv installs from the pinned files on both teammates' machines
- [x] CPU-only install imports torch successfully
- [x] ADR lists each pinned version with a one-line reason

**Estimated Effort:** 60 min
**Assigned To:** DevOps
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T1-006: Write scripts/check_env.py device and library report
**Description:** Print OS, Python, and versions of torch, torchvision, scikit-learn, scikit-image and cv2; `torch.cuda.is_available()`; device name; total/free VRAM; and a tiny tensor-on-device test. Honour `FORCE_CPU`. Exit non-zero on any import failure. Save each teammate's output as an appendix in ADR 0002.
**Dependencies:** T1-005
**Acceptance Criteria:**

- [x] Output states the device actually used and never claims GPU when the tensor ran on CPU
- [x] `FORCE_CPU=1` forces the CPU path
- [x] Both teammates' outputs are saved in the ADR

**Estimated Effort:** 30 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T1-007: SAM 2.1 feasibility spike (install, load, time, memory)
**Description:** Follow the SAM 2.1 installation instructions, load one checkpoint, segment 3 sample images per crop, and try to separate individual leaves in 3 whole-plant lettuce images with automatic masks. Record checkpoint name/size, load time, peak VRAM, how well individual leaves are separated, and per-image latency on GPU and on CPU in `docs/decisions/0003-sam2-feasibility.md`. ⚠️ UNCLEAR: the overview does not say whether to use automatic masks or point/box prompts for leaf separation; test both and report. Do not guess numbers; measure them.
**Dependencies:** T1-005, T1-006
**Acceptance Criteria:**

- [x] ADR states the exact checkpoint/config used plus measured (not estimated) numbers
- [x] ADR states whether CPU fallback is viable
- [x] If SAM 2.1 cannot run, the options are listed and the team decides before Phase 5 (no silent substitution)

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit and writes the concept note.
**Learning Focus:** Foundation-model segmentation: prompts, masks, and failure modes (SAM 2.1) → `docs/learning/foundationmodel-segmentation-prompts-masks-and-failure.md`

#### T1-008: Gemini API smoke test (key stays local)
**Description:** Write `scripts/gemini_smoke_test.py` that reads `GEMINI_API_KEY` and `GEMINI_MODEL` from the environment, sends one harmless prompt, and prints OK/FAIL with latency. Record the working model name and SDK in `docs/decisions/0004-gemini-config.md`. If no key is available the script must exit cleanly with a clear message, which proves the fallback path is the default.
**Dependencies:** T1-004, T1-005
**Acceptance Criteria:**

- [x] Script never prints or logs the key
- [x] Missing key gives a clean 'not configured' message and exit code 0
- [x] ADR records the model name from a real call, or states it could not be verified

**Estimated Effort:** 30 min
**Assigned To:** Backend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T1-009: Scaffold FastAPI app factory, /health, request-ID middleware, JSON logging
**Description:** Create `backend/app/main.py` with `create_app()`, router mounting under `/api/v1`, and `GET /api/v1/health` returning `status`, `device` (from a stub DeviceManager), and `db`/`registry` as 'unknown' until those exist. Add middleware that sets `request_id` and structured JSON logging with fields `event`, `request_id`, `level`.
**Dependencies:** T1-004, T1-005
**Acceptance Criteria:**

- [x] `GET /api/v1/health` returns 200 with documented fields
- [x] Every log line is JSON containing `request_id`
- [x] No secret appears in logs

**Estimated Effort:** 45 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T1-010: Scaffold Next.js (App Router, TypeScript) and Tailwind
**Description:** Create `frontend/` with routes `/`, `/analyze`, `/history`, `/history/[id]`, `/models`, `/about` as placeholder pages, Tailwind configured, and ESLint/Prettier configs. Add a `NEXT_PUBLIC_API_BASE_URL` variable to `.env.example`.
**Dependencies:** T1-002, T1-003
**Acceptance Criteria:**

- [x] `npm run build` succeeds
- [x] All six routes render placeholders
- [x] ESLint passes

**Estimated Effort:** 45 min
**Assigned To:** Frontend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T1-011: Add docker-compose service for PostgreSQL
**Description:** Create `docker-compose.yml` with a `db` service (named volume, healthcheck, credentials from env). Ensure `DATABASE_URL` in `.env.example` matches.
**Dependencies:** T1-004
**Acceptance Criteria:**

- [x] `docker compose up db` reports healthy
- [x] Data persists across restart
- [x] No password is committed

**Estimated Effort:** 30 min
**Assigned To:** DevOps
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T1-012: Configure linters, type-checkers, and pre-commit with secret scan
**Description:** Configure ruff, one Python type-checker (mypy or pyright; record the choice), ESLint, Prettier, and `pre-commit` including a secret scanner such as gitleaks. Expose one-command entry points (Makefile or `package.json` scripts).
**Dependencies:** T1-009, T1-010
**Acceptance Criteria:**

- [x] Pre-commit blocks a commit containing a fake API key
- [x] All linters pass on the scaffold
- [x] Commands are listed in the Makefile/scripts

**Estimated Effort:** 45 min
**Assigned To:** DevOps
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T1-013: Add GitHub Actions CI skeleton
**Description:** Workflow on every push to `main` (and on pull requests if ever used): backend (ruff, type-check, pytest), frontend (lint, typecheck, build), and secret scan, with dependency caching. Heavy training and SAM runs are excluded from CI (TECH_RULES).
**Dependencies:** T1-012
**Acceptance Criteria:**

- [x] Workflow passes on a clean push to `main`
- [x] Workflow fails on an intentional lint error
- [x] No secrets are needed to run it

**Estimated Effort:** 45 min
**Assigned To:** DevOps
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T1-014: Write CONTRIBUTING.md, commit checklist, and issue templates
**Description:** Document the commit-to-`main` workflow (team decision: no feature branches; `git pull --rebase` before each task and before each push; one task per commit; never force-push; a red `main` is fixed before the next task), Conventional Commits with the task ID in brackets, the pairing protocol (driver/navigator, concept note, teach-back), and a definition of done ('implementation and evidence exist'). `.gitmessage` (a commit template, enabled with `git config commit.template`) and a review checklist must cover: what/why, tests run, were splits touched?, metrics with split and metric named, limitations, link to learning note.
**Dependencies:** T1-013
**Acceptance Criteria:**

- [x] Commit template works with `git config commit.template`
- [x] CONTRIBUTING states that the navigator reviews each task's commit(s) after they are pushed
- [x] Definition of done matches Overview §27.4

**Estimated Effort:** 45 min
**Assigned To:** DevOps
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T1-015: Create the learning scaffold and progress tracker
**Description:** Add `docs/learning/_TEMPLATE_concept.md` (intuition, math, from-scratch snippet, library call, failure modes, where it appears in this project, 3-5 self-check questions), `_TEMPLATE_teachback.md`, and `docs/learning/PROGRESS.md` listing every concept from the PRD with one column per teammate (concept note written, teach-back done, self-check passed, date).
**Dependencies:** T1-002
**Acceptance Criteria:**

- [x] Every PRD concept is listed in PROGRESS.md
- [x] Templates render correctly on GitHub
- [x] PROGRESS.md has a column set for each teammate

**Estimated Effort:** 45 min
**Assigned To:** Full-stack
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T1-016: Probe the campus server (sensors + camera) and document the interface
**Description:** From a machine on the campus network (or VPN), use the IP provided by the campus to discover how to get sensor readings and the camera feed: protocol (HTTP, MQTT, WebSocket, RTSP, other), endpoints/topics, authentication, request and response formats, field names and units, timestamp format and timezone, update frequency, and camera format (snapshot URL, MJPEG, or RTSP). Save 2-3 redacted sample payloads and one sample frame under `docs/decisions/0008-campus-server/` and write `docs/decisions/0008-campus-server.md`. Keep the IP and any credentials in `.env` only. ⚠️ UNCLEAR: protocol, credentials, and whether the server is reachable outside the campus network; ask the campus contact when something cannot be determined, and never guess endpoints.
**Dependencies:** T1-004
**Acceptance Criteria:**

- [x] ADR lists protocol, endpoints, auth, fields with units, update rate, and camera format, with every unknown marked explicitly
- [x] No IP address or credential appears in any committed file
- [x] Sample payloads are redacted

**Estimated Effort:** 90 min
**Assigned To:** Backend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

---

## Phase 2: Database & Data Layer
*Goal: Verified datasets, provenance sheets, preserved split manifests, label/feature schemas, and the PostgreSQL schema.*

**Phase gate:** Both members tick the Phase 2 concept check in `docs/learning/PROGRESS.md`.

#### T2-001: Create the dataset provenance template
**Description:** Add `docs/datasets/_TEMPLATE.md` containing every Overview §9.1 field (source, URL, version/date, filenames, licence/redistribution, counts, class counts, preprocessing script/version, split protocol, synthetic/derived values, limitations) plus a per-column table tagged measured / adjusted / derived / synthetic / rule-generated.
**Dependencies:** T1-002
**Acceptance Criteria:**

- [x] All §9.1 fields present
- [x] Licence/redistribution field present
- [x] Column-type table present

**Estimated Effort:** 15 min
**Assigned To:** Full-stack
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T2-002: Fill provenance sheets for lettuce image and sensor data
**Description:** Complete two sheets: Kaggle 'Lettuce NPK Dataset' and HydroGrowNet (DOI 10.17632/g6cm3v3wdp.5). Use real counts from disk. ⚠️ UNCLEAR: dataset version/date and licence terms are not recorded in the overview; look them up and record 'unverified' where they cannot be confirmed.
**Dependencies:** T2-001, T1-001
**Acceptance Criteria:**

- [x] Both sheets saved under `docs/datasets/`
- [x] Counts come from files on disk, not from the overview
- [x] Unknown licence/version fields say 'unverified' rather than being guessed

**Estimated Effort:** 60 min
**Assigned To:** Full-stack
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T2-003: Fill provenance sheets for spinach image and sensor data
**Description:** Complete sheets for the two Mendeley image sources (ymtzd8cf6x v1, skf6w2s2h2 v3) and the spinach sensor data derived from the Meng et al. (2025) USDA file. ⚠️ UNCLEAR: the canonical URL of the USDA Ag Data Commons file is not recorded (Overview §13.1); find and record it. Mark each sensor column measured / adjusted / derived / synthetic per Overview §13.4.
**Dependencies:** T2-001, T1-001
**Acceptance Criteria:**

- [x] Canonical source URL recorded or marked 'not found'
- [x] Column-type table matches Overview §13.4
- [x] `tds_ppm` is marked derived/excluded

**Estimated Effort:** 60 min
**Assigned To:** Full-stack
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T2-004: Fill provenance sheets for tomato image and sensor data
**Description:** Complete sheets for the Mendeley (zfv4jj7855 v1) and Kaggle tomato disease image sources, and for the Kolar sensor dataset (200,000 rows, 29 columns). Record the tomato README's own statement that the Kolar file looks simulated (uniform, independent variables, random timestamps, no daily or seasonal pattern) and that the soil-to-hydroponic mappings are proxy transforms; attribute that observation to the README.
**Dependencies:** T2-001, T1-001
**Acceptance Criteria:**

- [x] Sheets saved
- [x] Wording about simulation is stated as an observation, not a fact
- [x] Units marked 'unverified' where applicable

**Estimated Effort:** 60 min
**Assigned To:** Full-stack
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T2-005: Audit the lettuce sensor CSVs against the script and README
**Description:** Write `scripts/audit_sensor_split.py` (reused for the other crops) and check `hydroponic_lettuce_{full,train,test}.csv` against facts read from `build_lettuce_dataset` and its README: `full` = 726 real labelled readings (HydroGrowNet EXP.1-3 handheld-meter rows joined with hourly air data); labels come from action bands (pH 5.5-7.0, EC 1.2-2.5, air temperature 18-30, humidity 50-80, water temperature 18-26; one variable out gives that variable's class, two or more give `Multiple_Stress`); classes with fewer than 8 rows (`WaterTemp_Low` 8, `EC_Low` 1) are dropped, leaving 7 classes; test = every 4th calendar day of each experiment (162 real rows, never resampled); train = the remaining days oversampled with replacement to 240 per class (1,680 rows from 558 real rows, seed 42). Confirm each number and record these known discrepancies in the audit: the script's docstring says test is EXP.3, but the code and README use every 4th day; the README names `build_dataset.py` and `train_model.py` while the repo has `build_lettuce_dataset` and no `train_model.py`; the CSVs carry many columns that are not model inputs (trend features, daily aggregates, `tds`, `co2`, `lux_meter`, `label_next`). The three `EXP_*.xlsx` workbooks are not in the repo, so the script cannot be re-run unless they are downloaded; that is not needed for this project. Save `reports/data/lettuce_sensor_audit.md`; never overwrite files.
**Dependencies:** T1-001, T2-002
**Acceptance Criteria:**

- [x] Every count above is confirmed or listed as a discrepancy
- [x] The test rows' (experiment, date) pairs follow the every-4th-day rule and never appear in train
- [x] The number of duplicated rows created by oversampling is quantified

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit and writes the concept note.
**Learning Focus:** Train/test splits, oversampling, and day-based (grouped) separation → `docs/learning/traintest-splits-oversampling-and-daybased-grouped.md`

#### T2-006: Audit the spinach sensor CSVs against the script and README
**Description:** Check `spinach_7class_{full,train,test}.csv` against facts read from `generate_7class_dataset.py` and its README: 1,400 rows, 7 classes, 200 per class drawn (160 train / 40 test); test chosen as whole six-hour blocks (`bench|date|hour//6`, about 20 percent of rows) so neighbouring readings never straddle the split; no oversampling (balance comes from drawing); air temperature and humidity are real, except the 200 `Humidity_High_Mildew_Risk` rows whose humidity was shifted (`humidity_source=adjusted`); water temperature, pH, EC and TDS are synthetic; rules are water > 21 C, EC > 2.3 or < 1.8, pH < 6.0 or > 7.0, humidity > 75 or < 45, and every non-Normal row breaks exactly one rule, so there is no priority order to recover; `tds_ppm` is EC x 700 plus noise (derived, excluded from features). Verify the counts and that the block key built from `bench` and `source_timestamp_EDT` never spans train and test. The real source file (`Meng_..._greenhouse_temperature_humidity.csv`) is not in the repo, so the script cannot be re-run.
**Dependencies:** T1-001, T2-003, T2-005
**Acceptance Criteria:**

- [x] Counts per class per split match the README or discrepancies are listed
- [x] No six-hour block appears in both train and test
- [x] Real vs adjusted vs synthetic columns are recorded and `tds_ppm` is marked derived

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T2-007: Audit the tomato sensor CSVs against the script and README
**Description:** Check `tomato_hydroponic_{full,train,test}.csv` against facts read from `build_tomato_dataset.py` and its README: eight features in order `air_temperature_C, relative_humidity_percent, water_temperature_C, ph, ec_mS_cm, nitrogen_mg_L, phosphorus_mg_L, potassium_mg_L`; proxy mappings (humidity + 15, pH - 1.0, EC x 1.4, N x 4, P x 3, K x 6); water temperature synthetic = 8 + 0.55 x air + 0.003 x (solar - 700) + Normal(0, 0.6), seed 42, no clipping; thresholds (water > 25 C, EC < 2.0 or > 4.0, pH < 6.0 or > 6.5, humidity < 50 or > 85, N < 90, P < 35, K < 144 mg/L); default `primary` mode labels a row with its most severe fault (deviation past the threshold divided by the largest such deviation in the whole mapped file), giving 10 classes (`--mode single` would add `Multiple_Faults`); the split puts random whole calendar months in test (about 20 percent of rows) and has no validation partition; classes are balanced by sampling equal rows per class; `n_rules_broken`, `source_row`, `source_timestamp` are diagnostics, not features. Resolve from the CSVs what the script cannot tell you: the real row count (the overview says 44,141 but the README estimates about 33,850 from a mock), 10 vs 11 classes, which mode was used, and the test months. Write `processed_data/sensor/tomato/manifest.json` and `docs/datasets/tomato_sensor_contract.md`, stating the README's own remark that the Kolar file looks simulated. ⚠️ UNCLEAR: the per-rule severity scales depend on the full Kolar file (`kolar_tomato_precision_agri_dataset.csv`, 200,000 rows), which is not in the repo; without it primary-mode labels cannot be fully reproduced (see the rules-oracle task).
**Dependencies:** T1-001, T2-004
**Acceptance Criteria:**

- [x] Every item above is confirmed from the CSVs or listed as a discrepancy
- [x] Manifest contains the ordered feature list, row counts per class and per month
- [x] Mode used (primary or single) and the test months are recorded

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T2-008: Archive the Kolar source file and record its provenance and severity scales
**Description:** Download `kolar_tomato_precision_agri_dataset.csv` (200,000 rows) again from its Kaggle source (the team confirmed it can be re-downloaded). Keep it **outside git** and read-only; it is the input to `build_tomato_dataset.py`. The owner can update a Kaggle dataset, so record the version/date and SHA-256 immediately. Record in the tomato provenance sheet: SHA-256, row and column counts, source URL, version/date, licence, and the exact command line used to build the 44k-row set (mode, `--per-class`, `--test-frac`, seed). Using the script's own mapping, compute the nine per-rule severity scales (largest deviation past each threshold over the whole mapped file) once and store them in `docs/datasets/tomato_sensor_contract.md` and the manifest, so the label oracle never needs the raw file again. The original copy's checksum was never recorded, so the re-download is verified by the reproduction task, not by a hash comparison.
**Dependencies:** T2-007
**Acceptance Criteria:**

- [x] Download date, source version, SHA-256 and counts are recorded
- [x] The nine severity scales are stored as constants with the script version that produced them
- [x] The raw file is not committed to git

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T2-009: Re-run build_tomato_dataset.py and compare with the existing CSVs
**Description:** In a temporary folder run the script with the recorded options and seed and compare the output with `tomato_hydroponic_{full,train,test}.csv` by row identity (`source_row`) and hash. Explain every difference. One thing to explain: 44,141 rows cannot be an equal-per-class total for 10 classes (4,414.1 each) or 11 classes (4,013.7 each), so the CSVs may come from different options or another script version. This is also the check that the re-downloaded file matches the one originally used. If the output differs, record the differences and ask the teammates which script version and file produced the shipped CSVs before trusting the scales.
**Dependencies:** T2-008
**Acceptance Criteria:**

- [x] Result is 'identical' or each difference is explained
- [x] Script version and options that match the shipped CSVs are recorded
- [x] Nothing in `sensor_data/` was overwritten

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T2-010: Create the lettuce sensor validation set (grouped hold-out)
**Description:** Write `scripts/make_sensor_validation.py <crop>` and run it for lettuce. Reconstruct the real training pool as `hydroponic_lettuce_full.csv` minus the test days (every 4th calendar day of each experiment, group key = `experiment` + `date`) and minus the two dropped classes; confirm that this pool equals the 558 real rows behind `train.csv`. Hold out whole days from that pool as validation (default: every 4th remaining day of each experiment; record the exact fraction). Then re-apply the script's own balancing (oversample to the majority count with replacement, seed 42, `min_keep` 8) to the **new training pool only**. Write `processed_data/sensor/lettuce/{train,val,test}.csv` as v2 with a manifest; the original files stay as v1. Limitation to record: neighbouring days of the same tank are correlated, so day-grouped validation is optimistic compared with contiguous blocks. ⚠️ UNCLEAR: `pH_Low` has 10 real rows in total, so it may have almost none left in train after validation and test; report class coverage and ask before merging or dropping any class.
**Dependencies:** T2-005
**Acceptance Criteria:**

- [x] Test rows are identical to the original test file
- [x] No (experiment, date) pair appears in more than one of train/val/test
- [x] Per-class real-row counts per partition are in the manifest and any class under 5 real training rows is flagged

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit and writes the concept note.
**Learning Focus:** Hold-out validation design for time-correlated data → `docs/learning/holdout-validation-design-for-timecorrelated-data.md`

#### T2-011: Create the spinach sensor validation set (six-hour blocks)
**Description:** Run `scripts/make_sensor_validation.py spinach` on `spinach_7class_train.csv`: build the block key from `bench`, the date and hour//6 of `source_timestamp_EDT`, and hold out whole blocks (about 15 percent of training rows) as validation. Keep the test file untouched. The data is balanced by drawing, not oversampling, so do not resample: per-class training counts will fall below 160, which is accepted and handled with class weights if needed. Write v2 files and a manifest. A block can contain rows of several classes, so report per-class counts after the hold-out.
**Dependencies:** T2-006, T2-010
**Acceptance Criteria:**

- [x] Test rows unchanged
- [x] No block key appears in more than one partition
- [x] Per-class counts per partition are in the manifest

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T2-012: Create the tomato sensor validation set (calendar months)
**Description:** Run `scripts/make_sensor_validation.py tomato` on `tomato_hydroponic_train.csv`: derive the month from `source_timestamp`, keep the existing test months, and hold out whole months from the training months as validation (about 15-20 percent of training rows; list the months). The script has no validation partition, so this is new. Never split a month across partitions; do not resample (classes were balanced by sampling in the build script). Report per-class counts per partition, especially for `Normal`, the rarest class in the source.
**Dependencies:** T2-007, T2-010
**Acceptance Criteria:**

- [x] Test months unchanged
- [x] Month lists per partition are recorded and disjoint
- [x] Per-class counts per partition are in the manifest

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T2-013: Write the image dataset audit script (counts, sizes, corrupt files)
**Description:** `scripts/audit_images.py <crop>` reports class counts, image dimensions/format distribution, unreadable files, and whether images are whole-plant or leaf crops (manual sample check). Sample 30 images per class and record whether the images are single-leaf, multi-leaf, or whole-plant (the team states spinach and tomato images are leaf images; confirm it). Handle the nested layout (`Spinach_Unhealthy/<class>`, `Tomato_Unhealthy/<class>`). Save `reports/data/image_audit_<crop>.json` for all three crops. Record actual counts; do not copy them from source descriptions (Overview §12.3).
**Dependencies:** T1-001, T2-002, T2-003, T2-004
**Acceptance Criteria:**

- [x] A JSON report exists for each crop
- [x] Corrupt files are listed, not deleted
- [x] Single-leaf vs whole-plant status per crop is recorded from the sample
- [x] Class labels are the exact folder names and every difference from Overview §6 is listed (e.g. `Tomato_Powdery_Mildew`)

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T2-014: Detect exact and near-duplicate images
**Description:** Add `scripts/find_duplicates.py`: SHA-256 for exact duplicates and a perceptual hash with a documented Hamming-distance threshold for near-duplicates. Output a report only; do not delete anything. ⚠️ UNCLEAR: choice of perceptual-hash library and threshold; start conservative and note the choice in the report.
**Dependencies:** T2-013
**Acceptance Criteria:**

- [x] Duplicate groups are reported per crop
- [x] Threshold and library are documented
- [x] No file is modified or deleted

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit and writes the concept note.
**Learning Focus:** Near-duplicate detection and why duplicates across splits inflate image accuracy → `docs/learning/nearduplicate-detection-and-why-duplicates-across.md`

#### T2-015: Create or preserve the 70/15/15 image split manifests
**Description:** Write `scripts/make_image_splits.py`: stratified 70/15/15 per crop with a fixed seed, performed after duplicate checks, with duplicate groups kept together; leaf crops created later (Phase 5) inherit the split of their source image, so this task splits whole images only. No split manifest exists yet; write them with row/class counts.
**Dependencies:** T2-014
**Acceptance Criteria:**

- [x] No source-image identity appears in more than one split
- [x] Split is stratified per class with a recorded seed
- [x] Manifests contain row and class counts

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T2-016: Implement the class-label registry and display mapping
**Description:** Create `ml/labels.py` with the labels taken from the actual folder names under `image_data/`: lettuce 4 (`Lettuce_Healthy`, `Lettuce_Nitrogen-Deficiency`, `Lettuce_Phosphorus-Deficiency`, `Lettuce_Potassium-Deficiency`), spinach 6 (`Spinach_Healthy` plus 5 under `Spinach_Unhealthy/`), tomato 11 (`Tomato_Healthy` plus 10 under `Tomato_Unhealthy/`, including `Tomato_Powdery_Mildew`, which is not in Overview §14.2), plus the sensor class lists. Display mapping is deterministic (`healthy` or `unhealthy/<original class>`). Keep source spelling exactly (including the space in `Tomato_Spider_mites Two-spotted_spider_mite`). Unknown labels raise an error; never infer a class from a filename.
**Dependencies:** T1-002, T2-013
**Acceptance Criteria:**

- [x] Label lists equal the folder names on disk (21 image classes in total)
- [x] Unknown label raises a clear error
- [x] Display mapping is deterministic

**Estimated Effort:** 45 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T2-017: Update PROJECT_OVERVIEW.md with verified facts and decisions
**Description:** Deliberately update `PROJECT_OVERVIEW.md` §4.1, §6, §11, §12, §13, §14, §15 and §31 with: the verified folder layout; corrected class lists (tomato images have 11 classes); decisions D1-D10 from the PRD; and the discrepancies found in the audits (lettuce script docstring says test = EXP.3 but code and README use every 4th day; tomato README estimates about 33,850 rows vs 44,141 in the overview; README file names differ from the repo). Add a dated 'Changes' note at the top. Change no other section.
**Dependencies:** T2-016, T2-013, T2-007
**Acceptance Criteria:**

- [x] Class lists match `ml/labels.py`
- [x] Change note is dated
- [x] Both members reviewed the diff

**Estimated Effort:** 30 min
**Assigned To:** Full-stack
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T2-018: Define sensor feature schemas and leakage blacklists as a single source of truth
**Description:** Create `ml/sensor_models/schemas.py` with ordered feature lists, units, hard (physically impossible) bounds, and an explicit **leakage blacklist** per crop. Lettuce features `ph, ec, water_temp, humidity, air_temp` (hard bounds from the script's plausibility limits: pH 3-10, EC 0-6, air temperature 0-55, humidity 0-100, water temperature 0-45); blacklist: `label_next`, the trend columns (`*_prev`, `*_delta`, `*_roll3`), daily aggregates, `tds`, `co2`, `lux_meter`, `hours_logged`, `experiment`, `replicate`, `treatment`, `date`, `day_index`, `slot`, `timestamp`. Spinach features `air_temperature_C, relative_humidity_percent, water_temperature_C, ph, ec_mS_cm`; blacklist `tds_ppm`, `label_id`, `humidity_source`, `sample_id`, `source_timestamp_EDT`, `bench`. Tomato features: the eight names from the tomato manifest; blacklist `n_rules_broken`, `source_row`, `source_timestamp`, `label_id`, `sample_id`. Import from here in training and inference. ⚠️ UNCLEAR: hard bounds for spinach and tomato fields (EC, N, P, K); start from physical limits and list them for team review.
**Dependencies:** T2-007, T2-016
**Acceptance Criteria:**

- [x] Feature order matches the audited order
- [x] Blacklists are enforced by a function that raises on any blacklisted column in a feature matrix
- [x] Hard bounds are separate from the normal-band hints

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T2-019: Add a split-manifest hash guard
**Description:** Create `ml/evaluation/split_guard.py` that stores SHA-256 hashes of prepared split files in `processed_data/split_hashes.json` and verifies them before any training or evaluation. Training scripts must call it and abort on mismatch (Overview §9.2: do not overwrite prepared splits).
**Dependencies:** T2-010, T2-011, T2-012, T2-015
**Acceptance Criteria:**

- [x] Altering a split file makes verification fail
- [x] Guard is invoked by the training entry points
- [x] Hash file is committed

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T2-020: Build the dataset manifest generator
**Description:** `scripts/build_manifest.py <dataset>` writes JSON with the Overview §9.1 fields: counts before/after preprocessing, class counts per split, split protocol, preprocessing script version (git commit), derived/synthetic flags. Fail loudly on unexplained count changes.
**Dependencies:** T2-019, T2-002, T2-003, T2-004
**Acceptance Criteria:**

- [x] Manifest exists for each sensor and image dataset
- [x] Count changes between stages are explained or the script fails
- [x] Output is deterministic

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T2-021: Create SQLAlchemy models and the initial Alembic migration
**Description:** Implement `analysis_records`, `predictions`, `inputs`, `recommendations` as described in TECH_RULES (UUID keys, `observed_at` as timestamptz plus `observed_tz`, `source` constrained to manual/live/imported, JSONB for warnings/evidence). Generate the first Alembic migration.
**Dependencies:** T1-011, T1-009
**Acceptance Criteria:**

- [x] `alembic upgrade head` creates all tables
- [x] Downgrade works
- [x] `source` rejects values outside the allowed set

**Estimated Effort:** 60 min
**Assigned To:** Backend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T2-022: Implement repositories and session management
**Description:** Create `backend/app/database/repositories.py` with `create_record`, `get_record`, `list_records(limit, offset, filters)`; a session dependency; and an explicit result type that distinguishes 'saved' from 'not saved' so the API can never claim a failed save succeeded.
**Dependencies:** T2-021
**Acceptance Criteria:**

- [x] Round-trip preserves the original timestamp and timezone
- [x] List is paginated
- [x] DB failure surfaces as a typed error

**Estimated Effort:** 45 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T2-023: Add history indexes
**Description:** Migration adding indexes on `analysis_records(created_at DESC)`, `(crop, created_at)`, and `predictions(record_id)`.
**Dependencies:** T2-021
**Acceptance Criteria:**

- [x] Migration applies and reverts
- [x] `EXPLAIN` on the list query uses an index
- [x] Migration is committed

**Estimated Effort:** 15 min
**Assigned To:** Backend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

---

## Phase 3: Security Baseline (no login in v1)
*Goal: Authentication is explicitly out of scope (Overview §2.3); this phase delivers input validation, error handling, rate limiting, secrets and transport rules instead.*

#### T3-001: Write the ADR: no authentication in v1
**Description:** Create `docs/decisions/0005-no-auth.md`: why there is no login, which capabilities are therefore withheld over HTTP (no model promotion, no dataset writes, no code execution), and what triggers a new threat model.
**Dependencies:** T1-002
**Acceptance Criteria:**

- [x] ADR lists withheld capabilities
- [x] ADR states the trigger for adding auth
- [x] README security note links to it

**Estimated Effort:** 15 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T3-002: Implement server-side upload validation utility
**Description:** `backend/app/core/uploads.py`: check content-type and magic bytes, decode with an image library, enforce `MAX_UPLOAD_MB` and dimension caps, strip EXIF orientation safely, save under a generated UUID name in `uploads/` (never the client filename). Raise typed errors `UNSUPPORTED_MEDIA`, `PAYLOAD_TOO_LARGE`, `INVALID_INPUT`. ⚠️ UNCLEAR: maximum upload size (default 10 MB).
**Dependencies:** T1-009, T1-004
**Acceptance Criteria:**

- [x] A renamed text file with a .jpg extension is rejected
- [x] Path traversal in filename has no effect
- [x] Oversized file returns 413 code `PAYLOAD_TOO_LARGE`

**Estimated Effort:** 60 min
**Assigned To:** Backend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T3-003: Create per-crop sensor request validators
**Description:** Pydantic models per crop built from `ml/sensor_models/schemas.py`: `extra='forbid'`, numeric bounds, explicit units, and Optional fields so 'missing' is distinguishable from 0. Return field-level errors naming the field and unit. Missing required field raises `MISSING_FIELD`; no coercion of null to zero. ⚠️ UNCLEAR: whether any model should accept partially missing input; default is reject.
**Dependencies:** T2-018
**Acceptance Criteria:**

- [x] Unknown field is rejected
- [x] `null` and 0 are treated differently
- [x] Error message names field, unit, and allowed range

**Estimated Effort:** 60 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T3-004: Implement the shared error envelope and exception handlers
**Description:** Create `backend/app/core/errors.py` with the envelope `{error:{code,message,details,retryable}}` and handlers for validation errors, `SEGMENTATION_FAILED`, `SCHEMA_MISMATCH`, `MODEL_UNAVAILABLE`, `DEVICE_UNAVAILABLE`, `PERSISTENCE_FAILED`, and unhandled exceptions (generic message, logged with `request_id`). Status codes follow the TECH_RULES table.
**Dependencies:** T1-009
**Acceptance Criteria:**

- [x] Each code returns the documented HTTP status
- [x] Unhandled error response contains no stack trace or path
- [x] `retryable` is true only for segmentation failure and device states

**Estimated Effort:** 45 min
**Assigned To:** Backend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T3-005: Configure CORS allow-list and security headers
**Description:** Apply `ALLOWED_ORIGINS` to CORS (no wildcard) and add basic security headers (`X-Content-Type-Options`, `Referrer-Policy`, `Content-Security-Policy` suitable for the Next.js app).
**Dependencies:** T1-009, T1-004
**Acceptance Criteria:**

- [x] Disallowed origin is blocked
- [x] Headers present on responses
- [x] Allowed origin works from the dev frontend

**Estimated Effort:** 30 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T3-006: Add rate limiting
**Description:** Per-IP rate limits using SlowAPI or equivalent, with stricter limits on `/predict/image` and `/analyze`; exceed returns 429 using the error envelope.
**Dependencies:** T3-004
**Acceptance Criteria:**

- [x] Burst test returns 429 in the standard envelope
- [x] Limits are configurable via env
- [x] Health endpoint is exempt

**Estimated Effort:** 30 min
**Assigned To:** Backend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T3-007: Add the trusted model-path guard
**Description:** `backend/app/core/paths.py`: resolve and verify that any artifact path lies under `MODEL_REGISTRY_PATH`; reject symlink escapes. All model loading goes through it. Never unpickle files outside configured directories.
**Dependencies:** T1-004
**Acceptance Criteria:**

- [x] Path outside the registry is rejected
- [x] Symlink escape is rejected
- [x] Loader uses the guard

**Estimated Effort:** 30 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T3-008: Document HTTPS/TLS for non-localhost deployment
**Description:** Write `docs/deployment/https.md` with a sample reverse-proxy config (Caddy or nginx) terminating TLS, HSTS at the proxy, and the rule that plain HTTP is only for localhost development.
**Dependencies:** T1-009
**Acceptance Criteria:**

- [x] Doc includes a working sample config
- [x] States localhost-only exception
- [x] Linked from README

**Estimated Effort:** 30 min
**Assigned To:** DevOps
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

---

## Phase 4: ML Core: Sensor Models
*Goal: Leakage-safe evaluation tooling, a model registry, and Decision Tree / Random Forest / SVM (plus baselines) for all three crops, selected by validation macro-F1.*

**Phase gate:** Teach-back and self-check completed by BOTH members (T4-026) before Phase 6 sensor endpoints are treated as final.

#### T4-001: Notebook: why random splits inflate scores on correlated time-series data
**Description:** Create `notebooks/01_leakage_demo.ipynb` using ONLY the lettuce training partition: compare validation macro-F1 from a random row-wise split against a split grouped by calendar date, with a fixed seed. Write the navigator's concept note explaining the gap. The held-out test file must not be loaded in this notebook.
**Dependencies:** T2-005
**Acceptance Criteria:**

- [x] Notebook runs top to bottom with a fixed seed
- [x] Both scores and the explanation are shown
- [x] A grep of the notebook shows no reference to the test file

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit and writes the concept note.
**Learning Focus:** Data leakage and grouped / time-based splits → `docs/learning/data-leakage-and-grouped-timebased-splits.md`

#### T4-002: Record the sensor validation protocol (ADR)
**Description:** Write `docs/decisions/0006-sensor-validation-protocol.md`: validation sets are the grouped hold-outs created in Phase 2 (calendar day / six-hour block / calendar month) and are used for model selection and early stopping; hyperparameter search uses grouped CV inside the training partition with oversampling performed inside each fold; the test partition is evaluated once. State why the lettuce report's GridSearchCV on already-oversampled rows can leak duplicates across folds and how this protocol avoids it.
**Dependencies:** T2-010, T2-011, T2-012, T4-001
**Acceptance Criteria:**

- [x] Protocol is written per crop with the group key named
- [x] Oversampling-inside-fold rule is stated
- [x] Both members signed off in the commit review

**Estimated Effort:** 30 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T4-003: From-scratch: Gini impurity and best split
**Description:** In `notebooks/02_gini_from_scratch.ipynb` implement Gini impurity and a single greedy best-split search over a numeric feature using NumPy, run it on a 2-feature slice of the lettuce training data, and compare the chosen threshold with `DecisionTreeClassifier(max_depth=1)`. Navigator writes the concept note.
**Dependencies:** T4-002
**Acceptance Criteria:**

- [x] Your threshold equals scikit-learn's (or the difference is explained)
- [x] Notebook includes a hand-worked example
- [x] Self-check questions answered in the concept note

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit and writes the concept note.
**Learning Focus:** Decision trees: Gini impurity and greedy splits → `docs/learning/decision-trees-gini-impurity-and-greedy.md`

#### T4-004: From-scratch: confusion matrix, precision, recall, macro-F1
**Description:** Implement confusion matrix and per-class/macro precision, recall, F1 in NumPy in `notebooks/03_metrics_from_scratch.ipynb` and verify against `sklearn.metrics` on a small multi-class example, including a class with zero predictions. Explain why accuracy hides rare-class failure.
**Dependencies:** T4-002
**Acceptance Criteria:**

- [x] Results match scikit-learn to numerical tolerance
- [x] Zero-support/zero-prediction behaviour is explained
- [x] Concept note written by the navigator

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit and writes the concept note.
**Learning Focus:** Confusion matrix, precision/recall, macro-F1 → `docs/learning/confusion-matrix-precisionrecall-macrof1.md`

#### T4-005: Implement ml/evaluation/metrics.py
**Description:** Function returning accuracy, macro precision/recall/F1, per-class precision/recall/F1/support, and confusion matrix (labels in fixed order) as one dict, using scikit-learn; reject inputs with label sets that differ from the model's class list.
**Dependencies:** T4-004
**Acceptance Criteria:**

- [x] All Overview §18.2 metric fields present
- [x] Class order is fixed and explicit
- [x] Unknown label raises an error

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T4-006: Implement the evaluation report writer
**Description:** `ml/evaluation/report.py` writes JSON and Markdown reports under `reports/<crop>/` containing metrics, train/validation/test counts, class counts per split, split protocol, dataset manifest reference, label-generation method, hyperparameters, seeds, device used, library versions, and a mandatory 'Limitations' block (rule-derived labels, oversampling, etc.). Reports state which split each metric came from.
**Dependencies:** T4-005, T2-020
**Acceptance Criteria:**

- [x] Every Overview §18.2 item has a field
- [x] Validation and test metrics are never mixed
- [x] Limitations block is always included

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T4-007: Implement grouped / time-aware CV with in-fold oversampling
**Description:** `ml/evaluation/cv.py`: wrappers around `GroupKFold` (or an equivalent) that accept group keys, and a helper that oversamples minority classes inside each training fold only. Include a guard that raises if any group appears in both train and validation of a fold.
**Dependencies:** T4-002, T4-005
**Acceptance Criteria:**

- [x] Group overlap raises an error
- [x] Oversampling never touches the validation fold
- [x] Unit test cases are listed for T8 tasks

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit and writes the concept note.
**Learning Focus:** Cross-validation and class imbalance (oversampling inside folds) → `docs/learning/crossvalidation-and-class-imbalance-oversampling-inside.md`

#### T4-008: Define the model-metadata schema
**Description:** `ml/registry/metadata_schema.py`: typed schema covering every field in Overview §24.1 (crop, task, modality, family, commit, dataset manifest hash, classes and display mapping, split protocol/counts, ordered feature list and units, input unit (always one leaf for image models), image size/normalisation/channel order, segmentation and feature config, rule version, preprocessing version, hyperparameters, validation metrics, final test metrics when available, confidence/threshold settings, dependency versions, expected input schema, training date, `experimental` flag, `status`).
**Dependencies:** T2-016, T2-018
**Acceptance Criteria:**

- [x] Schema validation fails on any missing mandatory field
- [x] `experimental` and `status` fields exist
- [x] Example metadata file is committed under `models/manifests/`

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T4-009: Implement registry save / load / promote / rollback
**Description:** `ml/registry/registry.py` and `scripts/promote_model.py`: save artifacts under versioned directories, promotion is an explicit command (never 'newest wins'), keep the previous known-good version for rollback, and verify feature-schema compatibility on load. Load goes through the trusted path guard.
**Dependencies:** T4-008, T3-007
**Acceptance Criteria:**

- [x] Promote then rollback restores the previous version
- [x] Loading an artifact with a mismatched feature list fails with `SCHEMA_MISMATCH`
- [x] Promotion is CLI-only

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T4-010: Build the sensor training pipeline skeleton
**Description:** `ml/sensor_models/train_sensor.py` CLI (`--crop --model --seed`): verify split hashes, load prepared splits, validate columns against the schema (blacklisted fields absent, target not in features), fit the scaler on training data only when the model needs it, train, evaluate on validation per the protocol ADR, save artifact plus metadata, write the report. It must not read the test split.
**Dependencies:** T4-002, T2-019, T2-018, T4-006, T4-007, T4-009
**Acceptance Criteria:**

- [x] Running with the test file temporarily made unreadable still succeeds
- [x] Leakage-blacklist columns cause a hard failure
- [x] Scaler is saved inside the artifact

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T4-011: Add the Decision Tree trainer
**Description:** Plug `DecisionTreeClassifier` into the pipeline with configurable `max_depth`, `min_samples_leaf`, and `criterion`; save the exported text rules next to the artifact.
**Dependencies:** T4-010
**Acceptance Criteria:**

- [x] Runs for lettuce end to end
- [x] Exported rules file is written
- [x] Seeds recorded

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit and writes the concept note.
**Learning Focus:** Decision tree hyperparameters, depth, and overfitting → `docs/learning/decision-tree-hyperparameters-depth-and-overfitting.md`

#### T4-012: Add the Random Forest trainer
**Description:** Plug `RandomForestClassifier` into the pipeline (`n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight` options) and save feature importances.
**Dependencies:** T4-011
**Acceptance Criteria:**

- [x] Runs for lettuce end to end
- [x] Feature importances saved
- [x] Seeds recorded

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit and writes the concept note.
**Learning Focus:** Random Forest: bagging, feature subsampling, variance reduction → `docs/learning/random-forest-bagging-feature-subsampling-variance.md`

#### T4-013: Add the SVM trainer (with scaling)
**Description:** Plug `SVC` into the pipeline using a scaler fitted on training data only; options for `C`, `gamma`, `kernel`. Include a one-off experiment showing SVM performance with and without scaling on the training/validation data.
**Dependencies:** T4-012
**Acceptance Criteria:**

- [x] Scaling comparison is recorded in the concept note
- [x] Scaler is part of the saved artifact
- [x] Probability output setting is documented

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit and writes the concept note.
**Learning Focus:** SVM: margin, C, RBF kernel and gamma, why scaling matters → `docs/learning/svm-margin-c-rbf-kernel-and.md`

#### T4-014: Add KNN and Logistic Regression baselines
**Description:** Add `KNeighborsClassifier` and `LogisticRegression` as baselines (scaled inputs, regularisation options) so the report compares at least five families as in the lettuce report.
**Dependencies:** T4-013
**Acceptance Criteria:**

- [x] Both models train end to end
- [x] Baselines appear in the comparison table
- [x] Concept note explains why a distance-based model is sensitive to feature scale

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit and writes the concept note.
**Learning Focus:** KNN and logistic regression baselines (distance, regularisation) → `docs/learning/knn-and-logistic-regression-baselines-distance.md`

#### T4-015: Implement hyperparameter search with grouped CV on macro-F1
**Description:** `ml/sensor_models/tune.py`: grid search per model family using the CV wrappers and macro-F1; search spaces in versioned YAML under `ml/sensor_models/configs/`; results saved as JSON. Test labels are never passed in.
**Dependencies:** T4-011, T4-012, T4-013, T4-014, T4-007
**Acceptance Criteria:**

- [x] Search uses group-aware folds only
- [x] Best parameters and CV scores are saved
- [x] Runtime and seed recorded

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit and writes the concept note.
**Learning Focus:** Hyperparameter search with grouped CV and its pitfalls → `docs/learning/hyperparameter-search-with-grouped-cv-and.md`

#### T4-016: Implement the rule-label reference function per crop
**Description:** `ml/sensor_models/rules.py`: pure functions that reproduce each crop's labels exactly as the scripts do. Lettuce: the five action bands, one variable out gives that variable's class, two or more give `Multiple_Stress`. Spinach: the six threshold rules (every non-Normal row breaks exactly one). Tomato: nine threshold rules where the label is the most severe fault, severity being the deviation divided by the largest deviation per rule over the whole mapped Kolar file. Used only as a test oracle and for explanations, never as a substitute for the model. Use the severity scales recorded by the source-archive task; if they are 'not available', restrict the tomato oracle to rows that break at most one rule and say so.
**Dependencies:** T2-006, T2-007, T2-005, T2-008
**Acceptance Criteria:**

- [x] Oracle reproduces the labels in the prepared CSVs (mismatch count reported)
- [x] Boundary values (exactly on a threshold) behave as in the scripts
- [x] Tomato scales are sourced or the restriction is documented

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T4-017: Run and report lettuce sensor experiments
**Description:** Train and tune DT, RF, SVM, KNN, LogReg for lettuce with the seven-class setup, per the validation protocol. Write the comparison report (validation metrics only) and note that the model reproduces project thresholds, not biological stress. The lettuce README already reports grouped 5-fold CV macro-F1 of 0.837 +/- 0.075 for a random forest and calls the single-split 0.994 accuracy one favourable split; compare your numbers with those and explain differences.
**Dependencies:** T4-015, T4-016
**Acceptance Criteria:**

- [x] Report covers all five families with required metrics
- [x] No test metrics appear
- [x] Limitations block states rule-derived labels

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T4-018: Run and report spinach sensor experiments
**Description:** Same as the lettuce run for the spinach schema, preserving six-hour block separation and the 160/40 per-class split; exclude `tds_ppm`, `label_id`, `humidity_source`, `sample_id`, `source_timestamp_EDT`, `bench`.
**Dependencies:** T4-015, T4-016
**Acceptance Criteria:**

- [x] Report lists excluded fields
- [x] Group separation test passes
- [x] Limitations block states mixed real/synthetic features

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T4-019: Run and report tomato sensor experiments
**Description:** Same for the tomato schema using the month-based train/validation/test partitions created in the validation-split task. Exclude `n_rules_broken`, `source_row` and `source_timestamp` from features. Mark artifacts `experimental=true` until live-sensor compatibility is validated (Overview §15.7); the tomato README itself says the Kolar source looks simulated and the soil-to-hydroponic mappings are proxies.
**Dependencies:** T4-015, T4-016
**Acceptance Criteria:**

- [x] Month-based assignment is used
- [x] Metadata has `experimental=true`
- [x] Report documents transformed and synthetic features

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T4-020: Train a live-compatible tomato sensor variant
**Description:** The campus system now grows tomato only, and the tomato model needs eight features including `nitrogen_mg_L`, `phosphorus_mg_L`, `potassium_mg_L`. Using the field list from the campus-server probe, check which of the eight features the server really provides. If any are missing, train the same families on the subset that exists (expected: air temperature, humidity, water temperature, pH, EC), exclude classes that depend on the missing features (N/P/K deficiency) from this variant, and register it as a separate model with a `feature_set` entry in its metadata. Report it separately from the full eight-feature model. ⚠️ UNCLEAR: which fields the server provides; do not start before the probe ADR exists. The live variant stays `experimental=true` because the training data is proxy-transformed and partly synthetic.
**Dependencies:** T4-019, T1-016
**Acceptance Criteria:**

- [x] Feature availability table (server field vs model feature, units compared) is in the report
- [x] Variant has its own metadata, report and class list
- [x] Live adapter maps only to features the variant uses

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit and writes the concept note.
**Learning Focus:** Train/serve skew and feature availability (live-compatible models) → `docs/learning/trainserve-skew-and-feature-availability-livecompatible.md`

#### T4-021: Analysis: show that trees recover the label rules
**Description:** For each crop export the tree's thresholds and compare them with the documented bands; explain why 0.994 test accuracy on rule-derived labels does not mean field accuracy. Write `reports/analysis/tree_vs_rules.md`.
**Dependencies:** T4-017, T4-018, T4-019
**Acceptance Criteria:**

- [x] Learned thresholds are tabulated beside documented bands
- [x] Document answers LR-5 from the PRD in plain language
- [x] Both members can restate the conclusion in one paragraph

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit and writes the concept note.
**Learning Focus:** Why rule-derived labels inflate scores → `docs/learning/why-rulederived-labels-inflate-scores.md`

#### T4-022: Run grouped 5-fold CV by calendar date for lettuce
**Description:** Execute the stronger check recommended in Overview §11.7/§18.4 (the lettuce README reports 0.837 +/- 0.075 macro-F1 with a random forest) using groups = (experiment, date) and in-fold oversampling, and report mean and spread of macro-F1 per model. Make this the headline lettuce score and label the single held-out split as one favourable split. Update the audit note that the check was actually run.
**Dependencies:** T4-017
**Acceptance Criteria:**

- [x] Fold composition lists dates per fold with no overlap
- [x] Report shows mean and standard deviation and compares with the README's figure
- [x] Result is not described as field accuracy

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T4-023: Implement model selection by validation macro-F1
**Description:** `scripts/select_model.py <crop>`: reads validation reports only, chooses the best by macro-F1, and writes `reports/<crop>/sensor_selection.json` with tie-break and per-class failure notes (Overview §17.2). Any exception to macro-F1 must be recorded with a reason.
**Dependencies:** T4-017, T4-018, T4-019, T4-022
**Acceptance Criteria:**

- [x] Script fails if test metrics are supplied
- [x] Selection JSON lists runners-up
- [x] Per-class weaknesses are noted

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T4-024: Implement the run-once final test evaluation
**Description:** `scripts/final_eval_sensor.py <crop>`: evaluates only the selected model on the held-out test file, writes the final report, and appends to `reports/final_eval_log.jsonl`. Re-running requires `--force` and a written reason; the log makes any re-run visible.
**Dependencies:** T4-023
**Acceptance Criteria:**

- [x] First run writes metrics and a log entry
- [x] Second run without `--force` refuses
- [x] Report labels metrics as test results on the documented split

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T4-025: Promote the selected sensor models
**Description:** Use the registry to promote one sensor model per crop with complete metadata; tomato stays `experimental=true`. Record promotion rationale in `models/manifests/`.
**Dependencies:** T4-024, T4-009
**Acceptance Criteria:**

- [x] Three promoted entries exist
- [x] Registry validation script passes
- [x] Rollback tested once on a dummy version

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T4-026: Phase 4 teach-back and concept check
**Description:** Hold a teach-back session: each member explains half of the Phase 4 concepts (leakage, Gini, metrics, CV and imbalance, DT, RF, SVM, KNN/LogReg, tuning, rule-derived labels) without notes, then answers the self-check questions from each concept note. Record outcomes in `docs/learning/PROGRESS.md`.
**Dependencies:** T4-025, T4-021
**Acceptance Criteria:**

- [x] Every Phase 4 concept has a note and a teach-back entry for both members
- [x] Self-check results are recorded honestly (gaps become follow-up items)
- [x] Both members tick the phase gate

**Estimated Effort:** 90 min
**Assigned To:** QA
**Driver / Navigator:** Both. Pair session; each member presents half and quizzes the other.

---

## Phase 5: ML Core: Image Pipeline
*Goal: Feature extraction, SAM 2.1 segmentation with mask validation, classical image classifiers for all crops, and custom CNN / EfficientNetB0 for lettuce, compared fairly on validation macro-F1.*

**Phase gate:** Teach-back and self-check completed by BOTH members (T5-034).

#### T5-001: Notebook: colour spaces and vegetation indices (BGR vs RGB bug)
**Description:** In `notebooks/04_color_spaces.ipynb` load one lettuce image with OpenCV, show what happens when BGR is treated as RGB in `ExG = 2G - R - B` and `ExR = 1.4R - G`, then convert to HSV and visualise yellow/brown thresholds. Navigator writes the concept note.
**Dependencies:** T2-013
**Acceptance Criteria:**

- [x] Notebook shows the wrong-channel result and the correct one
- [x] HSV hue range is explained
- [x] Self-check questions answered

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit and writes the concept note.
**Learning Focus:** Colour spaces (BGR/RGB/HSV) and vegetation indices (ExG, ExR) → `docs/learning/colour-spaces-bgrrgbhsv-and-vegetation-indices.md`

#### T5-002: Create the shared feature-extraction utilities
**Description:** `ml/features/common.py`: single image loader that returns RGB with documented conventions, mask-aware pixel selection, NaN/Inf checks, and a `FeatureVector` helper that attaches `schema_version`, ordered names, and extraction config. A failed feature raises; it is never replaced by 0.
**Dependencies:** T5-001, T1-002
**Acceptance Criteria:**

- [x] Loader documents channel order
- [x] NaN/Inf raises a typed error
- [x] Schema version is attached to every vector

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T5-003: Implement GLCM utility (quantise 32, distance 1, four angles)
**Description:** `ml/features/glcm.py`: grayscale conversion, quantise to 32 levels, distance 1, angles 0/45/90/135 degrees averaged, outputs `glcm_contrast` and `glcm_entropy`. No implementation exists yet, so follow the Overview §16.4 defaults. Document the symmetric/normed flags and the logarithm base used for entropy.
**Dependencies:** T5-002
**Acceptance Criteria:**

- [x] Flags and log base are documented in the module docstring
- [x] Constant-colour image does not crash or yield NaN
- [x] Output matches a hand-computed 4x4 example

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit and writes the concept note.
**Learning Focus:** Texture features: GLCM contrast and entropy → `docs/learning/texture-features-glcm-contrast-and-entropy.md`

#### T5-004: Implement lettuce feature extractor (7 features)
**Description:** `ml/features/lettuce.py` returning `exg_mean, exr_mean, yellow_pixel_ratio, brown_pixel_ratio, mean_hue, mean_saturation, contour_compactness` in that order. HSV thresholds for yellow/brown are configuration saved with the feature dataset. `contour_compactness = 4*pi*A / P^2` with safe handling of zero/invalid perimeter. Input: individual leaf crops.
**Dependencies:** T5-003, T2-016
**Acceptance Criteria:**

- [x] Order and names match Overview §16.1
- [x] Zero perimeter produces an explicit error
- [x] HSV thresholds are stored in config, not hard-coded

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T5-005: Implement spinach feature extractor (6 features)
**Description:** `ml/features/spinach.py` returning `exg_mean, exr_mean, yellow_pixel_ratio, brown_pixel_ratio, relative_leaf_area, glcm_contrast`. ⚠️ UNCLEAR: denominator of `relative_leaf_area`; default is segmented foreground pixels divided by total image pixels, documented in code and metadata.
**Dependencies:** T5-003, T2-016
**Acceptance Criteria:**

- [x] Order matches Overview §16.2
- [x] Denominator is documented and stored in metadata
- [x] Empty mask raises an error

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T5-006: Decide whether droop_angle is defensible (keep or remove)
**Description:** No tomato extractor exists yet (the repo is greenfield), so `droop_angle` is only a candidate from the overview. Prototype a definition with an explicit reference axis and mask assumptions and test it on 20 images where the true orientation is known or obvious; if no stem or gravity reference can be justified from the images, it cannot be defined defensibly. Write `docs/decisions/0007-droop-angle.md` with a keep/replace/remove decision. If removed or replaced, all tomato features and dependent models must be regenerated and retrained (Overview §16.3, §29.1). ⚠️ UNCLEAR: whether the team prefers keeping a possibly weak feature; default is remove if no defensible reference axis exists.
**Dependencies:** T1-001, T5-003
**Acceptance Criteria:**

- [x] Decision is written with evidence
- [x] Impact list of datasets/models to regenerate is included
- [x] Both members agree in the commit review

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T5-007: Implement tomato feature extractor per the droop_angle decision
**Description:** `ml/features/tomato.py` with `glcm_contrast, glcm_entropy, bounding_aspect_ratio, solidity` (and `droop_angle` only if retained). Record exact definitions of aspect ratio (bounding box or rotated box) and solidity (contour area / convex hull area) in the module and metadata.
**Dependencies:** T5-006
**Acceptance Criteria:**

- [x] Feature list equals the ADR decision
- [x] Definitions are written in metadata
- [x] Degenerate contour (zero hull area) raises an error

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit and writes the concept note.
**Learning Focus:** Shape descriptors: compactness, solidity, aspect ratio → `docs/learning/shape-descriptors-compactness-solidity-aspect-ratio.md`

#### T5-008: Implement the SAM 2.1 segmenter wrapper
**Description:** `ml/segmentation/sam2_segmenter.py`: lazy-load via env paths, use the device manager, return a mask plus segmentation config/version, free memory on failure, and surface CUDA OOM as a typed error that triggers a smaller setting or CPU fallback. Use the prompting approach chosen in the SAM spike.
**Dependencies:** T1-007
**Acceptance Criteria:**

- [x] Segmenter returns mask plus config/version string
- [x] Simulated OOM triggers the documented fallback or a typed error
- [x] Model is not loaded until first use

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T5-009: Implement per-leaf mask validation
**Description:** `ml/segmentation/mask_validation.py`: for each candidate leaf mask check that it is non-empty, its area ratio is within bounds (not tiny, not the whole frame), it is not dominated by the border/background, and it is a single connected region. Return a typed verdict per mask (ok / retry / reject) with a reason. ⚠️ UNCLEAR: numeric bounds; tune them on the mask review (task after next) and record the final values in config.
**Dependencies:** T5-008
**Acceptance Criteria:**

- [x] Empty, tiny, and whole-image masks are rejected
- [x] Thresholds are config values with a documented origin
- [x] Every verdict carries a reason string

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T5-010: Implement leaf extraction (plant photo to individual leaves)
**Description:** `ml/segmentation/leaf_extraction.py`: run SAM 2.1 (approach chosen in the spike) on a plant photo, keep masks that pass the per-leaf validator, remove near-duplicate/overlapping masks, cap the leaf count, and return for each leaf an id, bounding box, mask, and masked crop (background handling documented). Leaves that fail validation are returned in a `rejected[]` list with reasons, never silently dropped. The same routine is used to build lettuce training crops, to mask leaf images for feature extraction, and at inference. A photo that is already a single leaf can bypass splitting but still gets its leaf mask.
**Dependencies:** T5-009, T5-008
**Acceptance Criteria:**

- [x] Returns leaves plus a `rejected[]` list with reasons
- [x] Overlapping duplicate masks are merged or removed (rule documented)
- [x] Same function is used in training-data creation and inference

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit and writes the concept note.
**Learning Focus:** Instance segmentation for per-leaf prediction (SAM automatic masks) → `docs/learning/instance-segmentation-for-perleaf-prediction-sam.md`

#### T5-011: Qualitative review of leaf extraction (20 images per crop)
**Description:** Run leaf extraction on 20 lettuce whole-plant images and on 20 leaf images each of spinach and tomato (plus sample campus camera frames if the probe produced any). Record each validator verdict and a manual judgement in `reports/analysis/mask_review.md`, and list failure modes (touching leaves, pots, reflections, background). Adjust validator thresholds only based on this review.
**Dependencies:** T5-010, T2-015
**Acceptance Criteria:**

- [x] At least 60 reviewed images with verdicts recorded
- [x] Failure modes are listed
- [x] Threshold changes reference the review

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T5-012: Generate lettuce leaf crops from whole-plant images with source mapping
**Description:** `image_data/Lettuce_Images` holds whole-plant images in four class folders, but all lettuce image models are leaf-level (team decision). Generate individual leaf crops with the leaf-extraction routine and write `processed_data/lettuce_leaf_crops/source_map.csv` (`crop_id`, `source_image_id`, `class`, `mask_verdict`). Each crop inherits the **split and the class label of its source image**; this is a weak label (a leaf from a deficient plant may look healthy) and must be stated as a limitation. Manually review a random sample of 30 crops per class and record the visibly inconsistent fraction in `reports/analysis/crop_label_review.md`. Record how many source images yield zero valid crops.
**Dependencies:** T5-010, T5-011, T2-015
**Acceptance Criteria:**

- [ ] Every crop has exactly one `source_image_id` and the same split and class as that image
- [ ] Manual review of 30 crops per class has a recorded inconsistency rate
- [ ] Source images with no valid crop are counted and listed

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit and writes the concept note.
**Learning Focus:** Weak labels and instance-level (per-leaf) prediction → `docs/learning/weak-labels-and-instancelevel-perleaf-prediction.md`

#### T5-013: Build feature datasets from split manifests
**Description:** `scripts/build_features.py <crop>`: read the split manifest, mask each leaf image with the same leaf-mask routine used at inference, extract features, write per-split CSVs to `processed_data/features/<crop>/` with schema version, ordered names, extraction config and counts; refuse to overwrite existing files without a flag. Lettuce crops inherit their source image's split.
**Dependencies:** T5-004, T5-005, T5-007, T5-009, T5-012, T2-015, T2-020, T2-019
**Acceptance Criteria:**

- [ ] Row counts per split match the manifest (or the difference is explained)
- [ ] No NaN/Inf rows are written
- [ ] Re-running without the flag does not overwrite

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T5-014: Train and report lettuce OpenCV-feature classifiers
**Description:** Train RF and SVM on lettuce leaf-crop features via a shared `ml/image_models/train_features.py` (scaler fitted on training only; hyperparameters via validation or grouped CV); report validation macro-F1 and per-class metrics.
**Dependencies:** T5-013, T4-015
**Acceptance Criteria:**

- [ ] Validation-only report written
- [ ] Feature order saved in metadata
- [ ] Class support shown

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T5-015: Train and report spinach OpenCV-feature classifiers
**Description:** Same as lettuce for the spinach leaf images as provided (masked with the shared leaf-mask routine).
**Dependencies:** T5-014
**Acceptance Criteria:**

- [ ] Report written with per-class metrics
- [ ] Mask/segmentation config is stored with the model
- [ ] Leaves that fail masking are excluded and counted in the report

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T5-016: Train and report tomato OpenCV-feature classifiers
**Description:** Same for tomato with the final feature schema after the droop_angle decision.
**Dependencies:** T5-014
**Acceptance Criteria:**

- [ ] Report written with per-class metrics
- [ ] Feature schema version matches the ADR
- [ ] Excluded images are counted

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T5-017: From-scratch: 2D convolution, pooling and backpropagation intuition
**Description:** In `notebooks/05_conv_from_scratch.ipynb` implement 2D convolution and max-pooling in NumPy, apply an edge filter to a leaf image, then use PyTorch autograd on a tiny conv layer to show gradients. Navigator writes the concept note.
**Dependencies:** T5-001
**Acceptance Criteria:**

- [x] NumPy convolution matches `torch.nn.functional.conv2d` on a small example
- [x] Notebook visualises feature maps
- [x] Self-check questions answered

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit and writes the concept note.
**Learning Focus:** Convolution, pooling, and backpropagation → `docs/learning/convolution-pooling-and-backpropagation.md`

#### T5-018: Implement PyTorch datasets and training-only augmentation
**Description:** `ml/image_models/data.py`: Dataset reading split manifests where every sample is one leaf (lettuce crops from the crop map; spinach and tomato leaf images as provided), resizing/normalisation recorded in metadata, augmentation applied to the training split only, `num_workers` and seeds configurable.
**Dependencies:** T2-015, T2-019, T5-012
**Acceptance Criteria:**

- [ ] Validation/test transforms contain no random operations
- [ ] Normalisation constants are saved in metadata
- [ ] Dataset refuses to mix leaf sources from different crops

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit and writes the concept note.
**Learning Focus:** Data augmentation (training only) and input normalisation → `docs/learning/data-augmentation-training-only-and-input.md`

#### T5-019: Design the custom CNN architecture module
**Description:** `ml/image_models/custom_cnn.py`: small configurable CNN (conv blocks, batch norm, pooling, dropout, classifier head) whose output size equals the crop's class count; log parameter count and a shape table.
**Dependencies:** T5-017, T2-016
**Acceptance Criteria:**

- [ ] Forward pass works on a dummy batch for each crop's class count
- [ ] Parameter count logged
- [ ] Architecture is described in the concept note

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T5-020: Implement the generic PyTorch training loop
**Description:** `ml/image_models/train_image.py`: device from the device manager, seeds, optimiser/LR settings from config, early stopping on validation macro-F1, best-checkpoint saving, CUDA OOM handling (smaller batch, then CPU or a clear error), and a run log of device actually used. No test-set access.
**Dependencies:** T5-018, T5-019, T4-005, T4-006, T4-008
**Acceptance Criteria:**

- [ ] Overfit test: loop can reach near-zero training loss on 20 images
- [ ] OOM path is exercised (simulated)
- [ ] Report records device and versions

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit and writes the concept note.
**Learning Focus:** Optimisers, learning rate, and early stopping on validation macro-F1 → `docs/learning/optimisers-learning-rate-and-early-stopping.md`

#### T5-021: Train the custom CNN on lettuce leaf crops
**Description:** Train on the lettuce leaf crops with documented hyperparameters, record the full configuration, and produce the validation report (macro-F1, per-class, confusion matrix, class support). Remember crop labels are inherited from the whole plant.
**Dependencies:** T5-020
**Acceptance Criteria:**

- [ ] Full hyperparameter configuration saved with the run
- [ ] Report uses the validation split only
- [ ] Device actually used is recorded

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T5-022: Build the EfficientNetB0 transfer-learning model
**Description:** `ml/image_models/efficientnet.py`: construct EfficientNetB0 from torchvision with pretrained weights, replace the classifier head, support a freeze-then-unfreeze schedule, and record weights identifier, input size, and normalisation. Verify the weights licence/terms. Do not assume hyperparameters; take them from config.
**Dependencies:** T5-020
**Acceptance Criteria:**

- [ ] Head output size matches class count
- [ ] Freeze/unfreeze schedule is configurable and logged
- [ ] Weights identifier saved in metadata

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit and writes the concept note.
**Learning Focus:** Transfer learning and fine-tuning (EfficientNetB0) → `docs/learning/transfer-learning-and-finetuning-efficientnetb0.md`

#### T5-023: Train EfficientNetB0 on lettuce leaf crops
**Description:** Run frozen-backbone then fine-tune stages on the lettuce leaf crops and produce the validation report.
**Dependencies:** T5-022, T5-021
**Acceptance Criteria:**

- [ ] Both stages' settings are logged
- [ ] Same split manifest hash as the CNN run
- [ ] Validation report produced

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T5-024: Train the custom CNN on spinach leaf images
**Description:** Same pipeline as the lettuce CNN with the 6 spinach classes, using the leaf images as provided (team decision).
**Dependencies:** T5-021
**Acceptance Criteria:**

- [ ] Validation report produced with per-class metrics
- [ ] Input type (leaf image as provided) recorded in metadata
- [ ] Device actually used recorded

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T5-025: Train EfficientNetB0 on spinach images
**Description:** Frozen-backbone then fine-tune stages with the 6 spinach classes and the same split manifest as the CNN run.
**Dependencies:** T5-023, T5-024
**Acceptance Criteria:**

- [ ] Both stages' settings logged
- [ ] Same split manifest hash as the CNN run
- [ ] Validation report produced

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T5-026: Train the custom CNN on tomato leaf images
**Description:** Same pipeline with the 11 tomato classes (including `Tomato_Powdery_Mildew`) using the leaf images as provided; pay attention to class imbalance and per-class F1.
**Dependencies:** T5-024
**Acceptance Criteria:**

- [ ] Validation report with per-class F1 for all 11 classes
- [ ] Class support shown
- [ ] Input type recorded

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T5-027: Train EfficientNetB0 on tomato images
**Description:** Frozen-backbone then fine-tune stages with the 11 tomato classes.
**Dependencies:** T5-025, T5-026
**Acceptance Criteria:**

- [ ] Both stages' settings logged
- [ ] Same split manifest hash as the CNN run
- [ ] Validation report produced

**Estimated Effort:** 90 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T5-028: Compare lettuce image models fairly and select one
**Description:** Compare custom CNN, EfficientNetB0, and OpenCV+RF/SVM on the same leaf-level split assignments using validation macro-F1 only. All three families consume individual leaves, so the comparison is like-for-like. Write `reports/lettuce/image_comparison.md`.
**Dependencies:** T5-023, T5-014
**Acceptance Criteria:**

- [ ] Same split manifest hash used for all models
- [ ] Table includes macro-F1, per-class F1, and parameter counts
- [ ] Leaf-level input confirmed for every family

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T5-029: Compare spinach image models fairly and select one
**Description:** Same procedure as the lettuce comparison: custom CNN, EfficientNetB0, and OpenCV+RF/SVM on identical split assignments, validation macro-F1 only. Write `reports/spinach/image_comparison.md`.
**Dependencies:** T5-025, T5-015, T5-028
**Acceptance Criteria:**

- [ ] Same split manifest hash for all models
- [ ] Table includes macro-F1 and per-class F1
- [ ] Leaf-level input confirmed for every family

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T5-030: Compare tomato image models fairly and select one
**Description:** Same procedure for the 11 tomato classes. Write `reports/tomato/image_comparison.md`.
**Dependencies:** T5-027, T5-016, T5-028
**Acceptance Criteria:**

- [ ] Same split manifest hash for all models
- [ ] Table includes macro-F1 and per-class F1
- [ ] Leaf-level input confirmed for every family

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T5-031: Derive per-leaf uncertainty thresholds from validation data
**Description:** For each promoted image/feature model, derive the confidence thresholds (high / moderate / uncertain) from validation predictions on leaves, with an optional temperature-scaling experiment fitted on validation. Store them in model metadata with a statement of what the score means (raw softmax is not a calibrated probability, and lettuce labels are inherited from the plant).
**Dependencies:** T5-028, T5-029, T5-030
**Acceptance Criteria:**

- [ ] Thresholds are computed from validation only
- [ ] Metadata states score semantics
- [ ] Reliability plot saved

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit and writes the concept note.
**Learning Focus:** Softmax confidence, calibration, and uncertainty thresholds → `docs/learning/softmax-confidence-calibration-and-uncertainty-thresholds.md`

#### T5-032: Run-once final test evaluation for image models
**Description:** `scripts/final_eval_image.py <crop>`: evaluate the selected model for each of the three crops on its test split, log to `reports/final_eval_log.jsonl`, refuse re-runs without `--force`.
**Dependencies:** T5-031, T4-024
**Acceptance Criteria:**

- [ ] Log entry is written for each evaluated model
- [ ] Re-run without `--force` refuses
- [ ] Reports label metrics as test results with split named

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T5-033: Promote image and feature models to the registry
**Description:** Promote the selected models with complete metadata (preprocessing, segmentation config, feature schema, thresholds). Note which crop/model combinations are NOT trained so the UI never implies they exist.
**Dependencies:** T5-032, T4-009
**Acceptance Criteria:**

- [ ] Registry validation passes
- [ ] Missing combinations are listed in `models/manifests/COVERAGE.md`
- [ ] Rollback tested

**Estimated Effort:** 45 min
**Assigned To:** ML
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T5-034: Phase 5 teach-back and concept check
**Description:** Each member teaches half of the Phase 5 concepts (colour spaces/indices, GLCM, shape features, SAM prompts, convolution/pooling/backprop, augmentation, optimisers, transfer learning, softmax/calibration) without notes, then completes the self-checks. Record results in `docs/learning/PROGRESS.md`.
**Dependencies:** T5-033
**Acceptance Criteria:**

- [ ] Every Phase 5 concept has a note and a teach-back for both members
- [ ] Gaps become tracked follow-ups
- [ ] Both members tick the phase gate

**Estimated Effort:** 90 min
**Assigned To:** QA
**Driver / Navigator:** Both. Pair session; each member presents half and quizzes the other.

---

## Phase 6: Core Backend: APIs & Services
*Goal: All Must Have endpoints and business logic. Route names are PROPOSED (Overview §22); align them with any existing code found in the T1 audit.*

#### T6-001: Define Pydantic request/response schemas
**Description:** In `backend/app/schemas/`: `PredictionResponse` (crop, modality, predicted_class, display_label, confidence, confidence_semantics, uncertain, experimental, model_id, model_version, timestamp, provenance, warnings, limitations), an image-specific `LeafPrediction` (leaf_id, bbox, mask_ref, predicted_class, display_label, confidence, uncertain) with `input_mode`, `leaves[]`, `rejected_leaves[]` and a `summary` (leaf count and counts per predicted class; no averaged confidence), `RecordCreate/RecordRead`, `RecommendationRead`, and the error envelope. `extra='forbid'` on requests. ⚠️ UNCLEAR: if schemas already exist (audit), extend them instead of replacing.
**Dependencies:** T3-004, T2-016, T3-003, T2-022
**Acceptance Criteria:**

- [x] Schemas validate the example JSON in TECH_RULES
- [x] `confidence_semantics` is mandatory whenever `confidence` is present
- [x] OpenAPI renders without errors

**Estimated Effort:** 60 min
**Assigned To:** Backend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T6-002: Load promoted models at startup and expose GET /models
**Description:** Startup hook builds a `ModelRegistry` service from `MODEL_REGISTRY_PATH`, validates schema compatibility, and exposes `GET /api/v1/models` (crop, task, family, version, status, validation macro-F1, experimental flag). Missing or corrupt artifacts yield `MODEL_UNAVAILABLE` for the affected crop only; no silent substitution.
**Dependencies:** T4-025, T4-009, T6-001
**Acceptance Criteria:**

- [x] Removing one artifact breaks only that crop
- [x] Incompatible feature list is reported as `SCHEMA_MISMATCH`
- [x] `/health` reflects registry state

**Estimated Effort:** 60 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T6-003: Implement DeviceManager (runtime GPU/CPU detection)
**Description:** `backend/app/core/device_manager.py`: detect CUDA at runtime, honour `FORCE_CPU`, expose device name and memory, and provide a context helper that retries on CUDA OOM with a smaller batch then CPU. Never report GPU use unless a tensor was actually placed on the GPU.
**Dependencies:** T1-009, T1-006
**Acceptance Criteria:**

- [x] FORCE_CPU=1 yields CPU
- [x] Simulated OOM retries as documented
- [x] `/health` shows device actually used

**Estimated Effort:** 45 min
**Assigned To:** Backend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T6-004: Implement SensorService.predict
**Description:** Validate payload, order features per the schema, apply the saved scaler, predict, compute confidence/uncertain flag from metadata thresholds, and attach `experimental` for tomato from metadata. Missing required values return `MISSING_FIELD`; nothing is imputed or zero-filled.
**Dependencies:** T6-002, T3-003
**Acceptance Criteria:**

- [x] Out-of-order JSON keys still map to the correct feature order
- [x] Missing field is rejected
- [x] Tomato result carries `experimental=true`

**Estimated Effort:** 60 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T6-005: Add POST /predict/sensor
**Description:** Route accepts `crop` and readings with units, calls SensorService, returns `PredictionResponse` with `limitations[]` including 'rule-derived labels'. Add the rate limit.
**Dependencies:** T6-004, T3-006
**Acceptance Criteria:**

- [ ] Valid request returns display label and limitations
- [ ] Invalid unit returns 422 `INVALID_INPUT`
- [ ] Response includes model version and timestamp

**Estimated Effort:** 45 min
**Assigned To:** Backend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T6-006: Implement the per-leaf ImageService pipeline
**Description:** Orchestrate: validated image, then by `input_mode`: `leaf` (the photo is one leaf; still get its leaf mask) or `plant` (SAM leaf extraction into individual leaves). For every valid leaf run the selected model (CNN, EfficientNetB0, or OpenCV features plus RF/SVM) and return a per-leaf class and confidence. Leaves that fail validation are returned in `rejected_leaves[]` with reasons; if no valid leaf exists return `SEGMENTATION_FAILED` (retryable), never a prediction. The plant summary is only a count of leaves and per-class counts; it never averages confidences. Attach preprocessing provenance.
**Dependencies:** T6-002, T5-010, T5-002, T3-002, T6-003, T5-033
**Acceptance Criteria:**

- [ ] Plant mode returns one result per valid leaf plus a count summary
- [ ] Rejected leaves are listed with reasons, not dropped
- [ ] No averaged or merged confidence appears in the response

**Estimated Effort:** 90 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T6-007: Add POST /predict/image
**Description:** Multipart route with `crop`, required `input_mode` (`leaf` or `plant`) and optional `model_family`; calls ImageService; maps typed errors to the envelope; sets `uncertain` per leaf when below the stored threshold; stricter rate limit. If the requested crop/family has no promoted model, return `MODEL_UNAVAILABLE` naming what is available.
**Dependencies:** T6-006, T3-006
**Acceptance Criteria:**

- [ ] Missing `input_mode` is rejected with 422
- [ ] Plant mode returns per-leaf results and a summary
- [ ] Unavailable family is reported clearly

**Estimated Effort:** 45 min
**Assigned To:** Backend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T6-008: Implement the rule-based recommendation provider
**Description:** `ml/recommendations/rules/<crop>.v1.yaml` and `RuleProvider`: map (crop, predicted class, confidence tier, modality) to conservative checks and next steps with an `evidence` list and `rules_version`. Low confidence returns re-measure/inspect advice; unsupported combinations return an explicit 'insufficient information'. No nutrient dosage or concentrations (Overview §20). ⚠️ UNCLEAR: whether a domain expert will review the wording; until then every rule is marked 'advisory, unreviewed'.
**Dependencies:** T6-001, T2-016
**Acceptance Criteria:**

- [x] Every class of every crop has a rule or an explicit 'insufficient information'
- [x] No rule text contains dosage numbers
- [x] Output cites which prediction/rule produced it

**Estimated Effort:** 90 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T6-009: Implement RecommendationService with provider interface
**Description:** Define `RecommendationProvider` protocol, wire `RuleProvider` as the default, and leave a Gemini provider slot (implemented in Phase 10). Output carries `source` and `rules_version`.
**Dependencies:** T6-008
**Acceptance Criteria:**

- [ ] Service works with no Gemini configured
- [ ] Output includes `source: rules`
- [ ] Provider can be swapped by config

**Estimated Effort:** 45 min
**Assigned To:** Backend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T6-010: Add history endpoints
**Description:** `POST /history`, `GET /history` (pagination, filters by crop/modality/source/date), `GET /history/{id}`. Stored timestamps are the observation time; responses expose `observed_at`, `observed_tz`, `source`, and a `historical: true` flag. DB failure returns `PERSISTENCE_FAILED` with an explicit 'not saved'.
**Dependencies:** T2-022, T2-023, T6-001
**Acceptance Criteria:**

- [x] Round trip preserves observed time and source
- [x] Limit is capped at 100
- [x] Simulated DB outage returns 'not saved'

**Estimated Effort:** 60 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T6-011: Add POST /analyze (both modalities, partial success)
**Description:** Single route running whichever modalities were supplied and returning each result separately plus a `not_assessed[]` list and the recommendation. If one modality fails, the other still returns. No combined severity score and no 'contradiction' field (Overview §19).
**Dependencies:** T6-004, T6-006, T6-009
**Acceptance Criteria:**

- [ ] Image failure does not block the sensor result
- [ ] `not_assessed` explains what was skipped
- [ ] Response has no merged score

**Estimated Effort:** 60 min
**Assigned To:** Backend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T6-012: Implement the DeviceAdapter interface, ManualAdapter, MockAdapter and GET /device/status
**Description:** Define the `DeviceAdapter` interface (latest sensor readings, latest camera frame, status) and implement `ManualAdapter` and `MockAdapter` (fixtures clearly flagged `mock`, used for development and CI). Validate payload schema, units, timestamp and freshness (`CAMPUS_MAX_AGE_S`), and report one of `missing / disconnected / invalid / stale / ok` per reading. Add `GET /api/v1/device/status`.
**Dependencies:** T3-003, T6-003, T1-016
**Acceptance Criteria:**

- [x] Stale payload is reported stale and never used
- [x] Mock data is never labelled live
- [x] Manual entry works with `DEVICE_ADAPTER=manual`

**Estimated Effort:** 60 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T6-013: Implement CampusServerAdapter, GET /device/sensors/latest and GET /device/camera/frame
**Description:** Build the real adapter from the probe ADR (0008): HTTP (or whichever protocol the probe found) client with timeouts and retries, mapping the server's field names and units to each crop's feature schema and reading the camera frame. The backend calls only the configured host from `.env`. Tomato readings are always flagged experimental unless a validated mapping to the transformed training features is documented. ⚠️ UNCLEAR: everything protocol-specific until T1-016 is complete; do not start before that ADR exists.
**Dependencies:** T6-012, T1-016, T6-004
**Acceptance Criteria:**

- [ ] Adapter output maps exactly to the model feature schema
- [ ] Unreachable server returns unavailable (not zeros) with the last-success time
- [ ] No hard-coded IP or credentials anywhere

**Estimated Effort:** 90 min
**Assigned To:** Backend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T6-014: Route campus camera frames into the image pipeline
**Description:** Let `/predict/image` and `/analyze` accept `use_campus_frame=true`: the backend fetches the latest frame itself (never from a client-supplied URL), validates it like an upload (size, type, decodability), rejects frames older than `CAMPUS_MAX_AGE_S`, processes it in `plant` mode (per-leaf results), and tags provenance as `live` with capture time and device id. Add a warning that camera images may differ from the training images in lighting, angle, and background.
**Dependencies:** T6-013, T6-007, T3-002
**Acceptance Criteria:**

- [ ] Stale frame is rejected with `DEVICE_UNAVAILABLE`
- [ ] Result provenance shows source `live` and capture time
- [ ] Warning about training-data mismatch is included in the response

**Estimated Effort:** 60 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T6-015: Add inference threadpool, timeouts, and concurrency limit
**Description:** Run blocking inference off the event loop with request timeouts and a semaphore limiting concurrent GPU work; return a structured 503/504 on timeout.
**Dependencies:** T6-007, T6-005
**Acceptance Criteria:**

- [ ] Event loop stays responsive during image inference
- [ ] Timeout returns the envelope
- [ ] Concurrency limit is configurable

**Estimated Effort:** 45 min
**Assigned To:** Backend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T6-016: Export and snapshot the OpenAPI contract
**Description:** Script writes `docs/api/openapi.json` and a snapshot test fixture; update TECH_RULES proposed routes with the real paths.
**Dependencies:** T6-011, T6-010, T6-013, T6-014, T6-002
**Acceptance Criteria:**

- [ ] Snapshot is committed
- [ ] TECH_RULES route list matches the implementation
- [ ] Breaking changes fail the contract test later

**Estimated Effort:** 30 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

---

## Phase 7: Core Frontend: UI & Interactions
*Goal: All Must Have screens (S1-S7) and states from DESIGN.md, with uncertainty and limitations as first-class UI.*

#### T7-001: Implement design tokens in Tailwind
**Description:** Encode DESIGN.md colour, typography, spacing, radius, and shadow tokens in the Tailwind config / CSS variables; build `Button`, `Input`, `Card`, `Badge`, `Toast` base components with all listed states.
**Dependencies:** T1-010
**Acceptance Criteria:**

- [x] Token names match DESIGN.md
- [x] Each component shows default/hover/focus/disabled/loading states in a demo page
- [x] `prefers-reduced-motion` respected

**Estimated Effort:** 45 min
**Assigned To:** Frontend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T7-002: Verify contrast for every token pair (WCAG AA)
**Description:** Check all text/background and UI-component token pairs with a contrast tool; adjust failing hex values and update DESIGN.md. The ratios listed in DESIGN.md were computed with the WCAG formula by script for the listed pairs; confirm them with a contrast tool and check any pair not listed (for example text on the `--sand` panels and focus rings).
**Dependencies:** T7-001
**Acceptance Criteria:**

- [x] Text pairs are at least 4.5:1 and UI components at least 3:1
- [x] Updated hex values are reflected in DESIGN.md
- [x] Results table is saved in `docs/design/contrast.md`

**Estimated Effort:** 30 min
**Assigned To:** Frontend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T7-003: Build the MotionProvider and shared animation variants
**Description:** Create `MotionProvider` (reads `prefers-reduced-motion`, pauses decorative loops when the tab is hidden) and shared variants `fadeUp`, `stagger`, `grow` with the durations from DESIGN.md's Motion System table. All components import durations from tokens, never hard-coded values.
**Dependencies:** T7-001
**Acceptance Criteria:**

- [x] Reduced-motion setting turns movement into a fade or nothing
- [x] Only `transform`, `opacity`, and SVG stroke are animated
- [x] Variants match the DESIGN.md durations

**Estimated Effort:** 60 min
**Assigned To:** Frontend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T7-004: Build the layout shell (nav, footer disclaimer, health banner)
**Description:** Top bar (Analyze, History, Models & Limits, About), footer with the advisory disclaimer and version, and a non-blocking banner when `/health` fails.
**Dependencies:** T7-001, T7-003, T1-009
**Acceptance Criteria:**

- [x] Keyboard navigation reaches all links
- [x] Banner appears when the API is down
- [x] Disclaimer text avoids 'diagnosis' wording

**Estimated Effort:** 45 min
**Assigned To:** Frontend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T7-005: Create the typed API client
**Description:** `frontend/lib/api/` with types generated from `docs/api/openapi.json` (or hand-mirrored), a fetch wrapper that parses the error envelope into typed errors, and timeouts.
**Dependencies:** T6-016, T1-010
**Acceptance Criteria:**

- [ ] Error envelope becomes a typed object
- [ ] Types match the OpenAPI snapshot
- [ ] Network failure is distinguishable from API error

**Estimated Effort:** 45 min
**Assigned To:** Frontend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T7-006: Build S1 Home with the animated plant hero and crop selection
**Description:** Line-art plant hero that draws itself once (SVG stroke animation), `CropCard` x3 with line-art, supported-input chips and the tomato 'sensor mode: experimental' chip, staggered entrance, advisory banner, links.
**Dependencies:** T7-004, T7-003
**Acceptance Criteria:**

- [x] Selecting a card routes to `/analyze?crop=`
- [x] Empty/error/loading behaviour matches DESIGN S1
- [x] Layout follows the 3/2+1/1 column rule

**Estimated Effort:** 60 min
**Assigned To:** Frontend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T7-007: Build ImageDropzone with client-side checks
**Description:** Drag/drop and keyboard-operable file picker with preview, remove, accepted formats and size shown, an `InputModeToggle` (Leaf photo / Whole plant) that is always explicit, and an explainer drawer for what counts as a leaf photo versus a whole-plant photo. Client checks are UX only; server validation remains authoritative.
**Dependencies:** T7-001
**Acceptance Criteria:**

- [x] Wrong file type shows a specific message
- [x] Works with keyboard only
- [x] Mobile camera capture attribute is set

**Estimated Effort:** 60 min
**Assigned To:** Frontend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T7-008: Build SensorForm from per-crop schemas
**Description:** Fields, units, and normal-band hints per crop; empty fields are 'missing' (never coerced to 0); inline validation messages naming field and unit; provenance toggle (manual / device).
**Dependencies:** T7-001, T7-005
**Acceptance Criteria:**

- [ ] Empty field is sent as missing, not 0
- [ ] Field order and names match the backend schema
- [ ] Error messages name the unit

**Estimated Effort:** 60 min
**Assigned To:** Frontend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T7-009: Build the Analyze page flow and progress states
**Description:** Mode selector, source tabs (Manual / Upload / Campus live) with `CampusCameraPicker` and `CampusSensorCard`, 'what will be used' summary panel (sticky on desktop, accordion on mobile), submit to `/analyze`, a growing-vine step indicator (Validating, Segmenting, Classifying, Recommending), cancel, and device-unavailable note with manual fallback.
**Dependencies:** T7-007, T7-008, T6-011, T6-014, T7-003
**Acceptance Criteria:**

- [ ] Analyze is disabled until required inputs validate
- [ ] Cancel aborts the request
- [ ] Device-unavailable message offers manual entry

**Estimated Effort:** 60 min
**Assigned To:** Frontend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T7-010: Build S8 Live Monitor
**Description:** `/live`: connection status, live sensor grid (value, unit, observed time, freshness ring), camera panel with capture time, 'Analyze these readings / this frame' buttons that open `/analyze` with source = live, auto-refresh at `CAMPUS_POLL_INTERVAL_S` (paused when the tab is hidden), and the tomato limitation note. Stale or unavailable data is greyed and labelled, never shown as current.
**Dependencies:** T7-005, T7-003, T6-013
**Acceptance Criteria:**

- [ ] Server unreachable shows last-success time and Retry
- [ ] Stale values are labelled stale
- [ ] New data triggers a pulse and an aria-live announcement; reduced motion shows text only

**Estimated Effort:** 90 min
**Assigned To:** Frontend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T7-011: Build ConfidenceMeter and UncertaintyBadge
**Description:** Three tiers (higher / moderate / uncertain) using icon + text + pattern, never colour alone; tooltip text visible on touch; wording says 'model score, not a calibrated probability'.
**Dependencies:** T7-001
**Acceptance Criteria:**

- [ ] Uncertain tier is visually lighter than higher-confidence tier
- [ ] No tier relies on colour alone
- [ ] Tier thresholds come from the API, not constants

**Estimated Effort:** 45 min
**Assigned To:** Frontend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T7-012: Build ImageResultPanel with per-leaf cards
**Description:** Plant photo with a numbered leaf overlay, a `LeafGrid` of `LeafCard`s (display label, original class in details, its own `ConfidenceMeter` and uncertain flag), a plant summary showing leaf count and counts per predicted class (no averaged score), warnings, preprocessing chips, mask overlay toggle with alt text, selecting a card highlights its leaf, a 'not assessed' list for rejected leaves, and a segmentation-failure card with Retry and Choose-different-image.
**Dependencies:** T7-011, T7-005
**Acceptance Criteria:**

- [ ] Each leaf shows its own confidence tier
- [ ] Rejected leaves are listed as not assessed with reasons
- [ ] Segmentation failure is shown as an input failure, not a disease

**Estimated Effort:** 90 min
**Assigned To:** Frontend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T7-013: Build SensorResultPanel
**Description:** Predicted class, evidence rows (reading versus reference band), `experimental` badge with limitation banner for tomato, and 'rule-derived label' note.
**Dependencies:** T7-011, T7-005
**Acceptance Criteria:**

- [ ] Tomato shows the experimental banner
- [ ] Evidence rows show reading, unit, and reference band
- [ ] Label is described as rule-derived

**Estimated Effort:** 45 min
**Assigned To:** Frontend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T7-014: Build RecommendationCard and NotAssessedNote
**Description:** Summary, evidence list, suggested checks, source chip (Gemini or Rules), uncertainty note, and a 'Not assessed' note. No merged severity and no contradiction alerts.
**Dependencies:** T7-012, T7-013
**Acceptance Criteria:**

- [ ] Source chip reflects `source` from the API
- [ ] 'Not assessed' appears when one modality is missing
- [ ] No combined score is shown anywhere

**Estimated Effort:** 45 min
**Assigned To:** Frontend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T7-015: Add Save record with honest outcomes
**Description:** Save button posting to `/history`, disabled while saving, success toast, and a clear 'Not saved' error state on `PERSISTENCE_FAILED`.
**Dependencies:** T7-014, T6-010
**Acceptance Criteria:**

- [ ] Failed save never shows success
- [ ] Double-click does not create two records
- [ ] Toast announces via aria-live

**Estimated Effort:** 30 min
**Assigned To:** Frontend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T7-016: Build S4 History list
**Description:** Filter bar (crop, modality, source, date), table with timestamp+timezone, source chip, class, confidence tier, model version; pagination; stacked cards under 768 px; 'Historical' tag on every row.
**Dependencies:** T7-005, T7-001, T6-010
**Acceptance Criteria:**

- [ ] Source chip 'Live' appears only for records flagged live
- [ ] Empty/error/loading states match DESIGN S4
- [ ] Table scrolls inside its own container on small screens

**Estimated Effort:** 60 min
**Assigned To:** Frontend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T7-017: Build S5 Record detail
**Description:** Provenance card, inputs, predictions, model and preprocessing/rule versions, recommendation and evidence, limitations, copy-JSON; handle missing original image.
**Dependencies:** T7-016
**Acceptance Criteria:**

- [ ] Missing image shows metadata only
- [ ] Timestamps display with timezone
- [ ] Not-found state works

**Estimated Effort:** 45 min
**Assigned To:** Frontend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T7-018: Build S6 Models & Limitations
**Description:** Tabs (Models, Datasets, Limitations) using `/models`: registry table, metrics drawer (confusion matrix, class support, split protocol), data-type legend, and the Overview §28 limitations list with the statement that scores mean agreement with project rules on a documented split.
**Dependencies:** T7-005, T7-001, T6-002
**Acceptance Criteria:**

- [ ] Experimental models are labelled
- [ ] Limitations list matches Overview §28
- [ ] Empty state explains no promoted models

**Estimated Effort:** 60 min
**Assigned To:** Frontend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T7-019: Build S7 About / How it works
**Description:** Plain-language stepper and glossary (confidence, macro-F1, rule-derived label, segmentation, uncertainty).
**Dependencies:** T7-004
**Acceptance Criteria:**

- [ ] Glossary entries are consistent with concept notes
- [ ] Page is keyboard accessible
- [ ] No claims of diagnosis or guaranteed accuracy

**Estimated Effort:** 30 min
**Assigned To:** Frontend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T7-020: Audit empty/error/loading and responsive behaviour on all screens
**Description:** Walk through S1-S8 at 360/768/1024/1280 px and verify every state defined in DESIGN.md exists and behaves as specified; fix gaps.
**Dependencies:** T7-006, T7-009, T7-010, T7-015, T7-017, T7-018, T7-019
**Acceptance Criteria:**

- [ ] Checklist per screen/state/breakpoint saved in `docs/design/state_audit.md`
- [ ] No horizontal body scroll
- [ ] Touch targets are at least 44 px

**Estimated Effort:** 60 min
**Assigned To:** Frontend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T7-021: Accessibility pass (keyboard + automated + screen reader spot-check)
**Description:** Run an automated accessibility scan on each route, complete keyboard-only walkthroughs of the main flow, and spot-check with one screen reader. Fix issues; log leftovers.
**Dependencies:** T7-020, T7-002
**Acceptance Criteria:**

- [ ] Zero critical automated violations on all routes
- [ ] Main flow works with keyboard only
- [ ] Remaining issues are logged

**Estimated Effort:** 60 min
**Assigned To:** Frontend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

---

## Phase 8: Testing
*Goal: Unit, integration, contract, data/leakage, and E2E coverage for all MVP features (Overview §27). Report only tests that were actually run.*

#### T8-001: Unit tests: labels, schemas, validators
**Description:** Tests for class-name normalisation and display mapping (all 21 image classes), sensor feature order, unit/range validation, missing vs zero handling, unknown-field rejection, and error-envelope codes.
**Dependencies:** T2-016, T3-003, T3-004
**Acceptance Criteria:**

- [x] Every Overview §6 class is covered
- [x] Missing vs 0 behaviour is asserted
- [x] Tests run in CI

**Estimated Effort:** 60 min
**Assigned To:** QA
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T8-002: Unit tests: OpenCV feature formulas and edge cases
**Description:** Synthetic images with known answers for ExG, ExR, yellow/brown ratios, compactness (including zero perimeter), relative leaf area, solidity, aspect ratio, and `droop_angle` if retained; BGR/RGB channel-order test; NaN/Inf rejection.
**Dependencies:** T5-004, T5-005, T5-007
**Acceptance Criteria:**

- [ ] Each feature has at least one hand-computed expected value
- [ ] Channel-order bug is caught by a test
- [ ] Degenerate inputs raise typed errors

**Estimated Effort:** 90 min
**Assigned To:** QA
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T8-003: Unit tests: GLCM config, mask validation and leaf extraction
**Description:** Verify quantisation levels, distance, angles, averaging, entropy log base, constant image behaviour; mask validator cases (empty, tiny, full-image, border-dominated, multiple components); leaf extraction on synthetic images with 0, 1, and several leaves, overlapping masks, and the `rejected[]` list.
**Dependencies:** T5-003, T5-009, T5-010
**Acceptance Criteria:**

- [x] GLCM config is asserted not just run
- [x] All mask verdict branches covered
- [x] Tests are deterministic

**Estimated Effort:** 60 min
**Assigned To:** QA
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T8-004: Unit tests: thresholds, rules, registry, metrics, CV
**Description:** Confidence tier logic, rule-recommendation fallbacks (including insufficient information), registry compatibility/promote/rollback, metrics against hand-computed values, CV group-overlap guard, label-rule oracle boundaries and multi-rule priority.
**Dependencies:** T5-031, T6-008, T4-009, T4-005, T4-007, T4-016
**Acceptance Criteria:**

- [ ] Boundary values for each documented threshold are tested
- [ ] Promote/rollback tested
- [ ] Group overlap in CV raises

**Estimated Effort:** 60 min
**Assigned To:** QA
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T8-005: Integration tests: image and sensor prediction paths
**Description:** Using small fixtures and a tiny stub model: image upload to segmentation to prediction; leaf crop to features to prediction; sensor payload to prediction; segmentation failure returns retryable error; a request without `input_mode` is rejected and plant mode returns per-leaf results.
**Dependencies:** T6-007, T6-005
**Acceptance Criteria:**

- [ ] Each path has a success and a failure test
- [ ] Stub models are clearly marked as test fixtures
- [ ] No real datasets required in CI

**Estimated Effort:** 90 min
**Assigned To:** QA
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T8-006: Integration tests: fallbacks and failure modes
**Description:** Corrupt/missing artifact, DB outage on save ('not saved'), device unavailable/stale to manual path, GPU unavailable (FORCE_CPU) and simulated OOM, and (when implemented) Gemini unavailable to rules fallback.
**Dependencies:** T6-015, T6-012, T6-010
**Acceptance Criteria:**

- [ ] Each failure returns the documented code
- [ ] DB outage never reports success
- [ ] Gemini case is skipped with a clear marker until Phase 10

**Estimated Effort:** 60 min
**Assigned To:** QA
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T8-007: Integration tests: history round trip and pagination
**Description:** Save and retrieve with observed time/timezone/source preserved, filter and pagination limits, 'historical' flag, and that old records are never labelled live.
**Dependencies:** T6-010
**Acceptance Criteria:**

- [x] Timezone survives round trip
- [x] Limit above 100 is capped
- [x] Old record is not marked live

**Estimated Effort:** 45 min
**Assigned To:** QA
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T8-008: Contract tests against the OpenAPI snapshot
**Description:** Fail when response shapes or required fields change without updating the snapshot.
**Dependencies:** T6-016
**Acceptance Criteria:**

- [ ] Intentional field removal fails the test
- [ ] Snapshot update is an explicit commit
- [ ] Runs in CI

**Estimated Effort:** 45 min
**Assigned To:** QA
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T8-009: Data and leakage tests
**Description:** Class counts before/after preprocessing, no source-image overlap across splits, sibling lettuce crops share a split, sensor group/time separation holds, prohibited fields absent from features, target not in features, split hashes unchanged, no NaN/Inf reaching a model, and feature column order equals artifact metadata.
**Dependencies:** T2-019, T5-013, T4-025, T5-033
**Acceptance Criteria:**

- [ ] Each Overview §27.3 bullet maps to at least one test
- [ ] Tests skip with a clear message if data is absent locally
- [ ] Tests run in CI against tiny fixtures

**Estimated Effort:** 90 min
**Assigned To:** QA
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T8-010: E2E tests for critical flows
**Description:** Playwright: crop to sensor entry to result to save to history; campus live reading (mock adapter) to analyze; plant photo upload showing per-leaf cards; image upload with segmentation failure to retry; device unavailable to manual entry; low-confidence result shows the uncertain tier.
**Dependencies:** T7-021, T7-015, T7-009
**Acceptance Criteria:**

- [ ] Four flows pass headless
- [ ] Tests use the stub model and mock adapter
- [ ] Failures capture screenshots

**Estimated Effort:** 90 min
**Assigned To:** QA
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T8-011: Add coverage gate and test report
**Description:** Configure `pytest --cov` with a planned 70 percent line target on `backend/` and `ml/`, publish the summary in CI, and document unmeasured areas.
**Dependencies:** T8-001, T8-004, T8-005
**Acceptance Criteria:**

- [ ] Coverage is measured, not assumed
- [ ] Gate value recorded in CONTRIBUTING
- [ ] Gaps listed in `docs/testing.md`

**Estimated Effort:** 30 min
**Assigned To:** QA
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

---

## Phase 9: Deployment & DevOps
*Goal: Local-first containers, CI/CD, monitoring, backups, and a first demo release. No cloud services are assumed.*

#### T9-001: Write Dockerfiles for backend and frontend
**Description:** Multi-stage builds, non-root user, pinned base images, model/data directories mounted as volumes (never baked in), CPU build by default with documented GPU variant.
**Dependencies:** T8-011, T1-010, T1-005
**Acceptance Criteria:**

- [ ] Images build from clean checkout
- [ ] No secrets or datasets in image layers
- [ ] Container runs as non-root

**Estimated Effort:** 60 min
**Assigned To:** DevOps
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T9-002: Extend docker-compose for the full stack
**Description:** Add backend and frontend services with env files, healthchecks, and volumes for `models/` and `uploads/`; run migrations on start.
**Dependencies:** T9-001, T1-011
**Acceptance Criteria:**

- [ ] `docker compose up` serves the app end to end on a fresh clone
- [ ] Healthchecks gate dependent services
- [ ] Volumes persist models and uploads

**Estimated Effort:** 45 min
**Assigned To:** DevOps
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T9-003: Add database backup and restore scripts
**Description:** `scripts/backup_db.sh` and `restore_db.sh` using `pg_dump`/`psql`, with retention notes and a restore test.
**Dependencies:** T9-002
**Acceptance Criteria:**

- [ ] Restore recreates history rows
- [ ] Scripts read credentials from env
- [ ] Procedure documented

**Estimated Effort:** 30 min
**Assigned To:** DevOps
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T9-004: Add health details and fallback counters
**Description:** Extend `/health` with DB, registry, device, and GPU status and add counters for Gemini-to-rules, GPU-to-CPU, and segmentation-failure events exposed in logs or a metrics endpoint.
**Dependencies:** T6-015, T6-002
**Acceptance Criteria:**

- [ ] Counters increment in a simulated failure test
- [ ] Health reflects a stopped DB
- [ ] No sensitive data in output

**Estimated Effort:** 45 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T9-005: Add the reverse-proxy sample for TLS
**Description:** Provide `deploy/Caddyfile` (or nginx conf) fronting both services, tested locally with a self-signed certificate, matching `docs/deployment/https.md`.
**Dependencies:** T9-002, T3-008
**Acceptance Criteria:**

- [ ] Frontend and API reachable over HTTPS locally
- [ ] HTTP redirects to HTTPS
- [ ] Works with the CORS allow-list

**Estimated Effort:** 45 min
**Assigned To:** DevOps
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T9-006: Write the demo runbook and cut v0.1.0
**Description:** `docs/runbook.md`: start/stop, loading models, pointing the app at the campus server (IP in `.env`) and checking connectivity, environment variables, known limitations to state during a demo, fallback behaviour; tag `v0.1.0` after the checklist (tests run, reports committed, README accurate).
**Dependencies:** T9-005, T9-003, T9-004, T8-010
**Acceptance Criteria:**

- [ ] A teammate can run the demo from the runbook alone
- [ ] README claims match verified features
- [ ] Tag created only after the checklist passes

**Estimated Effort:** 45 min
**Assigned To:** DevOps
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

---

## Phase 10: Should Have Features
*Goal: Non-critical features from the PRD Should Have list plus extended model coverage. Start only after the MVP release (T9-006).*

#### T10-001: Implement GeminiProvider with strict output schema
**Description:** Prompt includes only structured evidence (no keys, no personal data), requests JSON that matches the recommendation schema, enforces a timeout and bounded retries, and uses the model name from `GEMINI_MODEL`.
**Dependencies:** T6-009, T1-008, T9-006
**Acceptance Criteria:**

- [ ] Output failing schema validation triggers fallback
- [ ] No secret appears in prompts or logs
- [ ] Provider selectable by config

**Estimated Effort:** 60 min
**Assigned To:** Backend
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T10-002: Sanitise and verify Gemini output
**Description:** Treat output as untrusted: reject URLs, shell-like strings, code blocks, and any dosage/concentration numbers; cap length; require that each claim references provided evidence; mark `source: gemini` only after validation, else use rules.
**Dependencies:** T10-001
**Acceptance Criteria:**

- [ ] Injected 'ignore instructions' text in an input does not change behaviour
- [ ] Dosage-like output is rejected
- [ ] Rejections increment the fallback counter

**Estimated Effort:** 60 min
**Assigned To:** Backend
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T10-003: Probability calibration experiment (validation-fit)
**Description:** Fit temperature scaling or Platt/isotonic calibration on validation data for one image model and one sensor model, compare reliability diagrams, and decide whether to adopt. Only change the UI wording from 'model score' if calibration is shown to help.
**Dependencies:** T5-031
**Acceptance Criteria:**

- [ ] Calibration is fit on validation only
- [ ] Before/after reliability plots saved
- [ ] Decision recorded in an ADR

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit and writes the concept note.
**Learning Focus:** Probability calibration (temperature scaling, reliability diagrams) → `docs/learning/probability-calibration-temperature-scaling-reliability-diagrams.md`

#### T10-004: Cross-crop model comparison report
**Description:** Write `reports/model_comparison.md` summarising all families per crop with validation and test results clearly separated, limitations, and the coverage matrix of what was and was not trained.
**Dependencies:** T9-006, T5-030
**Acceptance Criteria:**

- [ ] Validation and test results are in separate tables
- [ ] Untrained crop/family combinations are listed
- [ ] Every claim states dataset, split, and metric

**Estimated Effort:** 60 min
**Assigned To:** ML
**Driver / Navigator:** Yashwant drives / Sushil reviews the commit.

#### T10-005: Add history CSV export
**Description:** `GET /history/export` and a UI button exporting records with timestamps, source, and model versions; escape CSV injection characters.
**Dependencies:** T7-016, T6-010
**Acceptance Criteria:**

- [ ] Cells starting with = + - @ are neutralised
- [ ] Export respects filters
- [ ] Timestamps include timezone

**Estimated Effort:** 45 min
**Assigned To:** Full-stack
**Driver / Navigator:** Sushil drives / Yashwant reviews the commit.

#### T10-006: Write the final project report and demo script
**Description:** Compile methodology, datasets and provenance, evaluation (with limitations), architecture, and a learning appendix showing both members' concept notes. State every result with its data source, split, and metric (Overview §30.22).
**Dependencies:** T10-004
**Acceptance Criteria:**

- [ ] Every number has source, split, and metric
- [ ] Limitations section mirrors Overview §28
- [ ] Learning appendix links all concept notes

**Estimated Effort:** 90 min
**Assigned To:** QA
**Driver / Navigator:** Both. Pair session; each member presents half and quizzes the other.

#### T10-007: Final concept quiz and retrospective
**Description:** Each member writes 10 questions for the other covering all concepts, answers them closed-book, and records scores and follow-ups in `docs/learning/PROGRESS.md`; hold a short retrospective on balance of driving and understanding.
**Dependencies:** T10-006
**Acceptance Criteria:**

- [ ] Both members have a recorded score
- [ ] Weak concepts have a follow-up plan
- [ ] Driver balance table from the todo is reviewed

**Estimated Effort:** 60 min
**Assigned To:** QA
**Driver / Navigator:** Both. Pair session; each member presents half and quizzes the other.

---

## Appendix

### Critical Path
> Tasks that block the most downstream work; complete these first.

1. T1-001 (Inventory the data and scripts on disk (read-only))
2. T1-002 (Create or adapt the monorepo folder structure)
3. T1-005 (Verify and pin Python / ML dependencies)
4. T2-005 (Audit the lettuce sensor CSVs against the script and README)
5. T4-002 (Record the sensor validation protocol (ADR))
6. T4-010 (Build the sensor training pipeline skeleton)
7. T4-015 (Implement hyperparameter search with grouped CV on macro-F1)
8. T4-023 (Implement model selection by validation macro-F1)
9. T4-024 (Implement the run-once final test evaluation)
10. T4-025 (Promote the selected sensor models)
11. T6-002 (Load promoted models at startup and expose GET /models)
12. T6-004 (Implement SensorService.predict)
13. T6-011 (Add POST /analyze (both modalities, partial success))
14. T6-016 (Export and snapshot the OpenAPI contract)
15. T7-005 (Create the typed API client)
16. T7-009 (Build the Analyze page flow and progress states)
17. T8-010 (E2E tests for critical flows)
18. T9-006 (Write the demo runbook and cut v0.1.0)

**Second chain (image path), which gates `image_svc`:**

1. T1-007 (SAM 2.1 feasibility spike (install, load, time, memory))
2. T5-008 (Implement the SAM 2.1 segmenter wrapper)
3. T5-009 (Implement per-leaf mask validation)
4. T2-015 (Create or preserve the 70/15/15 image split manifests)
5. T5-013 (Build feature datasets from split manifests)
6. T5-020 (Implement the generic PyTorch training loop)
7. T5-028 (Compare lettuce image models fairly and select one)
8. T5-031 (Derive per-leaf uncertainty thresholds from validation data)
9. T5-032 (Run-once final test evaluation for image models)
10. T5-033 (Promote image and feature models to the registry)
11. T6-006 (Implement the per-leaf ImageService pipeline)

### Parallelizable Tasks
> Tasks with no shared dependencies that can run simultaneously. Remember: both members still attend every concept session.

- T2-002, T2-003, T2-004 (provenance sheets) can run in parallel after T2-001.
- T2-005 and T2-007 can run in parallel; T2-006 follows the lettuce audit because it reuses the audit script.
- Phase 3 tasks (T3-002, T3-004, T3-005, T3-008) can run alongside Phase 4 ML work.
- T1-007 and T1-016 (needs the campus network) can run alongside Phase 2-4 work, which de-risks Phases 5-6 early.
- Frontend foundations (T7-001, T7-002, T7-004, T7-007, T7-011) depend only on the scaffold and can run alongside Phases 4-5.
- T4-011 to T4-014 are sequential by design (shared pipeline) but T4-017, T4-018, T4-019 can run in parallel once T4-015 and T4-016 are done.
- T5-015 and T5-016 can run in parallel after T5-014.
- T8-001, T8-002, T8-003, T8-004 can all run in parallel.

### Risk Register

| Risk | Affected Tasks | Likelihood | Mitigation |
|---|---|---|---|
| Preparation scripts/READMEs disagree with the overview or with the CSVs (thresholds, priorities, split rules) | T1-001, T2-005, T2-006, T2-007 | Medium | Audit against the scripts; record every discrepancy; mark anything not stated as 'not stated' and ask; keep tomato sensor results experimental. |
| Campus server protocol unknown or unreachable off campus; camera images differ from training images | T1-016, T6-013, T6-014, T7-010 | High | Probe early on the campus network; build against the mock adapter meanwhile; manual fallback always works; show a training-data-mismatch warning on camera frames. |
| Tomato is the live crop, but live readings may lack N/P/K and differ from the proxy-transformed, partly synthetic training data | T4-019, T4-020, T6-013 | High | Probe field list first; train a live-compatible subset model; keep tomato experimental; show limitations in the UI. |
| Lettuce leaf crops inherit the whole-plant label (weak labels) and leaf separation fails on touching leaves | T5-012, T5-010, T5-011 | High | Manual label review of crops, per-leaf confidence shown with limitation note, rejected leaves reported, validator tuned on the mask review. |
| Validation hold-out leaves rare classes (e.g. pH_Low) missing from a partition | T2-010, T2-011, T2-012 | Medium | Report class coverage per partition; ask before dropping or merging classes. |
| SAM 2.1 too heavy for RTX 3050 / CPU-only campus machine | T1-007, T5-008, T6-006 | Medium | Measure early; smaller checkpoint; explicit team decision on any alternative (never a silent substitute). |
| No validation sets existed, so model selection could drift onto the test set | T4-002, T4-007, T4-023 | High | Grouped hold-out validation sets (Phase 2), grouped CV with oversampling inside folds, hash guard, run-once final evaluation log. |
| High scores on rule-derived labels are read as real-world accuracy | T4-021, T7-013, T7-018, T10-006 | High | Limitations block in every report and UI; teach-back on why trees recover rules. |
| Duplicate/near-duplicate or sibling-crop leakage across image splits | T2-014, T5-012, T2-015, T8-009 | Medium | Duplicate report, source-image grouping, data tests. |
| `droop_angle` is not defensible, forcing regeneration of tomato features and retraining | T5-006, T5-007, T5-016 | Medium | Decide early; schema versioning; dependent models marked incompatible until retrained. |
| Tomato live-sensor inference is incompatible with transformed training features | T4-019, T6-004, T6-013 | High | Keep `experimental=true`; document mapping before removing the flag. |
| Unequal learning (one member ends up doing most ML) | T4-001, T4-026, T5-034, T10-007 | Medium | Strict alternation, concept notes by navigator, teach-backs, balance table review. |
| Dataset licence/redistribution terms unclear | T2-002, T2-003, T2-004 | Medium | Record 'unverified' honestly; do not commit datasets; check before sharing artifacts. |
| Gemini unavailable, rate-limited, or returns unsafe text | T10-001, T10-002 | Medium | Rule provider is the default; schema validation; sanitiser; fallback counters. |
| Scope exceeds the (unknown) deadline: all 3 crops x all model families is large | Phases 4-7 | High | Do lettuce end to end first, then spinach, then tomato; Phase 10 is optional; cut optional ablations before cutting evaluation rigor. |
| GPU/runtime mismatch between teammates' machines | T1-005, T1-006, T6-003 | Medium | Pinned envs, `check_env.py` outputs recorded, CPU fallback tested. |
| Circular dependencies in the plan | All | Low | Dependency graph was validated to be acyclic by the generator (every dependency points to an earlier task). |

### Recommended Build Order
> Optimal sequence for two people working as a pair.

1. Complete Phase 1 (T1-001 to T1-016); do T1-007 and T1-016 early.
2. Phase 2: provenance and audits (T2-001 to T2-007), then image audits and splits (T2-013 to T2-015), then schemas, guards, manifests (T2-016 to T2-020), then the database (T2-021 to T2-023).
3. Phase 3 security baseline (can overlap Phase 4).
4. Phase 4 in order: leakage concept, protocol, from-scratch notebooks, evaluation tooling, registry, trainers, tuning, per-crop runs, selection, final evaluation, promotion, teach-back (T4-026).
5. Phase 5 in order: colour/GLCM/feature extractors, `droop_angle` decision, segmentation, lettuce crops, feature datasets, classical classifiers, CNN concepts, data loaders, CNN and EfficientNetB0 for lettuce then spinach then tomato, comparisons, thresholds, final evaluation, promotion, teach-back (T5-034).
6. Phase 6 backend (sensor path first, then image path, then recommendations, history, `/analyze`).
7. Phase 7 frontend (tokens and shell can start earlier in parallel).
8. Phase 8 testing (write unit tests alongside earlier phases where possible; this phase closes gaps and adds E2E).
9. Phase 9 deployment and the `v0.1.0` release (T9-006).
10. Phase 10 extensions in order of value: Gemini provider and sanitiser, calibration, other crops' deep models, report, quiz.

### Driver balance (equal-effort check)

| Phase | Yashwant drives | Sushil drives | Joint tasks |
|---|---|---|---|
| 1 | 8 | 8 | 0 |
| 2 | 12 | 11 | 0 |
| 3 | 4 | 4 | 0 |
| 4 | 12 | 13 | 1 |
| 5 | 16 | 17 | 1 |
| 6 | 8 | 8 | 0 |
| 7 | 11 | 10 | 0 |
| 8 | 5 | 6 | 0 |
| 9 | 3 | 3 | 0 |
| 10 | 3 | 2 | 2 |
| **Total** | **82** | **82** | **4** |

Estimated driving effort: Yashwant 77.0 h, Sushil 78.0 h (plus joint sessions 5.5 h). These are planning estimates, not measurements; re-balance if actual time diverges.

By role (drives):

| Role | Yashwant | Sushil |
|---|---|---|
| Backend | 16 | 18 |
| DevOps | 8 | 4 |
| Frontend | 12 | 10 |
| Full-stack | 4 | 5 |
| ML | 37 | 39 |
| QA | 5 | 6 |

### Concept coverage matrix
> Navigator writes the note; the driver implements. Both complete the teach-back and self-check.

| Concept | Task | Drives | Writes note |
|---|---|---|---|
| Foundation-model segmentation: prompts, masks, and failure modes (SAM 2.1) | T1-007 | Sushil | Yashwant |
| Train/test splits, oversampling, and day-based (grouped) separation | T2-005 | Yashwant | Sushil |
| Hold-out validation design for time-correlated data | T2-010 | Sushil | Yashwant |
| Near-duplicate detection and why duplicates across splits inflate image accuracy | T2-014 | Yashwant | Sushil |
| Data leakage and grouped / time-based splits | T4-001 | Sushil | Yashwant |
| Decision trees: Gini impurity and greedy splits | T4-003 | Yashwant | Sushil |
| Confusion matrix, precision/recall, macro-F1 | T4-004 | Sushil | Yashwant |
| Cross-validation and class imbalance (oversampling inside folds) | T4-007 | Yashwant | Sushil |
| Decision tree hyperparameters, depth, and overfitting | T4-011 | Sushil | Yashwant |
| Random Forest: bagging, feature subsampling, variance reduction | T4-012 | Yashwant | Sushil |
| SVM: margin, C, RBF kernel and gamma, why scaling matters | T4-013 | Sushil | Yashwant |
| KNN and logistic regression baselines (distance, regularisation) | T4-014 | Yashwant | Sushil |
| Hyperparameter search with grouped CV and its pitfalls | T4-015 | Sushil | Yashwant |
| Train/serve skew and feature availability (live-compatible models) | T4-020 | Yashwant | Sushil |
| Why rule-derived labels inflate scores | T4-021 | Sushil | Yashwant |
| Colour spaces (BGR/RGB/HSV) and vegetation indices (ExG, ExR) | T5-001 | Yashwant | Sushil |
| Texture features: GLCM contrast and entropy | T5-003 | Sushil | Yashwant |
| Shape descriptors: compactness, solidity, aspect ratio | T5-007 | Yashwant | Sushil |
| Instance segmentation for per-leaf prediction (SAM automatic masks) | T5-010 | Sushil | Yashwant |
| Weak labels and instance-level (per-leaf) prediction | T5-012 | Yashwant | Sushil |
| Convolution, pooling, and backpropagation | T5-017 | Sushil | Yashwant |
| Data augmentation (training only) and input normalisation | T5-018 | Yashwant | Sushil |
| Optimisers, learning rate, and early stopping on validation macro-F1 | T5-020 | Sushil | Yashwant |
| Transfer learning and fine-tuning (EfficientNetB0) | T5-022 | Yashwant | Sushil |
| Softmax confidence, calibration, and uncertainty thresholds | T5-031 | Sushil | Yashwant |
| Probability calibration (temperature scaling, reliability diagrams) | T10-003 | Yashwant | Sushil |

Concept notes written: Yashwant 13, Sushil 13.

### Open questions (collected from tasks)
> Answer these before starting the listed task. Each `⚠️ UNCLEAR:` in a task body is repeated here.

- **T1-005** (Verify and pin Python / ML dependencies): Overview §25 lists both Windows 11 and Ubuntu; confirm which OS each teammate uses.
- **T1-007** (SAM 2.1 feasibility spike (install, load, time, memory)): the overview does not say whether to use automatic masks or point/box prompts for leaf separation; test both and report.
- **T1-016** (Probe the campus server (sensors + camera) and document the interface): protocol, credentials, and whether the server is reachable outside the campus network; ask the campus contact when something cannot be determined, and never guess endpoints.
- **T2-002** (Fill provenance sheets for lettuce image and sensor data): dataset version/date and licence terms are not recorded in the overview; look them up and record 'unverified' where they cannot be confirmed.
- **T2-003** (Fill provenance sheets for spinach image and sensor data): the canonical URL of the USDA Ag Data Commons file is not recorded (Overview §13.1); find and record it.
- **T2-007** (Audit the tomato sensor CSVs against the script and README): the per-rule severity scales depend on the full Kolar file (`kolar_tomato_precision_agri_dataset.csv`, 200,000 rows), which is not in the repo; without it primary-mode labels cannot be fully reproduced (see the rules-oracle task).
- **T2-010** (Create the lettuce sensor validation set (grouped hold-out)): `pH_Low` has 10 real rows in total, so it may have almost none left in train after validation and test; report class coverage and ask before merging or dropping any class.
- **T2-014** (Detect exact and near-duplicate images): choice of perceptual-hash library and threshold; start conservative and note the choice in the report.
- **T2-018** (Define sensor feature schemas and leakage blacklists as a single source of truth): hard bounds for spinach and tomato fields (EC, N, P, K); start from physical limits and list them for team review.
- **T3-002** (Implement server-side upload validation utility): maximum upload size (default 10 MB).
- **T3-003** (Create per-crop sensor request validators): whether any model should accept partially missing input; default is reject.
- **T4-020** (Train a live-compatible tomato sensor variant): which fields the server provides; do not start before the probe ADR exists.
- **T5-005** (Implement spinach feature extractor (6 features)): denominator of `relative_leaf_area`; default is segmented foreground pixels divided by total image pixels, documented in code and metadata.
- **T5-006** (Decide whether droop_angle is defensible (keep or remove)): whether the team prefers keeping a possibly weak feature; default is remove if no defensible reference axis exists.
- **T5-009** (Implement per-leaf mask validation): numeric bounds; tune them on the mask review (task after next) and record the final values in config.
- **T6-001** (Define Pydantic request/response schemas): if schemas already exist (audit), extend them instead of replacing.
- **T6-008** (Implement the rule-based recommendation provider): whether a domain expert will review the wording; until then every rule is marked 'advisory, unreviewed'.
- **T6-013** (Implement CampusServerAdapter, GET /device/sensors/latest and GET /device/camera/frame): everything protocol-specific until T1-016 is complete; do not start before that ADR exists.
