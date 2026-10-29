# TECH_RULES — Hydroponic Nutrition AI

> **Status:** Draft v0.2 · **Date:** 2026-10-20
> **Authority:** `PROJECT_OVERVIEW.md` is canonical. These rules operationalise it. If a rule here conflicts with the overview, stop and update one of them deliberately.
> **Version policy:** This document deliberately does **not** hard-code library/runtime versions. Pin exact versions at setup *after verifying compatibility* (see [Verify at Setup](#verify-at-setup)). Inventing versions here would be guessing.
> **Route names** below are **PROPOSED** (Overview §22: real routes must come from code).

---

## Architecture Overview

### Frontend Architecture
- **Next.js (App Router) + React + TypeScript.** Server components for static pages; client components for forms/upload/results.
- State: React state + a small data-fetching layer (`fetch` wrapper or TanStack Query — decide at setup). No global store unless justified.
- API access through a single typed client in `frontend/lib/api/`. Types mirror backend Pydantic schemas (generate from OpenAPI where feasible).
- Styling: Tailwind CSS with design tokens from `DESIGN.md` (decide at setup; CSS Modules acceptable).
- Animation: Framer Motion (verify package name/version at setup) behind a single `MotionProvider` that honours `prefers-reduced-motion`; CSS transitions for micro-interactions; animate only `transform`/`opacity`/SVG stroke.

### Backend Architecture
- **FastAPI** (async routes), Pydantic v2-style schemas (verify installed major version), layered:
  `api/` (routers) → `services/` (orchestration) → `ml/` (inference modules) → `database/` (repositories).
- Inference is **CPU/GPU-agnostic** via a `DeviceManager` (runtime detection, never assumed).
- Heavy inference (SAM 2.1, CNN) runs in a **worker threadpool or separate process** so the event loop is not blocked. `⚠️ UNCLEAR:` whether to introduce a task queue (default: no; threadpool + timeouts).
- A **ModelRegistry** service loads only *promoted* artifacts from configured paths and validates schema compatibility at startup.
- A **RecommendationService** with two providers: `GeminiProvider`, `RuleProvider` (fallback), behind one interface.
- A **DeviceAdapter** interface with `ManualAdapter`, `MockAdapter`, and `CampusServerAdapter` (HTTP client to the configured campus server IP: latest sensor readings + camera frame; exact protocol decided after the server probe in `todo.md`). All calls go only to the configured host.

### Database Architecture
- **PostgreSQL** for application records only (history, model-run metadata references). Datasets, features, and model artifacts live on local disk (Overview §4).
- Migrations: **Alembic** with SQLAlchemy 2.x (verify versions). Timestamps stored as `timestamptz` (UTC) **plus** the original timezone string.

Core tables (proposed; final names set in migration tasks):
| Table | Key columns |
|---|---|
| `analysis_records` | `id` (UUID), `crop`, `mode`, `created_at`, `observed_at`, `observed_tz`, `source` (`manual|live|imported`), `notes` |
| `predictions` | `id`, `record_id` FK, `modality` (`image|sensor|image_features`), `predicted_class`, `display_label`, `confidence`, `confidence_semantics`, `uncertain` bool, `experimental` bool, `model_id`, `model_version`, `preprocessing_version`, `warnings` JSONB |
| `inputs` | `id`, `record_id` FK, `kind` (`sensor|image`), `payload` JSONB (sensor values) or `file_ref` (path/hash), `units` JSONB |
| `recommendations` | `id`, `record_id` FK, `source` (`gemini|rules`), `rules_version`, `content` JSONB, `evidence` JSONB |

### Deployment Architecture
- Local-first: `docker compose` (Postgres + backend + frontend) for reproducibility; model/data directories mounted as volumes.
- GPU via host runtime when available (Docker GPU passthrough optional; native venv acceptable on the Victus). CPU fallback mandatory.
- No cloud storage assumed. Hosting choice for demo: `⚠️ UNCLEAR` (default: run locally / campus machine).

### Architecture Diagram
```
 Browser (Next.js)
     │  HTTPS (dev: http://localhost)
     ▼
 FastAPI ── validation (Pydantic) ── rate limit ── CORS
     │
     ├─ ImageService ─► preprocess ─► [SAM 2.1 leaf extractor] ─► per-leaf mask validator ─► one prediction per leaf
     │        ├─► CNN / EfficientNetB0 (PyTorch)  ◄── ModelRegistry (promoted artifacts, schema check)
     │        └─► OpenCV features ─► RF / SVM (sklearn)
     ├─ SensorService ─► schema+unit+range check ─► scaler+classifier (sklearn)
     ├─ RecommendationService ─► Gemini ▸ (timeout/invalid) ▸ RuleProvider
     ├─ DeviceAdapter (Manual | Mock | CampusServer ─► campus IP: sensors + camera)
     └─ HistoryRepository ─► PostgreSQL
 Local disk: image_data/ sensor_data/ processed_data/ models/ (artifacts + manifests) uploads/ masks/
```

---

## Tech Stack

| Layer | Technology | Purpose | Why Chosen | Alternatives Considered |
|---|---|---|---|---|
| Frontend | Next.js + React + TypeScript | UI | Mandated by overview; SSR/routing; typed | Vite+React, SvelteKit |
| Styling | Tailwind CSS | Tokens/responsive | Fast, consistent with design tokens | CSS Modules, MUI |
| Animation | Framer Motion + CSS | Purposeful motion (vine progress, meters, entrances) | Declarative React variants, reduced-motion support | CSS only, GSAP, Lottie |
| Backend | FastAPI | REST API | Mandated; async; auto OpenAPI; Pydantic validation | Flask, Django REST |
| HTTP client | httpx | Campus server calls (timeouts, retries) | Async, FastAPI-friendly | requests |
| DB | PostgreSQL | History | Mandated; `timestamptz`, JSONB | SQLite (fine for tests only) |
| ORM/Migrations | SQLAlchemy + Alembic | Persistence | Standard w/ FastAPI | SQLModel, raw SQL |
| Deep learning | PyTorch (+ torchvision) | Custom CNN, EfficientNetB0 | Mandated; torchvision provides EfficientNet-B0 | TensorFlow/Keras |
| Classical ML | scikit-learn | DT/RF/SVM/KNN/LogReg, scalers, metrics | Mandated; transparent for learning | XGBoost/LightGBM (optional extras) |
| CV | OpenCV, scikit-image | Feature extraction, GLCM | OpenCV for colour/contours; scikit-image GLCM (`graycomatrix`/`graycoprops`) | mahotas |
| Segmentation | SAM 2.1 | Plant/leaf isolation | Mandated initial approach | classical HSV/GrabCut fallback (see below) |
| Recommendations | Gemini API (+ rules) | Advisory text | Mandated; fallback required | Rules only |
| Data/analysis | pandas, NumPy, matplotlib/seaborn, Jupyter | EDA & plots | Standard; learning-friendly | Polars |
| Packaging | `uv` or `pip-tools` + venv; `npm`/`pnpm` | Reproducible envs | Lockfiles | Poetry, conda |
| Lint/format | ruff, ESLint, Prettier | Quality | Fast, standard | black+flake8 |
| Types | mypy *or* pyright; TypeScript strict | Safety | Catch contract errors | — |
| Tests | pytest, pytest-cov, httpx, Playwright | Layered tests | Standard | unittest, Cypress |
| CI | GitHub Actions | Lint+test | Repo is on GitHub | — |
| Container | Docker + Compose | Reproducible dev | Parity | Native venvs |

**Optional fallback:** if SAM 2.1 cannot run on the available hardware, a classical fallback (HSV threshold + largest contour) may be added **only** as an explicitly labelled alternative with its own metadata; it must never be silently substituted (Overview §8.1).

---

## Folder Structure

Adapt to the existing repo; do **not** create parallel duplicates (Overview §4.1). Proposed layout (adds `docs/learning/`, `reports/`):

```text
.
├── README.md
├── PROJECT_OVERVIEW.md        # canonical source of truth
├── PRD.md  DESIGN.md  TECH_RULES.md  todo.md
├── .env.example  .gitignore  .gitattributes  pre-commit config
├── frontend/
│   ├── app/                   # routes: /, /analyze, /history, /history/[id], /models, /about
│   ├── components/            # CropCard, ModeSelector, ImageDropzone, SensorForm, ConfidenceMeter, ...
│   ├── lib/api/               # typed API client
│   ├── lib/validation/        # client-side mirrors of server rules (UX only)
│   └── public/
├── backend/
│   └── app/
│       ├── api/               # routers: health, predict, history, models, device
│       ├── schemas/           # Pydantic request/response models
│       ├── services/          # image_service, sensor_service, recommendation_service, history_service
│       ├── database/          # models, session, repositories, alembic/
│       └── core/              # config, logging, errors, device_manager, security
├── ml/
│   ├── image_models/          # custom_cnn.py, efficientnet.py, train_image.py
│   ├── sensor_models/         # train_sensor.py, schemas (feature order), rules
│   ├── segmentation/          # sam2_segmenter.py, leaf_extraction.py, mask_validation.py
│   ├── features/              # lettuce.py, spinach.py, tomato.py, glcm.py
│   ├── fusion/                # placeholder: NO fusion model (see rules)
│   ├── recommendations/       # rules/ (versioned), gemini_prompt.py, schema.py
│   ├── evaluation/            # metrics.py, report.py, cv.py
│   └── registry/              # registry.py, metadata_schema.py, promote.py
├── image_data/                # EXISTING raw images, READ-ONLY, git-ignored
│   ├── Lettuce_Images/        #   4 class folders
│   ├── Spinach_Images/        #   Spinach_Healthy + Spinach_Unhealthy/<5 classes>
│   └── Tomato_Images/         #   Tomato_Healthy + Tomato_Unhealthy/<10 classes>
├── sensor_data/               # EXISTING raw sensor files, READ-ONLY, git-ignored
│   └── Lettuce_/Spinach_/Tomato_Sensor-Readings/
├── processed_data/            # GENERATED: splits/, features/, sensor/ (train,val,test), manifests/
├── models/
│   ├── image/  sensor/        # versioned artifacts (git-ignored by default)
│   └── manifests/             # metadata JSON committed to git
├── reports/                   # evaluation reports (committed: small JSON/MD/PNG)
├── notebooks/                 # exploration ONLY; scripts are authoritative
├── scripts/                   # build_*_dataset.py, run_eval.py, promote_model.py
├── tests/
│   ├── unit/  integration/  contracts/  data/
└── docs/
    ├── learning/              # concept notes, teach-back logs, quizzes
    ├── datasets/              # provenance sheets
    └── decisions/             # ADRs
```

**Storage rules (Overview §4.1, §9.2):** raw data immutable; processed/generated data separated; split manifests never overwritten; large binaries git-ignored (document how to obtain them in `docs/datasets/`).

---

## Coding Standards

### Naming
| Item | Convention | Example |
|---|---|---|
| Python files/modules | `snake_case` | `mask_validation.py` |
| Python classes | `PascalCase` | `SensorSchema` |
| Python functions/vars | `snake_case` | `extract_lettuce_features` |
| Constants | `UPPER_SNAKE` | `LETTUCE_SENSOR_FEATURES` |
| React components | `PascalCase.tsx` | `ConfidenceMeter.tsx` |
| TS vars/functions | `camelCase` | `fetchHistory` |
| API paths | lowercase kebab, plural nouns | `/api/v1/history` |
| JSON fields | `snake_case` (match Pydantic) | `model_version` |
| Class labels | **verbatim from source** in metadata; display mapping separate | `Spinach_bacterial spot` |
| DB | `snake_case`, plural tables | `analysis_records` |

### Component / Module Structure
- One responsibility per module; services never import routers; ML code never imports FastAPI.
- Feature schemas live in **one** place per crop (`ml/features/<crop>.py` and `ml/sensor_models/schemas.py`) and are imported by training *and* inference (single source of truth for names + order).
- Frontend: presentational components are pure; data fetching in route-level hooks.

### API Standards (ESTABLISHED v1)
- Base: `/api/v1`. JSON in/out except multipart image upload.
- Endpoints: `GET /health`, `GET /models`, `POST /predict/sensor`, `POST /predict/image`, `POST /analyze` (both modalities + recommendation), `GET /history`, `GET /history/{record_id}`, `POST /history`, `GET /history/export`, `GET /device/status`, `GET /device/sensors/latest`, `GET /device/camera/frame`, `POST /device/manual`.
- Reject unknown fields (`extra="forbid"`). Pagination: `limit` (default 20, max 100) + `offset`/cursor.
- Prediction response includes: `crop, modality, predicted_class, display_label, confidence, confidence_semantics, uncertain, experimental, model_id, model_version, timestamp, provenance, warnings[], limitations[]`. Image predictions also include `input_mode`, `leaves[]` (per leaf: `leaf_id, bbox, mask_ref, predicted_class, display_label, confidence, uncertain`), `rejected_leaves[]`, and `summary` (leaf count and per-class counts; never an averaged confidence).
- Versioning: breaking changes → `/api/v2`. OpenAPI is the contract; contract tests in `tests/contracts/`.

### Error Handling Strategy
Single error envelope:
```json
{ "error": { "code": "SEGMENTATION_FAILED", "message": "…", "details": {…}, "retryable": true } }
```
| Situation | HTTP | `code` |
|---|---|---|
| Invalid field/unit/range | 422 | `INVALID_INPUT` |
| Missing required sensor field | 422 | `MISSING_FIELD` |
| Unsupported image/too large | 415 / 413 | `UNSUPPORTED_MEDIA` / `PAYLOAD_TOO_LARGE` |
| Segmentation/mask poor | 422 | `SEGMENTATION_FAILED` (`retryable: true`) |
| Feature schema mismatch | 409 | `SCHEMA_MISMATCH` |
| No promoted/corrupt model | 503 | `MODEL_UNAVAILABLE` |
| Device stale/disconnected | 200 w/ warning or 503 | `DEVICE_UNAVAILABLE` |
| Gemini failure | — | handled internally → fallback, `source: "rules"` |
| DB failure on save | 503 | `PERSISTENCE_FAILED` (response states **not saved**) |
Never leak stack traces, paths, or secrets.

### Logging Standards
- Structured JSON logs (`event`, `request_id`, `crop`, `model_id`, `latency_ms`, `device`, `level`).
- Log stages: validation, segmentation, feature extraction, inference, recommendation fallback, persistence.
- Never log secrets, raw API keys, full uploaded content, or personal data. Log counts/hashes instead.

---

## ML Engineering Rules (project-critical)

1. **Split discipline.** Image split 70/15/15 stratified **after** duplicate/near-duplicate checks. Sensor data: keep the existing *test* partitions exactly as prepared (hash-guarded); create the *validation* set (none exists) from the source CSV used for the split, as a grouped hold-out **before** oversampling (lettuce: calendar day; spinach: six-hour block; tomato: calendar month). Oversample the training partition only. Never overwrite existing files; write v2 splits to `processed_data/`; tests assert manifest hashes.
2. **Group integrity.** All leaf crops from one original lettuce image share a split.
3. **Fit-on-train-only.** Scalers, imputers (if any), calibration, and thresholds are fit on training (or validation for calibration/thresholds) — never test.
4. **Augmentation** training-only.
5. **Selection** by **validation macro-F1**; test set used exactly once for final reporting and logged.
6. **Reporting** per Overview §18.2 (accuracy, macro P/R/F1, per-class P/R/F1/support, confusion matrix, counts, protocol, versions).
7. **Cross-validation** for sensor time series: group/time-aware only (e.g., `GroupKFold` by calendar date). Never random row-wise folds on correlated data.
8. **Leakage blacklist (per crop, enforced in code):** lettuce: `label_next`, trend columns (`*_prev/_delta/_roll3`), daily aggregates, `tds`, `co2`, `lux_meter`, `hours_logged`, `experiment`, `replicate`, `treatment`, `date`, `day_index`, `slot`, `timestamp`; spinach: `tds_ppm`, `label_id`, `humidity_source`, `sample_id`, `source_timestamp_EDT`, `bench`; tomato: `n_rules_broken`, `source_row`, `source_timestamp`, `label_id`, `sample_id`. The target never enters features.
9. **Feature contracts** are versioned. Change a feature → regenerate features → retrain dependents → mark old artifacts incompatible.
10. **Missing values:** never `0`. Reject or use an explicitly trained/validated path.
11. **Seeds & determinism:** set and record seeds for Python/NumPy/PyTorch; note non-determinism sources (cuDNN, DataLoader workers).
12. **Device honesty:** log the device actually used; never claim GPU use without runtime evidence (`torch.cuda.is_available()` + actual tensor device).
13. **Confidence:** softmax ≠ calibrated probability. Label as "model score" unless calibration is fitted and validated.
14. **Model registry metadata** (Overview §24.1) is mandatory; promotion is explicit; keep previous artifact for rollback.
15. **No fusion model** may be trained on unpaired image/sensor data.
16. **Leaf-level unit.** Every image model consumes exactly one leaf. Lettuce crops inherit the split and label of their source image (weak labels; document the manually reviewed inconsistency rate); spinach/tomato use the provided leaf images. Training and inference use the same leaf-mask routine. Per-leaf predictions are never averaged into one score.
17. **Notebooks** may explore; any result used in a report must be reproduced by a script.

---

## Security Rules

- **Authentication:** none in v1 by design (Overview §2.3, §23). Document this in README. Adding auth = scope change + new threat model.
- **Authorization:** N/A in v1; instead restrict *capabilities*: no endpoint can overwrite models, datasets, or run code; admin actions (promote/rollback) are CLI-only, not HTTP.
- **Validation:** Pydantic with `extra="forbid"`; numeric bounds per field; image MIME + magic bytes + `PIL`/OpenCV decode test + dimension/size caps; never use client filename as a path (generate UUID names).
- **Gemini safety:** treat output as untrusted; parse to schema; strip/reject URLs, commands, dosage numbers; never send API keys or unnecessary personal data; timeouts + retries bounded.
- **Rate limiting:** per-IP limits (e.g., SlowAPI or reverse-proxy), stricter on `/predict/image`.
- **CORS:** explicit allow-list (frontend origin).
- **Campus server calls:** the server IP lives in `.env` only (never committed). The backend calls only the configured host (no user-supplied URLs, to prevent SSRF); enforce connect/read timeouts, a response-size cap for frames, and content-type checks; treat all device payloads as untrusted input and validate them like manual input.
- **Secrets:** `.env` (git-ignored) + committed `.env.example` with placeholders only; secret scanning in CI/pre-commit (e.g., gitleaks).
- **Model loading:** only from configured trusted directories; prefer `weights_only`-style safe loading where supported by the installed PyTorch (verify); never unpickle untrusted files.
- **HTTPS/TLS:** required for any non-localhost deployment (reverse proxy); HSTS at proxy; local dev may use HTTP.
- **Dependencies:** lockfiles; periodic `pip-audit` / `npm audit`.
- **Licences:** verify dataset/model-weight licences before redistribution.

---

## Performance Rules

- **Frontend:** code-split heavy components; lazy-load mask overlay; compress/resize images client-side *only as an optional upload aid* (the backend remains authoritative on preprocessing); cache static assets.
- **Backend:** load models once at startup (lazy for SAM 2.1 if memory-limited); run blocking inference off the event loop; request timeouts; bounded concurrency semaphore for GPU work.
- **GPU/CPU:** adaptive batch size; catch CUDA OOM → smaller batch → CPU fallback or clear error; use mixed precision only if verified to not change results materially.
- **DB:** index `analysis_records(created_at DESC)`, `(crop, created_at)`, `predictions(record_id)`; paginate every list endpoint; no `SELECT *` in hot paths.
- **Caching:** HTTP caching for static assets; in-memory cache for registry metadata. No Redis in v1 (`⚠️` revisit only if needed).

---

## Testing Rules

| Layer | Tools | Scope | Target |
|---|---|---|---|
| Unit | pytest | class-name mapping; schema/order validation; unit & range checks; missing-value handling; label-rule boundaries & priority; each feature formula + edge cases (zero perimeter); GLCM config; mask validation; confidence thresholds; rule recommendations; registry compatibility | ≥ 70% lines on `backend/` and `ml/` *(planned)* |
| Integration | pytest + httpx + test DB | image upload→segment→predict; leaf crop→features→predict; sensor payload→predict; Gemini down→fallback; device unavailable→manual; save/retrieve history; corrupt/missing artifact; GPU-unavailable fallback | every route |
| Contract | OpenAPI schema tests | response shapes stable | all v1 routes |
| Data/Leakage | pytest | class counts before/after; no image identity overlap; same-source crops same split; sensor group/time separation; prohibited fields absent; target not in features; manifests unchanged (hash); no NaN/Inf; column order == artifact metadata | all datasets |
| E2E | Playwright | crop→sensor→result→save→history; image upload w/ seg failure→retry; device unavailable | critical paths |

Rule: a feature is "done" only when tests/artifacts demonstrate it (Overview §27.4). Don't report tests as passing unless they were run.

---

## Git Workflow

- **Branches:** none (team decision). Both teammates commit directly to `main` in small, single-task commits.
- **Commits:** Conventional Commits — `feat(sensor): add lettuce RF trainer`, `fix(seg): reject empty mask`, `docs(learning): add SVM note`. Imperative, ≤ 72 chars subject.
- **Review:** the navigator reviews each task's commit(s) after they are pushed (`git log --grep "[T<id>]"`), checking: what/why, tests run, data/splits touched, metrics with split and metric named, limitations, and the linked learning note. Problems become follow-up `fix` commits by the driver.
- **Keeping `main` healthy:** `git pull --rebase` before starting a task and before every push; run lint and the relevant tests before committing; never force-push or rewrite pushed commits; if CI on `main` is red, fixing it comes before the next task.
- **Large files:** don't commit raw datasets or heavy artifacts; commit manifests/metadata.
- **Pairing protocol:** driver commits, navigator reviews; roles alternate per task as assigned in `todo.md`.

---

## Deployment Rules

- **CI (GitHub Actions):** on every push to `main`: ruff, type-check, pytest (unit/contract/data with tiny fixtures), frontend lint/typecheck/build; secret scan. Heavy training/SAM tests are **not** in CI (run locally; record reports).
- **Environments:** `dev` (local), `demo` (campus/laptop). `staging/prod` not defined `⚠️ UNCLEAR`.
- **Config:** all via env vars (`DATABASE_URL`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `MODEL_REGISTRY_PATH`, `IMAGE_DATA_PATH`, `SENSOR_DATA_PATH`, `PROCESSED_DATA_PATH`, `CAMPUS_SERVER_URL`, `CAMPUS_CAMERA_URL`, `CAMPUS_POLL_INTERVAL_S`, `CAMPUS_MAX_AGE_S`, `CAMPUS_REQUEST_TIMEOUT_S`, `ALLOWED_ORIGINS`, `MAX_UPLOAD_MB`, `DEVICE_ADAPTER`, `SAM2_CHECKPOINT`, `SAM2_CONFIG`, `FORCE_CPU`).
- **Monitoring:** structured logs + `/health` (DB, registry, device availability, GPU status); fallback counters (Gemini→rules, GPU→CPU).
- **Backup:** nightly `pg_dump` for demo DB (script); model manifests in git; artifacts backed up manually; keep previous promoted artifact.

---

## Verify at Setup

These items are *not* asserted here; verify and record in `docs/decisions/`:

1. Python & Node versions compatible with chosen PyTorch/CUDA build for the RTX 3050 (and CPU-only build for campus).
2. SAM 2.1 install method, checkpoint names/sizes, config paths, and VRAM needs on the RTX 3050; CPU viability.
3. Current Gemini model name/quota and SDK usage (keep `GEMINI_MODEL` configurable; do not hard-code).
4. scikit-image GLCM function names/normalisation in the installed version (`graycomatrix`/`graycoprops`), and whether symmetric/normed flags match Overview §16.4.
5. Pydantic / SQLAlchemy major versions and their idioms.
6. Dataset licences and exact versions (Overview §29.3).
7. `torchvision` EfficientNet-B0 weights enum/API in the installed version.
8. Campus server: protocol, endpoints, auth, payload fields/units, update rate, camera format (probe from the campus network; record in `docs/decisions/`).
9. Framer Motion package name/version and React/Next.js compatibility.
