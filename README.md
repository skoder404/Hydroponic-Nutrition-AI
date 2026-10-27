# 🌱 Hydroponic Nutrition AI

A crop-condition **decision-support** web application for **lettuce, spinach, and tomato** grown hydroponically. It combines plant-image analysis, sensor-based condition classification, and cautious, explainable recommendations — and it is honest about what it does *not* know.

> **Project status:** 🚧 Planning complete · implementation in progress. Nothing in this README claims a feature works until the matching task in [`todo.md`](./todo.md) is checked off **and** backed by tests/artifacts.
>
> **Team:** Yashwant Vadhan M · Sushil Kumar M A — a two-person project where **both members are expected to learn every ML concept equally** (see [How we work](#-how-we-work-equal-learning)).

---

## ⚠️ Read this first — what this tool is (and isn't)

- A model output is a **prediction**, not a confirmed diagnosis.
- Several sensor labels are **rule-derived** (thresholds), not observed plant-health outcomes. High scores mean *agreement with the project rules on a documented split*, **not** field accuracy.
- **Image and sensor datasets are not paired.** Results are shown side by side, never merged into one score.
- Missing readings are **missing** (never zero). Old records are **never** shown as live.
- Recommendations are **advisory**. The system never controls pumps or doses nutrients.
- Authentication is **not** included in v1 (intentional scope decision).

Full details: [`PROJECT_OVERVIEW.md`](./PROJECT_OVERVIEW.md) §2, §19, §28.

---

## ✨ Features (planned MVP)

| Area | What it does |
|---|---|
| **Crop & mode** | Choose lettuce / spinach / tomato; analyse by image, sensors, or both |
| **Image classification** | Custom CNN · EfficientNetB0 (transfer learning) · OpenCV features + Random Forest/SVM |
| **Segmentation** | SAM 2.1 splits a plant photo into individual leaves; every leaf is validated and gets its own class and confidence |
| **Sensor classification** | Decision Tree · Random Forest · SVM (+ other classifiers) on each crop's exact schema |
| **Uncertainty** | Low-confidence results are labelled *uncertain* and don't drive strong advice |
| **Recommendations** | Gemini-assisted text with a **deterministic rule-based fallback** |
| **History** | PostgreSQL-backed, timestamped, provenance-tagged records |
| **Live campus data** | Latest sensor readings and camera frames from the campus server (configured IP), with freshness checks and manual fallback |
| **UI** | Earthy, nature-inspired React interface with purposeful animation, kept clean and accessible |
| **ML Lab** | Reproducible training, evaluation reports, and a versioned model registry with rollback |

---

## 🧱 Architecture (high level)

```
Next.js UI ──► FastAPI ──► validation ─► SAM 2.1 / OpenCV / PyTorch / scikit-learn
                  │                         │
                  ├─► Recommendation (Gemini ▸ rules fallback)
                  ├─► Device adapter (manual | mock | campus server IP: sensors + camera)
                  └─► PostgreSQL (history)      Local disk: image_data/ sensor_data/ models/
```
Details: [`TECH_RULES.md`](./TECH_RULES.md).

**Stack:** Next.js · React · FastAPI · PostgreSQL · PyTorch · scikit-learn · OpenCV · SAM 2.1 · Gemini API · local file storage.
**Hardware target:** HP Victus 16 (NVIDIA RTX 3050) *when the runtime supports it*; **CPU fallback is always supported**.

---

## 📚 Documentation map

| File | Purpose |
|---|---|
| [`PROJECT_OVERVIEW.md`](./PROJECT_OVERVIEW.md) | **Canonical source of truth** (scope, datasets, policies, limitations) |
| [`PRD.md`](./PRD.md) | Product requirements, personas, user stories, MVP scope, KPIs |
| [`DESIGN.md`](./DESIGN.md) | UX, screens, states, design system, accessibility |
| [`TECH_RULES.md`](./TECH_RULES.md) | Architecture, stack, coding/ML/security/testing/Git rules |
| [`ROADMAP.md`](./ROADMAP.md) | Milestones, dependencies, acceptance criteria, risk register |
| [`todo.md`](./todo.md) | Sequenced, atomic task list with pair assignments and learning goals |
| `docs/learning/` | Concept notes, teach-backs, quizzes (created during the project) |
| `docs/datasets/` | Dataset provenance sheets |
| `docs/decisions/` | Architecture decision records |
| [`docs/deployment/https.md`](./docs/deployment/https.md) | HTTPS/TLS reverse-proxy configuration (Caddy/NGINX) and HSTS |

---

## 🗂️ Datasets (sources as documented in the overview)

| Crop | Modality | Source |
|---|---|---|
| Lettuce | Image | Kaggle — *Lettuce NPK Dataset* (`baronn/lettuce-npk-dataset`) |
| Lettuce | Sensor | *HydroGrowNet*, Mendeley Data, DOI `10.17632/g6cm3v3wdp.5` |
| Spinach | Image | Mendeley Data `ymtzd8cf6x` (v1) and `skf6w2s2h2` (v3) |
| Spinach | Sensor | Derived from Meng et al. (2025), *J. Agric. Food Res.* 24:102530 (DOI `10.1016/j.jafr.2025.102530`) + synthetic/derived water-chemistry features |
| Tomato | Image | Mendeley Data `zfv4jj7855` (v1); Kaggle *Tomato Disease (Multiple Sources)* (11 class folders on disk, incl. `Tomato_Powdery_Mildew`) |
| Tomato | Sensor | Kaggle — *Autonomous Tomato Cultivation Dataset in Kolar* (transformed; see limitations) |

> Raw data lives in `image_data/` and `sensor_data/`; generated splits/features go to `processed_data/`. Licences and exact versions must be recorded in `docs/datasets/` before any redistribution. Raw data is kept read-only; see [`TECH_RULES.md`](./TECH_RULES.md#folder-structure).

---

## 🚀 Getting started

> **Template.** Exact commands, versions, and paths are finalised in Phase 1 of [`todo.md`](./todo.md) (tasks T1-xxx). Treat the block below as the intended shape, not verified instructions.

```bash
# 1. Clone
git clone https://github.com/skoder404/Hydroponic-Nutrition-AI hydroponic-nutrition-ai && cd hydroponic-nutrition-ai

# 2. Configure environment (placeholders only in .env.example)
cp .env.example .env        # then fill in values locally — never commit .env

# 3. Backend (Python env + deps → exact tool chosen in Phase 1)
#    create venv, install pinned requirements, run migrations, start API

# 4. Frontend
#    cd frontend && install deps && start dev server

# 5. Datasets & models
#    Place raw data in image_data/ and sensor_data/ (read-only); see docs/datasets/
```

**Key environment variables** (see `.env.example`): `DATABASE_URL`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `MODEL_REGISTRY_PATH`, `IMAGE_DATA_PATH`, `SENSOR_DATA_PATH`, `PROCESSED_DATA_PATH`, `CAMPUS_SERVER_URL`, `CAMPUS_CAMERA_URL`, `ALLOWED_ORIGINS`, `MAX_UPLOAD_MB`, `DEVICE_ADAPTER`, `SAM2_CHECKPOINT`, `SAM2_CONFIG`, `FORCE_CPU`.

---

## 🤝 How we work (equal learning)

Because the goal is for **both** teammates to master the ML, `todo.md` is built around a pairing protocol:

1. **Driver / Navigator.** Every task names a driver (writes the code) and a navigator (reviews the pushed commits). Roles alternate so that drive-counts are balanced in every phase.
2. **Concept note.** The *navigator* writes the concept note in `docs/learning/` (e.g., "Gini impurity", "Macro-F1"). Explaining is how you learn it.
3. **Teach-back.** At the end of each concept block, the person who *drove* teaches it back; both answer the self-check questions.
4. **From-scratch first.** For core ideas (decision-tree split, confusion matrix/macro-F1, convolution), both build a tiny from-scratch version before using the library.
5. **No phase is closed** until both members have ticked the "Concept check" box for that phase.

---

## 🧪 Testing & quality

Layered tests (unit · integration · contract · data/leakage · E2E). CI runs lint, type-checks, and the lightweight suites on every push to `main`; heavy training/SAM runs are executed locally with reports committed under `reports/`. A task is "done" only when the implementation **and** evidence (tests/artifacts) exist.

---

## ⚖️ Known limitations (summary)

Rule-generated labels · oversampled/artificial class balance · mixed real/synthetic spinach features · transformed tomato features with a synthetically derived water temperature · tomato source units not fully verified · unpaired image/sensor data · confidence ≠ calibrated probability · lettuce leaf crops inherit the whole-plant label (weak labels). Full list: [`PROJECT_OVERVIEW.md` §28](./PROJECT_OVERVIEW.md).

---

## 🔐 Security notes

No login in v1 (see [ADR 0005: No Auth](./docs/decisions/0005-no-auth.md)) · secrets only via `.env` · uploads validated (type, size, decodability) · Gemini output treated as untrusted · model loading limited to configured paths · HTTPS/TLS required for all non-localhost deployments (see [HTTPS Deployment Guide](./docs/deployment/https.md)). See [`TECH_RULES.md`](./docs/TECH_RULES.md#security-rules).

---


## 🙏 Acknowledgements

Dataset authors and maintainers listed above; Meta AI's SAM 2.1; PyTorch, scikit-learn, OpenCV, FastAPI, Next.js communities.
