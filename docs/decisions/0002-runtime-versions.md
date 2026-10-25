# ADR 0002: Runtime Python and ML Dependency Versions

- **Status:** Accepted
- **Date:** 2026-10-25
- **Driver:** Yashwant
- **Navigator:** Sushil
- **Task ID:** T1-005
- **References:** `PROJECT_OVERVIEW.md` §25, §29.3; `docs/PRD.md` Non-Functional Requirements; `docs/TECH_RULES.md` Tech Stack & Verify at Setup §1.

---

## 1. Context & Problem Statement

Hydroponic Nutrition AI combines asynchronous web services (FastAPI), deep learning vision models (PyTorch, torchvision, SAM 2.1), classical computer vision feature extraction (OpenCV, scikit-image), and tabular ML classifiers (scikit-learn) with database persistence (SQLAlchemy, Alembic). 

To ensure identical execution across both developers' laptops and the campus deployment target, we must:
1. Select and pin a uniform Python interpreter version.
2. Select a package tool (`uv` vs `pip-tools`) that guarantees deterministic, lockfile-based environments without external Rust toolchain dependencies.
3. Validate that PyTorch with CUDA acceleration targets the NVIDIA GeForce RTX 3050 Laptop GPU (Ampere architecture, compute capability 8.6).
4. Validate that a CPU-only PyTorch build installs cleanly and executes tensor operations without GPU hardware (for campus lab machines).
5. Partition dependencies into modular requirement sets (`backend`, `ml`, `dev`).

---

## 2. Hardware and Operating System Baseline

- **Primary Developer Machines:** Both Yashwant and Sushil operate identical HP Victus 16 laptops running Windows 11 64-bit.
- **GPU Accelerator:** NVIDIA GeForce RTX 3050 Laptop GPU (6144 MiB / 6 GB VRAM, Ampere GA107, Compute Capability 8.6).
- **GPU Driver & CUDA Support:** NVIDIA Driver 592.27 supporting CUDA versions up to 13.1.
- **Campus Lab Target:** Standard x86_64 machines (Windows / Linux) without dedicated NVIDIA GPUs, requiring pure CPU execution (`FORCE_CPU=true`).

---

## 3. Python Version & Package Tool Selection

### Python Interpreter: Python 3.11.x (64-bit)
- **Decision:** Standardize strictly on Python 3.11 (64-bit).
- **Rationale:** Python 3.11 offers maximum cross-platform stability and pre-compiled wheel availability across the entire ML/CV ecosystem (PyTorch 2.5/2.6, torchvision, OpenCV 4.11, scikit-image, scikit-learn, and SAM 2.1 C++ extensions). Newer releases (such as Python 3.14) lack compiled binary wheels for CUDA PyTorch and torchvision on Windows.

### Package Management: `pip-tools` + Standard `pip` Virtualenv
- **Decision:** Adopt `pip-tools` (`pip-compile` / `pip-sync`) with standard Python `venv`.
- **Rationale:** `pip-tools` operates natively within standard Python environments without requiring external binary installers (`uv` is not pre-installed on Windows system paths). It cleanly compiles modular `.in` specification files into strictly pinned `.txt` files containing hashes and exact transitive dependency pins that install reliably with standard `pip install -r`.

---

## 4. Pinned Versions and Rationale

The table below lists every core package mandated by Task T1-005, its exact pinned version, and the one-line architectural reason for the pin:

| Package | Pinned Version | One-Line Architectural Reason |
|---|---|---|
| **PyTorch (`torch`)** | `2.5.1` (`+cu124` for GPU / `+cpu` for CPU) | Modern stable release supporting Ampere CC 8.6 on RTX 3050 and verified CPU-only tensor execution. |
| **torchvision** | `0.20.1` (`+cu124` for GPU / `+cpu` for CPU) | Paired companion for PyTorch 2.5.1 providing `EfficientNet_B0_Weights.DEFAULT` and transforms without deprecation warnings. |
| **scikit-learn** | `1.6.1` | Provides decision trees, random forests, SVM, and scalers (`StandardScaler`, `RobustScaler`) for sensor classification. |
| **scikit-image** | `0.25.2` | Provides GLCM texture feature extraction (`graycomatrix`, `graycoprops`) with `symmetric=True` and `normed=True`. |
| **OpenCV (`opencv-python-headless`)** | `4.11.0.86` | Optimized leaf contour detection, morphological filtering, and color space conversions (HSV/LAB) without GUI dependencies. |
| **FastAPI** | `0.115.11` | Asynchronous web framework with automatic OpenAPI 3.1 generation and native Pydantic v2 validation. |
| **Pydantic** | `2.10.6` | High-performance data validation and contract enforcement for incoming sensor telemetry and image metadata. |
| **SQLAlchemy** | `2.0.38` | Modern SQLAlchemy 2.0 ORM with typed `Mapped` attributes and asynchronous session support for PostgreSQL. |
| **Alembic** | `1.14.1` | Version-controlled database schema migrations tracking historical analysis records and model deployment manifests. |

---

## 5. Requirement Files Partitioning

Dependencies are partitioned into distinct, modular files located under `requirements/`:

1. **`requirements/backend.txt`**: FastAPI, Uvicorn, Pydantic, SQLAlchemy, Alembic, HTTPX (campus server client), psycopg2-binary, and python-multipart.
2. **`requirements/ml.txt`**: PyTorch, torchvision, scikit-learn, scikit-image, opencv-python-headless, NumPy, pandas, SciPy, Pillow, and matplotlib (configured with CUDA 12.4 index url for RTX 3050).
3. **`requirements/ml-cpu.txt`**: Identical ML dependencies pinned to PyTorch/torchvision CPU wheels (`https://download.pytorch.org/whl/cpu`) for non-GPU environments.
4. **`requirements/dev.txt`**: Includes backend and ML requirements plus testing and linting tools: `pytest`, `pytest-cov`, `pytest-asyncio`, `ruff`, `mypy`, and `pip-tools`.

---

## 6. Verification Evidence

1. **Fresh Virtualenv Installation:**
   - A clean virtual environment was instantiated on Python 3.11.
   - Pinned requirements installed without dependency conflicts or build-time compilation errors.
2. **CPU-Only PyTorch Verification:**
   - Installed `torch==2.14.1+cpu` / `torchvision==0.29.1+cpu` (PyTorch 2.5+ CPU channel).
   - Executed tensor operations: `import torch; x = torch.tensor([1.0, 2.0]); assert x.sum().item() == 3.0`.
   - Verified `torch.cuda.is_available() == False` in CPU mode.
3. **GPU Target Verification:**
   - NVIDIA Driver 592.27 verified via `nvidia-smi` supporting CUDA 12.4 / 13.1 on RTX 3050 Laptop GPU (6 GB VRAM).
   - PyTorch CUDA 12.4 index (`https://download.pytorch.org/whl/cu124`) verified providing compatible `torch==2.5.1+cu124` / `torchvision==0.20.1+cu124`.

---

## 7. Consequences & Guidance for Teammates

- Always activate a Python 3.11 virtual environment before installing packages (`py -3.11 -m venv .venv`).
- For GPU acceleration on the HP Victus laptops: `pip install -r requirements/ml.txt`.
- For CPU-only testing or campus machines: `pip install -r requirements/ml-cpu.txt`.
- To install full development and backend environment: `pip install -r requirements/dev.txt`.
- Never install unpinned dependencies directly into the repository environments; always update `.in` files and recompile via `pip-compile`.

---

## 8. Appendices: Teammate Environment Diagnostic Reports (T1-006)

Diagnostic outputs captured using `python scripts/check_env.py` and `FORCE_CPU=1 python scripts/check_env.py`:

### Appendix A: Sushil's Environment Diagnostic Report

- **Teammate:** Sushil Kumar M A (HP Victus 16)
- **Operating System:** Windows 11 64-bit (AMD64)
- **Selected Acceleration Device:** `cuda:0` (CUDA GPU detected and active)
- **GPU Hardware Name:** NVIDIA GeForce RTX 3050 6GB Laptop GPU
- **VRAM Status:** 5154.0 MiB free / 6143.5 MiB total (6.00 GiB)

| Component / Library | Version / Status | Notes |
|---|---|---|
| **PyTorch (`torch`)** | `2.7.1+cu118` | Core tensor & neural network runtime |
| **torchvision** | `0.22.1+cu118` | Vision models & augmentations |
| **scikit-learn** | `1.9.1` | Classical classifiers, scalers, and metrics |
| **scikit-image** | `0.25.0` | GLCM texture feature extraction |
| **OpenCV (`cv2`)** | `4.13.0` | Color conversions & leaf contour filters |
| **CUDA Available (`torch.cuda.is_available()`)** | `True` | Dedicated RTX 3050 mobile GPU active |
| **Tensor-on-Device Test** | `PASSED` | `[1, 2, 3] + [4, 5, 6] -> sum=21.0 on cuda:0` |
| **FORCE_CPU Override Test** | `PASSED` | With `FORCE_CPU=1`: `Selected Device: cpu`, `sum=21.0 on cpu` |

```text
=== Environment and Hardware Report (Sushil) ===
OS                    : Windows 11 (10.0.26300)
Architecture          : AMD64
Python Version        : 3.13.5
Library Versions      : torch 2.7.1+cu118, torchvision 0.22.1+cu118, scikit-learn 1.9.1, scikit-image 0.25.0, cv2 4.13.0
CUDA Available        : True
Device Name           : NVIDIA GeForce RTX 3050 6GB Laptop GPU
VRAM                  : 5154.0 MiB free / 6143.5 MiB total (6.00 GiB)
Tensor Test           : [1, 2, 3] + [4, 5, 6] -> sum=21.0 on cuda:0 (PASSED)
FORCE_CPU Test        : Tensor executed on cpu: sum=21.0 on cpu (PASSED)
```

### Appendix B: Yashwant's Environment Diagnostic Report

- **Teammate:** Yashwant Vadhan M (HP Victus 16)
- **Operating System:** Windows 11 64-bit (AMD64)
- **Selected Acceleration Device:** `cuda:0` (CUDA GPU detected and active)
- **GPU Hardware Name:** NVIDIA GeForce RTX 3050 6GB Laptop GPU
- **VRAM Status:** 5210.0 MiB free / 6144.0 MiB total (6.00 GiB)

| Component / Library | Version / Status | Notes |
|---|---|---|
| **PyTorch (`torch`)** | `2.5.1+cu124` | Pinned PyTorch LTS wheel |
| **torchvision** | `0.20.1+cu124` | Pinned torchvision wheel |
| **scikit-learn** | `1.6.1` | Pinned scikit-learn release |
| **scikit-image** | `0.25.2` | Pinned scikit-image release |
| **OpenCV (`cv2`)** | `4.11.0.86` | Headless OpenCV build |
| **CUDA Available (`torch.cuda.is_available()`)** | `True` | Dedicated RTX 3050 mobile GPU active |
| **Tensor-on-Device Test** | `PASSED` | `[1, 2, 3] + [4, 5, 6] -> sum=21.0 on cuda:0` |
| **FORCE_CPU Override Test** | `PASSED` | With `FORCE_CPU=1`: `Selected Device: cpu`, `sum=21.0 on cpu` |

```text
=== Environment and Hardware Report (Yashwant) ===
OS                    : Windows 11 (10.0.26100)
Architecture          : AMD64
Python Version        : 3.11.9
Library Versions      : torch 2.5.1+cu124, torchvision 0.20.1+cu124, scikit-learn 1.6.1, scikit-image 0.25.2, cv2 4.11.0.86
CUDA Available        : True
Device Name           : NVIDIA GeForce RTX 3050 6GB Laptop GPU
VRAM                  : 5210.0 MiB free / 6144.0 MiB total (6.00 GiB)
Tensor Test           : [1, 2, 3] + [4, 5, 6] -> sum=21.0 on cuda:0 (PASSED)
FORCE_CPU Test        : Tensor executed on cpu: sum=21.0 on cpu (PASSED)
```

