# ADR 0003: SAM 2.1 Feasibility Spike (Memory, Latency & Viability)

- **Status:** Accepted
- **Date:** 2026-10-25
- **Driver:** Sushil
- **Navigator:** Yashwant
- **Task ID:** T1-007
- **References:** `PROJECT_OVERVIEW.md` §18.2, §23; `docs/PRD.md` Non-Functional Requirements; `docs/TECH_RULES.md` Rule 12 & Rule 16.

---

## 1. Context & Problem Statement

In hydroponic nutrient diagnosis, canopy images often capture whole plants with multiple overlapping leaves, background reservoir equipment, net pots, and support structures. Per `TECH_RULES.md` Rule 16:
> "Every image model consumes exactly one leaf. Training and inference use the same leaf-mask routine. Per-leaf predictions are never averaged into one score."

Segment Anything Model 2.1 (SAM 2.1) by Meta was selected as the foundation candidate for leaf-level instance segmentation. This spike measures real execution on the development laptop hardware (HP Victus 16 with NVIDIA GeForce RTX 3050 Laptop GPU, 6 GB VRAM) and CPU fallback to establish:
1. Exact checkpoint size, load times, and peak VRAM.
2. Latency comparison across GPU and CPU on sample images from all three crops (Lettuce, Spinach, Tomato).
3. Quality of leaf separation on whole-plant lettuce images comparing Automatic Mask Generation vs. Point/Box Prompts.
4. Viability of CPU-only deployment for campus lab machines.
5. Architectural options for Phase 5 (`ml/segmentation/`).

---

## 2. Tested Checkpoint and Environment

- **Architecture:** Segment Anything Model 2.1 (SAM 2.1 Hiera)
- **Model Size / Checkpoint:** `sam2.1_hiera_tiny.pt` (Smallest, most efficient foundation tier)
- **Model Checkpoint Path:** `models/image/sam2.1_hiera_tiny.pt`
- **Config File:** `configs/sam2.1/sam2.1_hiera_t.yaml`
- **Checkpoint File Size:** **148.78 MB** (155,999,755 bytes)
- **Hardware Platform:** HP Victus 16 Laptop (Windows 11 AMD64)
- **Dedicated GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (6,143.5 MiB / 6.00 GiB VRAM)
- **CUDA Runtime:** CUDA 11.8 / PyTorch 2.7.1

---

## 3. Measured Load Time & Peak VRAM

All numbers are measured using `scripts/benchmark_sam2.py` (not estimated):

| Execution Mode | Model Load Time | Peak VRAM Allocated | Total GPU Memory Headroom |
|---|:---:|:---:|:---:|
| **GPU (`cuda:0`)** | **0.732 s** | **2,316.0 MB (2.26 GiB)** | **3,827.5 MB free (62.3% headroom)** |
| **CPU (`cpu`)** | **0.409 s** | **0 MB (RAM only)** | **N/A** |

**Observation:** Peak VRAM consumption of 2.32 GiB fits comfortably within the 6 GB physical VRAM of the RTX 3050 Laptop GPU, leaving ample headroom (>3.8 GB) for concurrent classification inference (e.g. EfficientNetB0).

---

## 4. Latency Across Crops (GPU vs. CPU)

Latency benchmarked on 3 representative images per crop (9 images total) using the prompt predictor:

| Crop | Image Source / Resolution | GPU Latency (`cuda:0`) | CPU Latency (`FORCE_CPU=1`) | Speedup Factor |
|---|---|:---:|:---:|:---:|
| **Lettuce** | `Healthier_0001.jpg` (270x316) | 398.2 ms | 1,280.4 ms | 3.2x |
| **Lettuce** | `Healthier_0002.jpg` (288x293) | 441.5 ms | 1,295.1 ms | 2.9x |
| **Lettuce** | `Healthier_0003.jpg` (257x276) | 442.2 ms | 1,293.9 ms | 2.9x |
| **Spinach** | `Spinach_Healthy_0001.jpeg` (224x224) | 165.1 ms | 1,201.2 ms | 7.3x |
| **Spinach** | `Spinach_Healthy_0002.jpeg` (224x224) | 163.8 ms | 1,196.4 ms | 7.3x |
| **Spinach** | `Spinach_Healthy_0003.jpeg` (224x224) | 164.3 ms | 1,198.5 ms | 7.3x |
| **Tomato** | `Tomato_Healthy_0001.jpg` (256x256) | 182.4 ms | 1,195.1 ms | 6.5x |
| **Tomato** | `Tomato_Healthy_0002.jpg` (256x256) | 181.0 ms | 1,197.8 ms | 6.6x |
| **Tomato** | `Tomato_Healthy_0003.jpg` (256x256) | 181.4 ms | 1,197.8 ms | 6.6x |
| **Average** | **All Crops** | **257.8 ms** | **1,228.5 ms** | **4.8x** |

---

## 5. Leaf Separation: Automatic Masks vs. Point/Box Prompts

Tested on 3 whole-plant lettuce images to evaluate the resolution of the task's `⚠️ UNCLEAR`:

| Image | Mode | Masks Output | Latency | Separation Quality & Observations |
|---|---|:---:|:---:|---|
| `Healthier_0001.jpg` | **Auto Mask Generator** | 3 masks | 5,645.3 ms | Separated 3 dominant leaf clusters; background pot excluded. |
| `Healthier_0001.jpg` | **Point / Box Prompt** | 3 masks | 161.5 ms | High-confidence primary leaf (score 0.883); clear boundaries. |
| `Healthier_0002.jpg` | **Auto Mask Generator** | 4 masks | 5,083.8 ms | Identified 4 distinct foliage lobes; slight overlap at petioles. |
| `Healthier_0002.jpg` | **Point / Box Prompt** | 3 masks | 193.8 ms | High-confidence center leaf (score 0.807); tight boundary. |
| `Healthier_0003.jpg` | **Auto Mask Generator** | 2 masks | 4,993.6 ms | Merged two smaller touching leaves into one mask component. |
| `Healthier_0003.jpg` | **Point / Box Prompt** | 3 masks | 158.3 ms | Distinct single-leaf mask (score 0.877); no inter-leaf merging. |

### Evaluation Summary & Trade-offs
1. **Automatic Mask Generator (`SAM2AutomaticMaskGenerator`):**
   - **Pros:** Completely unsupervised; requires zero coordinate inputs from the user or upstream model.
   - **Cons:** High latency (~5.1 to 5.6 seconds per image on GPU; ~20-25 seconds on CPU). Prone to occasional under-segmentation (merging adjacent touching leaves) or over-segmentation (treating leaf veins/edges as separate masks).
2. **Prompt-Based Predictor (`SAM2ImagePredictor`):**
   - **Pros:** Extremely fast (160–195 ms on GPU, ~1.2 s on CPU); yields precise, clean boundaries for the targeted leaf; provides calibrated mask confidence scores (0.80–0.88).
   - **Cons:** Requires prompt coordinates (either user click on UI or an automated leaf-center peak detector / grid query).

---

## 6. Viability of CPU Fallback

- **Result:** **Viable.**
- Under `FORCE_CPU=1`, point/box prompt inference executes in **~1.2 seconds per image**, well within the PRD non-functional requirement target ($\le$ 30 s for image analysis on CPU).
- Automatic mask generation on CPU takes ~20–25 seconds; viable only if executed as an asynchronous background job with progress feedback.

---

## 7. Phase 5 Implementation Architecture Decision

Based on these measured findings, the team adopts the following strategy for Phase 5 (`ml/segmentation/`):

1. **Primary Foundation Model:**
   - Adopt `sam2.1_hiera_tiny.pt` as the primary foundation segmentation checkpoint.
   - Load onto GPU (`cuda:0`) when available (2.32 GB VRAM peak).
2. **Dual-Mode Segmentation Strategy:**
   - **Interactive / UI Upload Mode:** Use `SAM2ImagePredictor` with user click coordinates or automated bounding-box proposals (160 ms latency).
   - **Batch / Automatic Mode:** Use `SAM2AutomaticMaskGenerator` with minimum area threshold filtering (`min_mask_region_area=500`) to discard noise and background hardware.
3. **Fallback Options if SAM 2.1 is Unavailable:**
   - **Option A (Neural CPU Fallback):** `sam2.1_hiera_tiny.pt` on CPU using `SAM2ImagePredictor` (~1.2 s latency).
   - **Option B (Classical Computer Vision Fallback):** Otsu thresholding + HSV vegetation mask (ExG) + OpenCV morphological watershed contour separation (`< 15 ms` on CPU).
