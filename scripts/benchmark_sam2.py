#!/usr/bin/env python3
"""Benchmark and feasibility spike for SAM 2.1 on hydroponic crop images (T1-007).

Measures:
1. Exact checkpoint size and load times (GPU & CPU).
2. Peak VRAM consumption during loading and inference.
3. Latency comparison (GPU vs. CPU) on 3 sample images per crop (Lettuce, Spinach, Tomato).
4. Leaf separation evaluation on 3 whole-plant lettuce images comparing:
   - Automatic Mask Generator (prompt-free)
   - Prompt-based Predictor (point/box prompt)
5. Outputs a detailed JSON report and summary.
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

CHECKPOINT_PATH = "models/image/sam2.1_hiera_tiny.pt"
CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_t.yaml"


def find_sample_images() -> dict[str, list[str]]:
    """Select 3 representative sample images for each crop."""
    samples = {}

    # Lettuce
    lettuce_imgs = sorted(glob.glob("image_data/Lettuce_Images/*/*.jpg"))
    if not lettuce_imgs:
        raise FileNotFoundError("No lettuce images found under image_data/Lettuce_Images")
    samples["lettuce"] = lettuce_imgs[:3]

    # Spinach
    spinach_imgs = sorted(glob.glob("image_data/Spinach_Images/*/*.jpeg"))
    if not spinach_imgs:
        spinach_imgs = sorted(glob.glob("image_data/Spinach_Images/*/*/*.jpeg"))
    if not spinach_imgs:
        raise FileNotFoundError("No spinach images found under image_data/Spinach_Images")
    samples["spinach"] = spinach_imgs[:3]

    # Tomato
    tomato_imgs = sorted(glob.glob("image_data/Tomato_Images/*/*.jpg"))
    if not tomato_imgs:
        tomato_imgs = sorted(glob.glob("image_data/Tomato_Images/*/*/*.jpg"))
    if not tomato_imgs:
        raise FileNotFoundError("No tomato images found under image_data/Tomato_Images")
    samples["tomato"] = tomato_imgs[:3]

    return samples


def benchmark_sam2() -> dict:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    results = {
        "checkpoint": CHECKPOINT_PATH,
        "config": CONFIG_PATH,
        "checkpoint_size_mb": round(os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024), 2),
        "cuda_available": torch.cuda.is_available(),
        "gpu_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }

    samples = find_sample_images()
    results["sample_images"] = samples

    # 1. GPU Loading & VRAM Benchmark
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        sam2_gpu = build_sam2(CONFIG_PATH, CHECKPOINT_PATH, device="cuda")
        t_gpu_load = time.perf_counter() - t0

        vram_after_load = torch.cuda.memory_allocated(0) / (1024 * 1024)
        peak_vram_load = torch.cuda.max_memory_allocated(0) / (1024 * 1024)

        results["gpu_load_time_s"] = round(t_gpu_load, 3)
        results["gpu_vram_after_load_mb"] = round(vram_after_load, 1)
        results["gpu_peak_vram_load_mb"] = round(peak_vram_load, 1)
    else:
        results["gpu_load_time_s"] = None
        results["gpu_vram_after_load_mb"] = None
        results["gpu_peak_vram_load_mb"] = None

    # 2. CPU Loading Benchmark
    t0 = time.perf_counter()
    sam2_cpu = build_sam2(CONFIG_PATH, CHECKPOINT_PATH, device="cpu")
    t_cpu_load = time.perf_counter() - t0
    results["cpu_load_time_s"] = round(t_cpu_load, 3)

    # 3. Predictor Latency Benchmark (GPU vs CPU) across all 9 sample images
    predictor_gpu = SAM2ImagePredictor(sam2_gpu) if torch.cuda.is_available() else None
    predictor_cpu = SAM2ImagePredictor(sam2_cpu)

    crop_latencies = {}
    for crop, img_paths in samples.items():
        crop_latencies[crop] = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img_rgb.shape

            # Center point prompt
            input_point = np.array([[w // 2, h // 2]])
            input_label = np.array([1])

            # GPU measurement
            gpu_time = None
            if predictor_gpu:
                t_start = time.perf_counter()
                predictor_gpu.set_image(img_rgb)
                _masks_g, _scores_g, _ = predictor_gpu.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                torch.cuda.synchronize()
                gpu_time = round((time.perf_counter() - t_start) * 1000, 1)

            # CPU measurement
            t_start = time.perf_counter()
            predictor_cpu.set_image(img_rgb)
            _masks_c, _scores_c, _ = predictor_cpu.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            cpu_time = round((time.perf_counter() - t_start) * 1000, 1)

            crop_latencies[crop].append({
                "path": img_path,
                "resolution": f"{w}x{h}",
                "gpu_latency_ms": gpu_time,
                "cpu_latency_ms": cpu_time,
            })

    results["predictor_latencies"] = crop_latencies

    # 4. Leaf Separation Comparison on 3 whole-plant lettuce images
    # Compare Automatic Mask Generator vs. Point/Box Prompts
    lettuce_eval = []
    auto_mask_gen_gpu = SAM2AutomaticMaskGenerator(sam2_gpu) if torch.cuda.is_available() else None
    auto_mask_gen_cpu = SAM2AutomaticMaskGenerator(sam2_cpu)

    for l_path in samples["lettuce"]:
        img = cv2.imread(l_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        # A) Automatic Mask Generator
        t0 = time.perf_counter()
        if auto_mask_gen_gpu:
            masks_auto = auto_mask_gen_gpu.generate(img_rgb)
            torch.cuda.synchronize()
        else:
            masks_auto = auto_mask_gen_cpu.generate(img_rgb)
        auto_time_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Compute mask statistics
        mask_counts = len(masks_auto)
        areas = [m["area"] for m in masks_auto]
        avg_area = round(float(np.mean(areas)), 1) if areas else 0
        min_area = int(np.min(areas)) if areas else 0
        max_area = int(np.max(areas)) if areas else 0

        # B) Point/Box Prompt Predictor
        t0 = time.perf_counter()
        pred = predictor_gpu or predictor_cpu
        pred.set_image(img_rgb)
        # 3 distinct point prompts across upper-left, center, lower-right
        pts = np.array([[w // 3, h // 3], [w // 2, h // 2], [2 * w // 3, 2 * h // 3]])
        labels = np.array([1, 1, 1])
        prompt_masks, prompt_scores, _ = pred.predict(
            point_coords=pts,
            point_labels=labels,
            multimask_output=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prompt_time_ms = round((time.perf_counter() - t0) * 1000, 1)

        lettuce_eval.append({
            "image": l_path,
            "resolution": f"{w}x{h}",
            "auto_masks": {
                "count": mask_counts,
                "latency_ms": auto_time_ms,
                "avg_area_px": avg_area,
                "area_range_px": [min_area, max_area],
            },
            "prompt_masks": {
                "count": len(prompt_masks),
                "latency_ms": prompt_time_ms,
                "scores": [round(float(s), 3) for s in prompt_scores],
            },
        })

    results["lettuce_leaf_separation"] = lettuce_eval

    if torch.cuda.is_available():
        results["gpu_peak_vram_total_mb"] = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 1)

    return results


def main():
    print("=== Running SAM 2.1 Feasibility Benchmark ===")
    results = benchmark_sam2()

    out_file = Path("reports/sam2_feasibility_metrics.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved to {out_file}")

    # Print summary
    print("\n--- Summary of Measurements ---")
    print(f"Checkpoint           : {results['checkpoint']} ({results['checkpoint_size_mb']} MB)")
    print(f"GPU Load Time        : {results['gpu_load_time_s']} s")
    print(f"CPU Load Time        : {results['cpu_load_time_s']} s")
    if results['cuda_available']:
        print(f"Peak VRAM during run : {results['gpu_peak_vram_total_mb']} MB ({results['gpu_device_name']})")

    print("\nPer-Image Latency (Point Prompt):")
    for crop, items in results["predictor_latencies"].items():
        avg_gpu = np.mean([x["gpu_latency_ms"] for x in items if x["gpu_latency_ms"] is not None])
        avg_cpu = np.mean([x["cpu_latency_ms"] for x in items])
        print(f"  {crop.capitalize():<8}: GPU avg = {avg_gpu:.1f} ms | CPU avg = {avg_cpu:.1f} ms")

    print("\nLettuce Leaf Separation (Automatic Masks vs Prompts):")
    for item in results["lettuce_leaf_separation"]:
        print(f"  {Path(item['image']).name} ({item['resolution']}):")
        print(f"    Auto Masks   : {item['auto_masks']['count']} masks in {item['auto_masks']['latency_ms']} ms")
        print(f"    Prompt Masks : {item['prompt_masks']['count']} masks in {item['prompt_masks']['latency_ms']} ms (scores: {item['prompt_masks']['scores']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
