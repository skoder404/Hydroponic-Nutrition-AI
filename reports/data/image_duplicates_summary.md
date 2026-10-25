# Image Duplicate Audit Report (T2-014)

**Generated:** 2026-10-25T15:42:47.891598+00:00
**Script:** `scripts/find_duplicates.py`
**Safety Policy:** Non-destructive audit report only. No image files were modified or deleted.

## 1. Methodology, Libraries & Thresholds

- **Exact Duplicates:** `SHA-256` digest on raw file bytes.
- **Near-Duplicates:** Pillow (`PIL.Image`) difference hash (`dHash`, 64-bit gradient).
- **Perceptual Distance Threshold:** Hamming distance **≤ 4** bits (out of 64).
- **Threshold Rationale:** Conservative threshold chosen to capture identical/burst-shot photographs and slight compression differences without erroneously matching distinct plant specimens.
- **Search Optimization:** Multi-index inverted table (5 sub-fingerprint blocks) guaranteeing 100% recall.

## 2. Summary of Findings by Crop

| Crop | Total Images | Exact Groups | Exact Files | Near Pairs (≤4) | Near Clusters | Cross-Class Near Pairs |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `lettuce` | 580 | 0 | 0 | 211 | 31 | 1 |
| `spinach` | 18,302 | 124 | 248 | 19,060 | 394 | 13,474 |
| `tomato` | 33,564 | 1082 | 2164 | 5,142 | 2,699 | 513 |

## 3. Analysis & Implications for ML Splits

1. **Exact Duplicates:**
   - Identical byte-for-byte copies must be isolated so they never appear on both sides of a train/test split.
2. **Near-Duplicates & Burst Capture Contamination:**
   - Near-duplicates indicate multiple frames photographed from the same plant head or specimen.
   - If near-duplicates leak across train and test sets, model accuracy is artificially inflated.
   - Splits must be grouped by plant specimen/session rather than naive random sampling.
3. **Cross-Class Near-Duplicates:**
   - Cross-class near pairs highlight potential label noise or ambiguous border conditions between deficiency stages.

---
*Detailed per-crop JSON reports available at `reports/data/image_duplicates_{crop}.json`.*