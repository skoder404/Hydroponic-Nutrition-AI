# Dataset Provenance: [Dataset Name]

- **Document Version:** 1.0.0
- **Author:** [Author Name]
- **Related Tasks:** [Task IDs, e.g. T2-002, T2-006]
- **Target Crop:** [Lettuce | Spinach | Tomato]
- **Modality:** [Image | Sensor Time-Series | Multimodal]
- **Last Verified Date:** [YYYY-MM-DD]

---

## 1. Source Identification & Origin

- **Source Name:** [e.g., Kaggle Lettuce NPK Dataset / HydroGrowNet Batavia / USDA Ag Data Commons]
- **Source URL:** [https://...]
- **Source Version / Date Downloaded:** [e.g., Version 1, downloaded 2026-10-24]
- **Original Filename(s):** [List original archive or raw filenames, e.g. EXP_1.xlsx, archive.zip]
- **Current Repository Location:** [e.g., `image_data/Lettuce_Images/` or `sensor_data/Spinach_Sensor-Readings/`]

---

## 2. License, Terms & Redistribution Status

- **License Type:** [e.g., CC BY 4.0, CC0 Public Domain, MIT, Non-commercial, Academic only, or Unspecified]
- **Original Terms / Rights Holder:** [Attribution details and copyright owner]
- **Redistribution Status:** [e.g., Permitted under CC BY 4.0 / Local-only: raw binaries git-ignored / Redistribution restricted]
- **Commercial / Non-Commercial Constraints:** [e.g., Academic/Coursework use only]

---

## 3. Sample Counts & Volume

| Metric Stage | Sample Count (Rows / Images) | Notes / Filtering Rationale |
|---|---|---|
| **Original Raw Count** | [Count] | Raw instances before cleaning or filtering |
| **Filtered / Dropped Count** | [Count] | Dropped due to nulls, invalid values, or corruption |
| **Prepared Final Count** | [Count] | Retained count in processed dataset |

---

## 4. Class Labels & Distribution

| Class Label | Raw Count | Prepared / Balanced Count | Prevalence (%) | Diagnostic Definition |
|---|---|---|---|---|
| `Class_A` | [Count] | [Count] | [0.0%] | [Semantic meaning] |
| `Class_B` | [Count] | [Count] | [0.0%] | [Semantic meaning] |

---

## 5. Preprocessing & Transformation Pipeline

- **Preprocessing Script & Version:** [e.g., `scripts/build_lettuce_dataset.py` (v1.0.0)]
- **Cleaning & Imputation Steps:** [Detail dropping rules, filtering steps, missing-value strategy]
- **Normalization / Scaling:** [Detail scaling applied: min-max, standard, or unscaled]
- **Augmentation Details (if applicable):** [Training-only augmentation transforms or state 'None']

---

## 6. Partitioning & Split Protocol

- **Split Protocol:** [e.g., Grouped by calendar day / 6-hour time-block holdout / Plant-wise grouping]
- **Split Rationale:** [How temporal or spatial leakage was prevented]
- **Manifest Location:** [e.g., `processed_data/manifests/split_manifest.csv`]
- **Split Breakdown:**
  - **Train:** [Count] rows/images ([XX]%)
  - **Validation:** [Count] rows/images ([XX]%)
  - **Test:** [Count] rows/images ([XX]%)

---

## 7. Synthetic, Derived, Rescaled & Rule-Generated Values

<!--
Explicitly detail any value that was not a direct physical physical sensor measurement
or natural crop photograph.
-->

- **Synthetic Elements:** [Detail any simulated noise, generative values, or synthetic solution metrics]
- **Derived / Trend Features:** [Detail calculated deltas, rolling means, vegetation indices]
- **Rule-Generated Labels:** [Detail the specific threshold rules, conditions, or logic used to assign labels]
- **Rescaling / Adjustments:** [Detail unit conversions (e.g. soil EC to hydroponic mS/cm)]

---

## 8. Column / Field Taxonomy Table

Every column/feature in this dataset is tagged with one of the five approved provenance tags:
- `measured`: Direct, unmodified physical sensor or hardware reading
- `adjusted`: Physical reading calibrated, offset-corrected, or unit-converted
- `derived`: Calculated mathematically from other fields (e.g., ratios, differences, rolling averages)
- `synthetic`: Artificially simulated or mathematically generated without direct observation
- `rule-generated`: Categorical or discrete label assigned by evaluating conditional rule thresholds

| Column Name | Data Type | Provenance Tag | Unit / Value Range | Description & Derivation Formula |
|---|---|:---:|---|---|
| `timestamp` | `datetime` | `measured` | ISO 8601 | Physical hardware logger timestamp |
| `ph` | `float` | `measured` | 0.0 – 14.0 | Direct potentiometric pH electrode reading |
| `ec_mS_cm` | `float` | `adjusted` | mS/cm | Electrical conductivity calibrated for temperature |
| `ph_delta` | `float` | `derived` | pH units | Difference: `ph[t] - ph[t-1]` |
| `simulated_water_temp`| `float` | `synthetic` | °C | Synthesized diurnal water temperature profile |
| `label` | `string` | `rule-generated` | Category | Assigned based on threshold rules: e.g. `ph < 5.5` |

---

## 9. Known Limitations & Unsupported Interpretations

- **Biological Validity:** [e.g., Rule-derived sensor labels indicate threshold triggers, not biological plant pathology]
- **Sensor Calibration & Drift:** [Known sensor noise, probe drift, missing calibration logs]
- **Domain Shifts:** [Indoor controlled greenhouse vs. outdoor variable climate]
- **Unpaired Modality Caution:** [Reiterate Overview §19: this dataset cannot be paired with other modalities without direct plant-level synchronization]
- **Unsupported Claims:** [Claims that MUST NOT be made based on models trained on this dataset]
