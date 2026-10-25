# ML Fusion Directory Policy: No Unpaired Fusion

## Core Rule (Overview §19, TECH_RULES §15)

**No fusion model may be trained on unpaired image and sensor data.**

---

## 1. Context and Rationale

In this hydroponic system, image datasets and sensor datasets represent fundamentally different data-generation processes:
1. **Unpaired Observations:** The raw image files (under `image_data/`) and time-series sensor files (under `sensor_data/`) were collected independently without joint, paired time-synchronization or 1-to-1 plant identity linkages.
2. **Distinct Taxonomies and Modalities:** The visual deficiency/disease classes and the sensor environmental/nutrient state classes operate at different semantic and diagnostic granularities.
3. **Anti-Hallucination Mandate:** Synthesizing artificial pairings or forcing multimodal fusion between unrelated image and sensor rows introduces severe artificial correlations, data leakage, and misleading model diagnostics.

---

## 2. Operating Policies (PROJECT_OVERVIEW.md §19)

1. **Independent Predictions:** The image diagnosis model and sensor diagnosis model must execute and output their predictions independently.
2. **Metadata Preservation:** Store each prediction's modality (`image` or `sensor`), timestamp, input provenance, calibrated score/confidence, and model version separately.
3. **Strict Prohibition on Unpaired Fusion:** Do not train early-fusion, mid-fusion, or late-fusion neural networks or ensembles using unaligned image and sensor records.
4. **Validated Mappings Only:** Outputs from both modalities may only be comparatively displayed using an explicitly documented, semantically justified, and reviewed mapping table.
5. **Separate Reporting:** If no validated semantic mapping applies, show image and sensor inferences side-by-side as separate diagnostic evidence channels, clearly communicating that they are not directly comparable.
6. **Conservative Alerts:** Flag a mismatch only when a reviewed domain rule makes the comparison meaningful. Never turn an arbitrary image/sensor class difference into an alert or contradiction.
7. **Graceful Single-Modality Fallback:** If one modality is unavailable (e.g., no camera frame or offline sensors), proceed with the available modality and transparently disclose what was not assessed.
8. **Prerequisites for Future Fusion Models:** If a paired multimodal dataset is collected in the future, it must rigorously document:
   - Specific plant identity (tracking individual plant throughout lifecycle)
   - Synchronized physical timestamps
   - Shared crop and cultivar specifications
   - Controlled growing conditions
   - Explicit pairing contracts reviewed before any fusion model architecture is trained.
