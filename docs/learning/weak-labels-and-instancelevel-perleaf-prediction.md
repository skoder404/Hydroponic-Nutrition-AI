# Weak Labels and Instance-Level (Per-Leaf) Prediction

**Document ID:** `LEARN-ML-003`  
**Phase:** Phase 5 (Feature Extraction & Computer Vision)  
**Authors:** Sushil Kumar (Author) & Yashwant (Reviewer)  
**Task Reference:** `T5-012`  

---

## 1. Problem Formulation: Whole-Plant vs. Instance-Level Observation

In commercial and experimental hydroponic monitoring, imaging captures two distinct spatial regimes:
1. **Whole-Plant Imagery (Macro-Level / Canopy):** A single top-down or oblique photograph captures an entire plant specimen, comprising stem base, emerging crown leaflets, mature fan leaves, and senescent basal leaves within the hydroponic channel or net cup (e.g. `image_data/Lettuce_Images/`).
2. **Instance-Level Imagery (Leaf-Level / Micro-Level):** Individual leaf blades extracted from the canopy for focused morphological and chromatic inspection.

The agronomic diagnosis challenge stems from an inherent asymmetry:
Nutrient solutions and stress treatments are applied at the **system / plant level**, meaning the ground-truth experimental label (e.g., *Nitrogen Deficiency*, *Phosphorus Deficiency*, *Potassium Deficiency*, or *Healthier*) describes the **whole plant**. When automated instance segmentation (Meta SAM 2.1) extracts individual leaf instances from that plant, each leaf crop inherits the parent plant's diagnostic class label. This inherited label is fundamentally a **weak label**.

---

## 2. Agronomic Mechanism of Weak Label Noise

In plant physiology, nutrient deficiencies do not affect all leaves simultaneously or symmetrically. Nutrients exhibit varying degrees of phloem mobility inside the plant:

| Nutrient | Phloem Mobility | Symptom Progression | Weak Label Phenomenon |
| :--- | :--- | :--- | :--- |
| **Nitrogen ($N$)** | Highly Mobile | Mobile ions translocate from older, lower leaves to actively growing young crown leaves. Basal leaves exhibit chlorosis (yellowing) while apex leaves remain vibrant green. | Young leaves extracted from a severe $N$-deficient plant appear physiologically **normal / healthy**, yet inherit the `NitrogenDeficiency` label. |
| **Phosphorus ($P$)** | Highly Mobile | Symptoms first appear in mature foliage as dark green to purplish anthocyanin accumulation, with stunting. | Emerging leaves often show no noticeable anthocyanin discoloration, appearing healthy despite systemic $P$ starvation. |
| **Potassium ($K$)** | Highly Mobile | Symptoms manifest as marginal necrosis (leaf tip/edge burn) on older leaves first, progressing inward. | Inner wrapper leaves extracted from a $K$-deficient specimen may have intact margins and normal hue. |
| **Complete / Healthier** | Baseline | Uniform balanced nutrition throughout vegetative growth. | Minimal noise; nearly all extracted leaves present healthy green coloration and smooth contours. |

### Visual Discrepancy Rate
Because healthy-looking leaves on deficient plants are classified under the deficiency label, standard supervised loss functions penalize the model for recognizing healthy visual patterns on these instances. This label noise is intrinsic to whole-plant annotation and cannot be removed without instance-level agronomic ground truthing.

---

## 3. Data Integrity & Leakage Prevention Invariants

When transitioning from whole-plant images to leaf-level datasets:

### Invariant 1: Source Split Inheritance
All leaf crops derived from a single whole-plant image $I_k$ **must inherit the exact split assignment** (train, validation, or test) of $I_k$. 
$$\text{Split}(\text{Crop}_{k, j}) \equiv \text{Split}(I_k) \quad \forall j \in \{1, \dots, N_k\}$$
Under no circumstances may leaf crops from the same plant be distributed across both train and validation/test sets. Violating this rule causes severe **spatial data leakage**, yielding artificially inflated validation metrics.

### Invariant 2: Explicit Provenance Mapping
Each generated leaf crop is assigned a persistent, deterministic identifier (`<source_image_id>_crop_<index>`) and registered in `processed_data/lettuce_leaf_crops/source_map.csv`. The mapping records:
- `crop_id`: Unique leaf crop identifier
- `source_image_id`: Original source image stem
- `crop_path`: Filesystem path to masked RGB crop
- `class`: Inherited physiological class
- `split`: Inherited split partition (`train`, `val`, `test`)
- `mask_verdict`: Quality gate verdict (`OK`)

### Invariant 3: Zero-Crop Auditing
Whole-plant images that fail to produce any valid leaf instances (e.g. severe occlusion, camera blur, or mask rejection) are never silently dropped. They are explicitly logged in `processed_data/lettuce_leaf_crops/summary.json` with rejection reasons.

---

## 4. Inference & Aggregation Strategies for Weak Labels

At inference time, an operator photographs a hydroponic crop. The system segments the image into $M$ candidate leaves $\{\ell_1, \dots, \ell_M\}$ and extracts feature vectors $\{x_1, \dots, x_M\}$.

To overcome weak label noise during diagnosis, the inference service adopts **Multiple Instance Learning (MIL) Aggregation**:

1. **Instance Evaluation:** Each leaf $\ell_i$ is evaluated independently by the classifier, outputting a class probability distribution:
   $$P(y = c \mid x_i) \quad \text{for } c \in \mathcal{C}$$
2. **Uncertainty & Anomaly Gating:** Leaves whose prediction confidence falls below the calibrated uncertainty threshold or where symptoms are ambiguous are flagged as uncertain.
3. **Canopy-Level Voting / Pooling:**
   - **Soft Voting (Average Probability Pooling):**
     $$\bar{P}(c) = \frac{1}{M} \sum_{i=1}^M P(y = c \mid x_i)$$
   - **Deficiency Precedence (Max-Deficiency Heuristic):** In agronomic practice, if $2$ out of $5$ leaves exhibit severe, high-confidence potassium marginal necrosis, the plant is diagnosed with Potassium Deficiency even if the remaining 3 emerging leaves appear green.
4. **Transparent User Presentation:** The UI displays each segmented leaf as an individual card with its extracted bounding box, confidence score, and visual crop, allowing the grower to see exactly which leaf triggered the deficiency diagnosis.

---

## 5. Summary & Limitations

- **Supervised Training Ceiling:** Because roughly 15–25% of leaves from deficient plants exhibit normal green morphology, pure leaf-level classification accuracy on weakly-labeled datasets has a theoretical Bayes error floor.
- **Documentation Obligation:** Any deployment of leaf-level models trained on whole-plant manifests must state weak labeling as an explicit design limitation in the model card (`docs/models/` and Section 18.2 comparison reports).

