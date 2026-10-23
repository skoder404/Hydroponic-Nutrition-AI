# Hydroponic Nutrition AI — Canonical Project Overview

> **Document role:** This file is the canonical source of truth for the Hydroponic Nutrition AI project. It is intended to provide complete project context to developers, teammates, reviewers, and AI coding assistants so they do not repeatedly ask already-resolved questions or silently invent decisions.
>
> **Precedence:** Follow this document for project intent and documented defaults. For implementation-specific facts, the actual source code, tests, generated dataset manifests, trained artifacts, and current evaluation reports take precedence over assumptions in prose. When code and this document disagree, inspect the evidence and update this document deliberately.
>
> **Truth labels used throughout**
>
> - **Confirmed/documented:** explicitly established in the project notes or available training report.
> - **Planned default:** the selected intended behaviour; do not claim it is implemented until verified.
> - **Must verify in repository:** the actual implementation, script, dataset, test, or artifact must be inspected before an exact claim is made.
> - **Limitation:** a known caveat that must remain visible in documentation and the UI where relevant.
>
> This document includes decisions and context available in the project discussion. It does **not** pretend that source files unavailable for inspection have been inspected.

---

## 1. Executive summary

Hydroponic Nutrition AI is a crop-condition decision-support application for **lettuce, spinach, and tomato**. It is intended to combine plant-image analysis, sensor-based condition classification, and cautious recommendations in a web application.

The project explores multiple machine-learning approaches rather than assuming one algorithm is best:

1. **Image classification:** a custom CNN, EfficientNetB0 transfer learning, and OpenCV feature extraction followed by traditional machine-learning classifiers.
2. **Image segmentation:** SAM 2.1 as the initial segmentation approach for crop/leaf isolation, with mask validation and retry/failure handling.
3. **Sensor classification:** Decision Tree, Random Forest, SVM, and other suitable classifiers evaluated against each crop's exact input schema.
4. **Evidence presentation/fusion:** image and sensor outputs are preserved independently because the datasets are generally not paired and the label taxonomies differ.
5. **Recommendations:** a Gemini API-assisted recommendation layer with a deterministic rule-based fallback.

The application must communicate uncertainty. A model output is a **prediction**, not a confirmed diagnosis. A rule-based label is not a directly observed plant-health outcome. A recommended action is advisory and must not be represented as an automatically executed action.

## 2. Product objective and scope

### 2.1 Main objective

Provide a clear workflow in which a user can:

- choose the crop: lettuce, spinach, or tomato;
- submit a plant image for image-based classification;
- enter or receive sensor readings for sensor-based classification;
- view model predictions, confidence/uncertainty, and the evidence used;
- receive understandable, conservative suggestions for further checks or corrective action;
- view historical records with timestamps;
- understand when the model is uncertain, the input is invalid, the sensor values are unavailable, or the model is not validated for the requested use.

### 2.2 In scope

- Three crop families: lettuce, spinach, and tomato.
- Image-based classification and feature-based classification experiments.
- Crop/leaf segmentation and preprocessing.
- Sensor condition classification using crop-specific schemas.
- Separate display of image and sensor findings.
- Cautious recommendation generation using Gemini with rule-based fallback.
- Timestamped history and a database-backed application.
- A modular campus/device integration path with manual-entry fallback.
- Model versioning, reproducible preprocessing, evaluation reports, and tests.
- Local model artifacts and image files during the initial implementation.

### 2.3 Out of scope unless explicitly approved later

- Claiming medical- or laboratory-grade diagnosis of plant health.
- Treating synthetic or threshold-derived labels as independent biological ground truth.
- Automatically applying nutrient solutions or controlling pumps/actuators.
- Assuming every sensor is installed or available at the campus.
- Assuming image and sensor observations describe the same plant at the same time.
- Assuming all data sources have compatible licenses for redistribution.
- Adding login/authentication to the first version; **authentication is not required initially**.
- Assuming a cloud object-storage service is configured.
- Promising a particular production accuracy before the final evaluation is complete.

## 3. Core product behaviour

1. The user chooses a crop.
2. The user provides an image, sensor readings, or both.
3. The backend validates input format, schema, units, ranges, and required values.
4. The image path applies the preprocessing expected by the selected model. Where required, SAM 2.1 segments the plant or leaf.
5. Segmentation masks are checked before downstream use. A poor or failed mask produces a clear warning and a retry path rather than silently continuing with an invalid crop.
6. The image model returns a crop-specific class and confidence/uncertainty information.
7. The sensor model receives only the expected ordered features after the same transformations used during training.
8. The application stores the modality outputs independently, including timestamps, provenance, model version, and confidence where available.
9. The interface displays image and sensor results separately unless a reviewed and validated mapping makes direct comparison meaningful.
10. The recommendation layer explains what evidence informed its advice. It uses deterministic fallback rules when Gemini is unavailable or returns unusable output.
11. If an input is missing, unsupported, stale, invalid, or outside the model's documented domain, the system explains the limitation and does not invent a value.
12. Historical readings must show their observation timestamp and must never be labelled as current/live unless they were freshly acquired and validated.

## 4. Default technology stack

These are the intended defaults, not a claim that every component has already been integrated.

| Layer | Default | Notes |
|---|---|---|
| Frontend | Next.js + React | Nature-themed, clear, responsive UI |
| Backend/API | FastAPI | Input validation and inference endpoints |
| Application database | PostgreSQL | Store records and timestamped history |
| Deep learning | PyTorch | Custom CNN and EfficientNetB0 |
| Traditional ML | scikit-learn | Sensor classifiers and OpenCV-feature classifiers |
| Segmentation | SAM 2.1 | Initial segmentation choice across crops; validate masks |
| Recommendation API | Gemini API | Advisory generation only; deterministic fallback required |
| Initial file storage | Local files/artifacts | Do not assume cloud object storage |
| GPU target | HP Victus 16 NVIDIA RTX 3050 | Use only when the installed runtime supports it |
| CPU fallback | CPU inference/training where feasible | Graceful fallback and resource-aware behaviour |
| Authentication | None initially | Do not add login without an explicit scope change |
| Campus/device integration | Modular adapter | Manual-entry fallback if hardware is unavailable |
| Configuration | `.env` + `.env.example` | Real secrets excluded from version control |
| Tests | Layered automated tests | Unit, integration, contract, and inference checks |

### 4.1 Suggested monorepo layout

Adapt this proposal to the existing repository. Do not create duplicate or parallel structures if equivalent folders already exist.

```text
frontend/
  app/ or src/
  components/
  lib/
  public/
backend/
  app/
    api/
    schemas/
    services/
    database/
    core/
ml/
  image_models/
  sensor_models/
  segmentation/
  features/
  fusion/
  recommendations/
  evaluation/
  registry/
datasets/
  raw/
  images/
  sensors/
  processed/
models/
  image/
  sensor/
  manifests/
scripts/
tests/
  unit/
  integration/
  contracts/
docs/
.env.example
.gitignore
PROJECT_OVERVIEW.md
README.md
```

**Storage rule:** raw datasets should be preserved. Generated features, processed datasets, trained artifacts, masks, and evaluation outputs should be distinguishable from raw data. Do not overwrite the only copy of source data.

## 5. User interface and application behaviour

### 5.1 Visual direction

- Use a nature-inspired visual style appropriate for hydroponics and plant care.
- Make crop selection obvious.
- Use readable class names and plain-language explanations.
- Clearly distinguish image results, sensor results, recommendations, uncertainty, and historical records.
- Avoid presenting a low-confidence output with the same visual certainty as a high-confidence output.
- Provide meaningful loading, empty, error, retry, and unavailable-device states.
- Do not imply a device is connected or readings are live unless the integration confirms it.

### 5.2 Initial workflow

1. Select crop.
2. Select analysis mode: image, sensors, or both.
3. Upload an image and/or enter readings.
4. Validate input and explain any missing or invalid field.
5. Run the supported model pipeline.
6. Display each modality's prediction and its evidence.
7. Show a cautious recommendation or a clear explanation that the system cannot recommend confidently.
8. Optionally save a timestamped record.
9. Allow the user to review historical records with their actual timestamps.

### 5.3 Image input distinction

- Lettuce CNN/EfficientNetB0: source data consists of whole-plant images.
- Lettuce OpenCV feature-based pipeline: individual leaf crops are expected.
- Spinach and tomato whole-plant uploads must be segmented/cropped into the input form expected by the selected model.
- Do not mix whole-plant and isolated-leaf images without an explicit, tested preprocessing path.
- Store or record enough preprocessing metadata to reproduce an inference.

### 5.4 Sensor input and live-device distinction

- A missing reading is missing; it is not zero.
- Do not fabricate readings when a sensor or campus device is unavailable.
- Validate units and field names before inference.
- Display timestamps and data source.
- A historical record must not be displayed as a live reading.
- Tomato sensor inference has special limitations because the prepared data uses transformed values and a synthetically derived water-temperature feature. Do not claim that arbitrary live campus readings are compatible until the mapping is documented and validated.
- If compatibility is not demonstrated, present tomato sensor use as experimental/manual assessment rather than validated live inference.

## 6. Crop and class taxonomy

The model's internal class labels and the UI's display mapping must be maintained explicitly. Never infer a class from a filename alone at inference time.

### 6.1 Lettuce image classes

Internal/source classes:

- `Healthy`
- `Nitrogen_Deficient`
- `Phosphorus_Deficient`
- `Potassium_Deficient`

Display mapping:

- `Healthy` → `healthy`
- each deficiency class → `unhealthy/<original class>`

### 6.2 Spinach image classes

Internal/source classes:

- `Spinach_Healthy`
- `Spinach_anthracnose`
- `Spinach_bacterial spot`
- `Spinach_downy mildew`
- `Spinach_Non fresh`
- `Spinach_pest damage`

Display mapping:

- `Spinach_Healthy` → `healthy`
- every other class → `unhealthy/<original class>`

Preserve the exact source labels in metadata even if a more readable label is shown in the UI. Normalization must be deterministic and documented.

### 6.3 Tomato image classes

Internal/source classes:

- `healthy`
- `Tomato_Bacterial_spot`
- `Tomato_Early_blight`
- `Tomato_Late_blight`
- `Tomato_Leaf_Mold`
- `Tomato_Septoria_leaf_spot`
- `Tomato_Spider_mites_Two-spotted_spider_mite`
- `Tomato_Target_Spot`
- `Tomato_Tomato_mosaic_virus`
- `Tomato_Tomato_Yellow_Leaf_Curl_Virus`

Display mapping:

- `healthy` → `healthy`
- each disease class → `unhealthy/<original class>`

### 6.4 Sensor classes are not image classes

Sensor labels describe rule-defined conditions or stress categories. Image labels describe visible classes in the relevant image dataset. These taxonomies are not interchangeable. Do not map `EC_High_Toxicity`, for example, to a specific image disease without a validated semantic justification.

## 7. Image model strategy

### 7.1 Models to compare

For each crop where the dataset and input format support a fair comparison, compare:

1. **Custom CNN** trained on the available image dataset.
2. **EfficientNetB0 transfer learning**.
3. **OpenCV feature extraction + traditional ML**, using features appropriate to the crop and classifiers such as Random Forest and SVM.

Use PyTorch for the custom CNN and EfficientNetB0 implementation. Use scikit-learn for traditional classifiers. Do not claim that all model families have been trained for every crop unless the artifacts and evaluation reports confirm that.

### 7.2 Fair comparison requirements

- Use the same documented train/validation/test assignments for comparable experiments.
- Ensure input format and image preprocessing are defined for each model.
- Record image size, normalization, color-channel convention, segmentation configuration, and augmentation settings.
- Fit any learned preprocessing only on training data.
- Use augmentation only on training data.
- Do not let validation/test data enter augmentation pipelines or feature-fitting stages.
- Report class support and class-wise metrics; overall accuracy alone is insufficient.
- Select the preferred model using validation macro-F1 unless an explicitly documented, justified exception is approved.
- Use the untouched test set for final reporting only.

### 7.3 Transfer learning defaults

EfficientNetB0 is the planned transfer-learning baseline. The exact pretrained weights, input size, layer-freezing schedule, optimizer, learning rate, batch size, epochs, early stopping, and augmentation must be recorded from the actual training configuration. This document does not invent those values.

### 7.4 Confidence and uncertainty

- Return a confidence score only if its meaning is documented.
- Raw softmax probability must not automatically be described as calibrated real-world probability.
- If calibration is used, fit it using training/validation data, not the test set.
- Derive uncertainty thresholds from validation data.
- Low-confidence or out-of-domain predictions should be labelled **uncertain** and should not drive a strong recommendation.
- A segmentation failure or invalid image should be reported as an input-processing failure, not as a disease prediction.

## 8. Segmentation policy

SAM 2.1 is the initial segmentation approach for all crops, subject to compatibility and quality checks.

### 8.1 Required behaviour

- Check that a mask exists and is non-empty.
- Check that the mask area and geometry are plausible for the image.
- Avoid treating the background as the leaf/plant.
- If the mask is poor or segmentation fails, display a clear warning and allow retry.
- Do not silently continue with a failed mask.
- Keep segmentation parameters/version and relevant preprocessing metadata with the output.
- Use the same segmentation/preprocessing contract during training and inference for any model trained on segmented inputs.
- Save masks or reproducible references where useful and permitted by storage constraints.

### 8.2 Crop-specific input contracts

- Lettuce CNN/EfficientNetB0 uses whole-plant images from its source dataset.
- Lettuce OpenCV feature extraction uses individual leaf crops.
- Spinach and tomato whole-plant images require segmentation/cropping before a leaf-oriented feature extractor is used.
- The exact mask selection policy (largest component, multiple leaves, manual selection, etc.) must be implemented and documented in the repository; do not silently assume it.

## 9. Dataset governance and common rules

### 9.1 Dataset provenance

For each source, record:

- source name and URL;
- source version/date downloaded where known;
- original filename(s);
- license/terms and redistribution status;
- original row/image count;
- prepared row/image count;
- class labels and class counts;
- preprocessing script/version;
- train/validation/test split protocol and assignment;
- any synthetic, derived, rescaled, or rule-generated values;
- known limitations and unsupported interpretations.

A dataset being downloaded and used does not automatically verify its provenance, units, or license.

### 9.2 Raw, processed, and generated data

- Keep raw source data unchanged.
- Store prepared data separately.
- Keep feature extraction output separate from source images.
- Keep split manifests so the same examples remain in the same split across experiments.
- Do not overwrite prepared sensor splits.
- When a feature definition changes, regenerate the affected feature data and retrain dependent models.
- Record row counts and class counts before and after every major preprocessing stage.
- Investigate unexplained changes in counts rather than silently accepting them.

### 9.3 Leakage prevention

- Check exact duplicates and near-duplicates before splitting image datasets.
- All leaf crops from the same original lettuce plant image must remain in one split.
- Keep augmentation limited to training examples.
- Do not use target-derived fields as features.
- Do not use identifiers, timestamps, bench names, or source indicators when they leak the target or create unrealistic shortcuts.
- Preserve the documented time-based/group-based sensor split protocols.
- Do not choose thresholds, features, or hyperparameters based on final test performance.

### 9.4 Dataset labels and interpretation

Sensor datasets described below use rule-generated labels, unless an individual source explicitly proves otherwise. These labels indicate which threshold rule fired; they do not independently establish plant disease or nutrient status. Balanced synthetic class distributions do not represent natural field prevalence.

## 10. Lettuce image dataset

**Source:** Kaggle, “Lettuce NPK Dataset”  
URL: https://www.kaggle.com/datasets/baronn/lettuce-npk-dataset?resource=download

### 10.1 Documented classes

- `Healthy`
- `Nitrogen_Deficient`
- `Phosphorus_Deficient`
- `Potassium_Deficient`

### 10.2 Input type

The source images are whole-plant images. They are suitable for the whole-plant CNN/EfficientNet pipeline, subject to the actual preprocessing contract. OpenCV feature-based lettuce classification is based on individual leaf crops.

### 10.3 Split policy

- Target image split: 70% train, 15% validation, 15% test, stratified where appropriate.
- Deduplicate/near-duplicate-check before splitting.
- Every leaf crop generated from one original image must inherit that original image's split.
- Do not split sibling crops from the same source image across train, validation, and test.
- If an existing split manifest already exists, inspect and preserve it unless there is an explicit, reviewed reason to rebuild it.

### 10.4 Provenance caution

The source URL is documented. The exact dataset version, license conditions, original counts, and final split counts must be recorded in a dataset manifest before a complete reproducibility claim is made.

## 11. Lettuce sensor dataset

**Source:** HydroGrowNet, Mendeley Data, DOI `10.17632/g6cm3v3wdp.5`  
URL: https://data.mendeley.com/datasets/g6cm3v3wdp/5

### 11.1 Source context

The available sensor training report describes Batavia lettuce grown in an NFT (Nutrient Film Technique) hydroponic setup across three consecutive 30-day experiments (EXP.1, EXP.2, EXP.3) between March and June 2024.

The report documents:

- approximately 2,069 usable hourly air-logger rows;
- 726 real portable-meter readings containing water chemistry;
- the source sheet named “Water quality measurments Senso” had pH/EC/TDS/water-temperature headers but no populated values for those measurements;
- the populated portable-meter sheet contained handheld readings taken at 10:00 and 14:00 across six replicate tanks;
- air data was joined only at an exact matching hour where possible, yielding 690 exact matches out of 726 water readings;
- expanding the dataset by repeating water chemistry across all hourly air rows was rejected because it would multiply rows without adding real chemistry observations.

### 11.2 Input schema and target

Exact model inputs, in the documented order:

1. `ph`
2. `ec`
3. `water_temp`
4. `humidity`
5. `air_temp`

Target: `label`

The feature order and spelling are part of the model contract. Validate them in training and inference code.

### 11.3 Data cleaning documented in the report

- Dates with day/month swapped were corrected using the known experiment window.
- Relative-humidity and air-temperature columns were read by header and corrected where a header contradicted its values.
- Impossible readings were blanked rather than replaced: a pH reading of 2.0 and an EC reading of 26.7 were set to missing.
- No sensor reading was deliberately changed merely to make it look more plausible.
- Air readings were joined only at exact matching hours where possible.

The exact cleaning implementation should be checked against the preprocessing code if that code is available.

### 11.4 Rule-derived labels

The report explicitly says the labels are rule-derived, not observed plant-health outcomes. Documented tolerant normal bands:

| Variable | Normal band |
|---|---:|
| pH | 5.5–7.0 |
| EC | 1.2–2.5 mS/cm |
| Water temperature | 18–26 °C |
| Relative humidity | 50–80% |
| Air temperature | 18–30 °C |

Documented label logic:

- all five variables within range → `Normal`;
- exactly one variable outside range → the corresponding stress class;
- two or more variables outside range → `Multiple_Stress`;
- missing variables are skipped under the documented preprocessing logic;
- rows with all five inputs missing are removed.

The exact script remains authoritative for edge cases such as how partially missing rows are counted, how classes are named, and how boundaries are implemented.

### 11.5 Rare classes omitted

The original real-data class counts in the available report were:

| Class | Original real rows |
|---|---:|
| `Normal` | 327 |
| `Multiple_Stress` | 135 |
| `Humidity_Low` | 87 |
| `WaterTemp_High` | 75 |
| `EC_High` | 58 |
| `pH_High` | 25 |
| `pH_Low` | 10 |
| `WaterTemp_Low` | 8 |
| `EC_Low` | 1 |

`WaterTemp_Low` and `EC_Low` were omitted from the seven-class model because there were too few original examples to support a meaningful class. The remaining seven classes were used.

The training file described in the report contains 1,680 rows after oversampling 558 real training rows to 240 rows per class across seven classes. The held-out test file contains 162 real rows and was not oversampled. The report says every fourth calendar day of each experiment was held out, preventing a day from appearing in both train and test.

**Do not replace these prepared splits with a random split.** If current files differ, report the discrepancy and investigate.

### 11.6 Reported baseline results

The following are reported results from the available training report, not guaranteed production performance. The single test set contained 162 real rows.

| Model | Train accuracy | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| Decision Tree | 1.000 | 0.994 | 0.986 |
| Random Forest | 1.000 | 0.994 | 0.986 |
| SVM | 0.955 | 0.864 | 0.806 |
| Logistic Regression | 0.835 | 0.772 | 0.716 |
| KNN | 0.974 | 0.741 | 0.665 |

Random Forest tuning in the report used GridSearchCV with three-fold cross-validation, scored on macro-F1, over `n_estimators`, `max_depth`, and `min_samples_leaf`. The reported setting was `n_estimators=200`, `max_depth=8`, `min_samples_leaf=1`, with best three-fold CV macro-F1 of 0.9994.

The reported held-out classification metrics were:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `EC_High` | 1.00 | 1.00 | 1.00 | 9 |
| `Humidity_Low` | 1.00 | 1.00 | 1.00 | 16 |
| `Multiple_Stress` | 1.00 | 0.96 | 0.98 | 23 |
| `Normal` | 1.00 | 1.00 | 1.00 | 87 |
| `WaterTemp_High` | 1.00 | 1.00 | 1.00 | 19 |
| `pH_High` | 0.86 | 1.00 | 0.92 | 6 |
| `pH_Low` | 1.00 | 1.00 | 1.00 | 2 |

Reported overall accuracy was 0.994, macro precision 0.98, macro recall 0.99, and macro-F1 0.986. The report describes one error in the 162-row test set.

### 11.7 Interpretation and further evaluation

The very high tree-model results are expected because the target is created by threshold rules and decision trees can reproduce those thresholds. The report notes that a single train/test split is not sufficient because observations from the same tank/day can be correlated. Grouped five-fold cross-validation by calendar date is the recommended stronger check if not already completed.

Never describe these results as proof that the system detects real biological stress with 99.4% field accuracy. They describe agreement with the project-generated threshold labels on the documented split.

## 12. Spinach image dataset

**Sources:**

- https://data.mendeley.com/datasets/ymtzd8cf6x/1
- https://data.mendeley.com/datasets/skf6w2s2h2/3

### 12.1 Documented classes

- `Spinach_Healthy`
- `Spinach_anthracnose`
- `Spinach_bacterial spot`
- `Spinach_downy mildew`
- `Spinach_Non fresh`
- `Spinach_pest damage`

Display mapping: `Spinach_Healthy` → `healthy`; all other classes → `unhealthy/<original class>`.

### 12.2 Input and preprocessing

Whole-plant spinach uploads should be segmented/cropped before a leaf-oriented OpenCV feature extractor is used. Deep-learning models must use the image input format established in their training pipeline. Do not assume a feature extractor or CNN accepts the same crop type without validation.

### 12.3 Verification required

Record actual image counts per class, duplicate/near-duplicate findings, preprocessing configuration, split counts, and the exact dataset versions used. Do not invent these counts from source descriptions.

## 13. Spinach sensor dataset

### 13.1 Source and preparation context

The documented source is attributed to Meng et al. (2025), *Journal of Agriculture and Food Research*, volume 24, article 102530, DOI `10.1016/j.jafr.2025.102530`. The named file is `Meng_Biradar_Park_Kelley_2025_greenhouse_temperature_humidity.csv` in USDA Ag Data Commons; the exact canonical URL must be recorded from the actual source manifest if available.

The original measurements were greenhouse air temperature and humidity recorded approximately every ten minutes for about fifteen days across benches.

### 13.2 Prepared dataset

The prepared dataset is documented as 1,400 rows with seven classes, 200 rows per class.

Input features, in the documented order:

1. `air_temperature_C`
2. `relative_humidity_percent`
3. `water_temperature_C`
4. `ph`
5. `ec_mS_cm`

Target: `label`

### 13.3 Excluded fields

Do not use these fields as model inputs:

- `tds_ppm`: derived from EC × 700 with approximately 2% added noise; redundant/derived.
- `label_id`: direct target leakage.
- `humidity_source`: may reveal whether the high-humidity class was artificially created.
- `sample_id`: identifier.
- `source_timestamp_EDT`: timestamp.
- `bench`: location/group identifier.

The full documented schema also includes `sample_id`, `air_temperature_C`, `relative_humidity_percent`, `water_temperature_C`, `ph`, `ec_mS_cm`, `tds_ppm`, `label`, `label_id`, `humidity_source`, `source_timestamp_EDT`, and `bench`.

### 13.4 Real versus synthetic data

- Air temperature: real source data across all 1,400 rows.
- Relative humidity: real for 1,200 rows; 200 rows were adjusted to create the high-humidity class.
- Water temperature, pH, EC, and TDS: synthetic or derived.
- Water temperature was documented as approximately `0.55 × air temperature + 8.5 + noise`; the exact code and noise definition must be checked before claiming the formula is exact.
- TDS is derived from EC and must not be treated as an independent sensor input.

This is a mixed real/synthetic dataset, not a dataset of five independently measured sensor channels for every row.

### 13.5 Classes

- `Normal`
- `Water_Temp_High`
- `EC_High_Toxicity`
- `EC_Low_Deficiency`
- `pH_Imbalance`
- `Humidity_High_Mildew_Risk`
- `Humidity_Low_Stress`

### 13.6 Documented rule bands

- `Normal`: water temperature ≤21 °C, EC 1.8–2.3, pH 6–7, humidity 45–75%.
- `Water_Temp_High`: water temperature >21 °C.
- `EC_High_Toxicity`: EC >2.3.
- `EC_Low_Deficiency`: EC <1.8.
- `pH_Imbalance`: pH <6 or >7.
- `Humidity_High_Mildew_Risk`: humidity >75%.
- `Humidity_Low_Stress`: humidity <45%.

When multiple conditions apply, preserve the exact priority order implemented in the preprocessing script. The list above documents thresholds, not a guaranteed precedence order.

### 13.7 Split protocol

The prepared split is documented as 160 training and 40 test rows per class, separated by six-hour blocks with no shared readings across the split. Preserve the split and its group separation. Do not replace it with a random row-level split.

The report says there were zero label mismatches after applying the rules. Treat this as a check of rule consistency, not independent biological validation.

### 13.8 Known limitations

- The balanced 200-per-class distribution is artificial.
- Some features are generated or transformed.
- `humidity_source` can leak the class and is excluded.
- `label_id` leaks the target and is excluded.
- Timestamps and bench identifiers can create leakage or unrealistic shortcuts.
- The source primarily measures air temperature and humidity; other channels are not all independent measured observations.

## 14. Tomato image dataset

**Sources:**

- https://data.mendeley.com/datasets/zfv4jj7855/1
- https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources

### 14.1 Source status

The Kaggle tomato disease dataset was downloaded, inspected, categorized, and used for project work. The dataset itself should therefore not be described as “unverified” in the sense of not having been accessed or used. However, that does not automatically verify all original provenance claims, units, license terms, or redistribution rights.

### 14.2 Classes

- `healthy`
- `Tomato_Bacterial_spot`
- `Tomato_Early_blight`
- `Tomato_Late_blight`
- `Tomato_Leaf_Mold`
- `Tomato_Septoria_leaf_spot`
- `Tomato_Spider_mites_Two-spotted_spider_mite`
- `Tomato_Target_Spot`
- `Tomato_Tomato_mosaic_virus`
- `Tomato_Tomato_Yellow_Leaf_Curl_Virus`

Display mapping: `healthy` → `healthy`; disease labels → `unhealthy/<original class>`.

### 14.3 Input policy

Whole-plant tomato images require segmentation/cropping before a leaf-oriented OpenCV extractor is applied. The CNN/EfficientNet path must follow its actual training image format. Keep the input contract explicit for each model.

## 15. Tomato sensor dataset

### 15.1 Source

Kaggle: “Autonomous Tomato Cultivation Dataset in Kolar”  
URL: https://www.kaggle.com/datasets/eshwaranece91/autonomous-tomato-cultivation-dataset-in-kolar

Original file name: `kolar_tomato_precision_agri_dataset.csv`.

The source file was downloaded and verified for project use. The dataset is documented as 200,000 rows and 29 columns, with timestamps from 2015-01-01 to 2025-12-31. Timestamp gaps are irregular, with fractional seconds. The median interval was reported as approximately 29 minutes.

### 15.2 Original columns

Documented columns:

- `Ambient_Temp`
- `Humidity`
- `Solar_Radiation`
- `Soil_Moisture_VWC`
- `Soil_pH`
- `EC_mS_cm`
- `Nitrogen_Level`
- `Phosphorus_Level`
- `Potassium_Level`
- `Pore_Water_Velocity_v`
- `Dispersion_Coeff_D`
- `Nutrient_Lockout_Prob`
- `Solute_Flux_J`
- `Sensor_Node_1_Signal_RSSI` through `Sensor_Node_15_Signal_RSSI`

### 15.3 Observed value ranges and cautious interpretation

The following min/median/max values were observed in the source inspection. Units are not all independently verified.

| Column | Minimum | Median | Maximum | Qualification |
|---|---:|---:|---:|---|
| `Ambient_Temp` | 18 | 28 | 38 | Likely °C; unit not verified |
| `Humidity` | 30 | 52.5 | 75 | Likely percent; unit not verified |
| `Solar_Radiation` | 400 | 700 | 1000 | Likely W/m²; unit not verified |
| `Soil_Moisture_VWC` | 0.10 | 0.225 | 0.35 | VWC-like range |
| `Soil_pH` | 6.2 | 7.35 | 8.5 | pH-like range |
| `EC_mS_cm` | 0.5 | 1.75 | 3.0 | Column name suggests mS/cm |
| `Nitrogen_Level` | 10 | 30 | 50 | Original units require verification |
| `Phosphorus_Level` | 5 | 15 | 25 | Original units require verification |
| `Potassium_Level` | 15 | 37.5 | 60 | Original units require verification |
| `Pore_Water_Velocity_v` | 0.01 | 0.03 | 0.05 | Unit requires verification |
| `Dispersion_Coeff_D` | 0.005 | 0.0125 | 0.02 | Unit requires verification |
| `Nutrient_Lockout_Prob` | 0.1 | 0.23 | 0.7 | Interpretation requires source documentation |
| `Solute_Flux_J` | 0.11 | 0.91 | 2.5 | Unit requires verification |
| RSSI columns | about -110 | about -75 | about -40 | Signal-strength-like values |

The columns appear to occupy fixed ranges with limited temporal structure, which suggests the dataset may be simulated. This is an observation, not proof of the data-generation process. Do not state definitively that the source is synthetic unless the source documentation establishes that.

### 15.4 Prepared data and transformations

The project preparation script is named `build_tomato_dataset.py`. The documented prepared dataset contains 44,141 rows, eight input features, and target `label`. The original CSV was not pre-split; the prepared split is by calendar month.

Documented transformations:

- ambient/air temperature retained;
- humidity increased by 15;
- pH reduced by 1;
- EC multiplied by 1.4;
- nitrogen multiplied by 4;
- phosphorus multiplied by 3;
- potassium multiplied by 6;
- water temperature derived synthetically from air temperature and solar radiation.

Documented dropped groups:

- soil moisture;
- transport-related fields;
- solute-flux and nutrient-lockout fields;
- all 15 RSSI columns.

These transformations are project-defined data preparation, not validated physical unit conversions. The exact input columns and feature order must be read from the script or generated manifest.

### 15.5 Documented sensor labels and thresholds

- `Normal`
- `Water_Temp_High`
- `EC_High_Toxicity`: EC >4.0
- `EC_Low_Deficiency`: EC <2.0
- `pH_Imbalance`: pH <6.0 or pH >6.5
- `Humidity_High_Mildew_Risk`: humidity >85%
- `Humidity_Low_Stress`: humidity <50%
- `N_Deficiency`: N <90 mg/L
- `P_Deficiency`: P <35 mg/L
- `K_Deficiency`: K <144 mg/L

The project notes say the most severe fault wins when multiple rules match. The exact priority order must be taken from `build_tomato_dataset.py`; do not infer it from the order in this list.

### 15.6 Mandatory implementation inspection

Before calling the tomato sensor input contract final, inspect the actual script and prepared data to verify:

1. the exact eight feature names;
2. the exact order of the eight features;
3. the exact formula used to derive water temperature;
4. any random seed/noise or clipping used by that formula;
5. the exact high-water-temperature threshold;
6. the precise label priority order;
7. the exact list of dropped columns;
8. the exact calendar-month train/validation/test assignments and row counts;
9. whether preprocessing introduces any clipping, rounding, or missing-value handling not described here;
10. whether all transformed features have clearly documented units.

Do not fabricate the missing values. Record the results in a dataset manifest and update this document.

### 15.7 Suitability for live sensor inference

Because humidity, pH, EC, N, P, K, and water temperature are transformed or synthetically derived in the preparation process, the model must not automatically be treated as compatible with raw campus sensor readings. The live input mapping must be documented and validated against the training feature definitions. If the mapping cannot be demonstrated, keep tomato sensor inference in an experimental/manual-assessment mode and clearly disclose the limitation.

### 15.8 Label limitations

The labels are rule-generated, and class balance is artificial. A model can learn the project rules without learning independent biological outcomes. Report performance as agreement with those rules under the documented split, not as verified tomato disease/nutrient diagnosis accuracy.

## 16. OpenCV feature engineering contracts

The ordered feature list is part of the trained model's input schema. Save it in the model metadata and validate it at inference. Correctly handle OpenCV's BGR channel ordering when extracting RGB-based features.

If a feature definition, threshold, crop policy, image colour convention, or feature order changes, regenerate the affected feature dataset and retrain every model that depends on it. Do not load a model trained with one schema and pass it features in another order.

### 16.1 Lettuce features — seven fields

1. `exg_mean`
2. `exr_mean`
3. `yellow_pixel_ratio`
4. `brown_pixel_ratio`
5. `mean_hue`
6. `mean_saturation`
7. `contour_compactness`

Documented base colour formulas:

- Excess Green: `ExG = 2G - R - B`
- Excess Red: `ExR = 1.4R - G`

`yellow_pixel_ratio` and `brown_pixel_ratio` are crop-specific configurable HSV-threshold ratios computed over the leaf area. Save the exact HSV thresholds and mask policy with the feature dataset/model metadata.

`contour_compactness` is defined as:

\[
\text{contour\_compactness} = \frac{4\pi A}{P^2}
\]

where `A` is contour area and `P` is contour perimeter. Handle a zero/invalid perimeter safely; do not allow division-by-zero to silently produce invalid feature values.

### 16.2 Spinach features — six fields

1. `exg_mean`
2. `exr_mean`
3. `yellow_pixel_ratio`
4. `brown_pixel_ratio`
5. `relative_leaf_area`
6. `glcm_contrast`

`relative_leaf_area` must use a defined denominator (for example, segmented foreground relative to image area) in the implementation. The exact definition must be inspected and recorded; do not assume two implementations use the same denominator.

### 16.3 Tomato candidate features — five fields, conditional on droop angle

Candidate feature list:

1. `droop_angle`
2. `glcm_contrast`
3. `glcm_entropy`
4. `bounding_aspect_ratio`
5. `solidity`

**Safe default for `droop_angle`:**

- Retain it only if the current implementation contains a reproducible, defensible calculation that is meaningful for the actual input image and uses a clearly defined reference axis/geometry.
- Inspect the implementation, its tests, and example outputs before deciding.
- An isolated leaf on a plain background does not inherently provide a reliable gravity direction or stem reference. A guessed angle can be misleading.
- If no defensible calculation exists, remove or replace `droop_angle`.
- If removed/replaced, regenerate the entire tomato feature dataset, update feature schema/version metadata, retrain all dependent models, and rerun validation and final test evaluation using the preserved split policy.
- Do not retain a questionable feature merely to keep a five-column schema.
- Do not claim the feature was retained or removed until the current extractor has actually been inspected and the decision applied.

`bounding_aspect_ratio` and `solidity` also need exact definitions recorded in the implementation. Typical definitions are not a substitute for inspecting the actual code.

### 16.4 GLCM defaults

Unless the existing, tested implementation explicitly establishes a different contract:

- convert the isolated leaf crop to grayscale;
- quantize to 32 gray levels;
- use pixel distance 1;
- use directions 0°, 45°, 90°, and 135°;
- average the relevant GLCM properties across directions.

The current code must be inspected to confirm whether the library normalizes matrices, symmetrizes them, handles borders in a particular way, and computes entropy with a particular logarithm convention. These details can change feature values. Do not silently change the feature calculation after training.

### 16.5 Feature extraction failure behaviour

- Reject or flag empty crops and invalid masks.
- Check for NaN/Inf features before model inference.
- Check the number, names, and order of features against the artifact metadata.
- Do not silently substitute zero for a failed feature.
- If the feature schema changes, make the model incompatible until retrained or explicitly migrated.

## 17. Sensor model strategy and validation

### 17.1 Candidate models

Evaluate:

- Decision Tree;
- Random Forest;
- SVM;
- other suitable classifiers where they add value.

Use the exact crop-specific feature schema. Scale features where the selected algorithm requires it; fit the scaler on training data only and package it with the model. Ensure training and inference use the same preprocessing.

### 17.2 Selection criteria

- Primary selection metric: validation macro-F1.
- Also report accuracy, macro precision, macro recall, class-wise precision/recall/F1, confusion matrix, and class support.
- Consider per-class failure impact and whether the model's errors could lead to unsafe or costly recommendations.
- Do not choose a model using final test results.
- Report class distribution and split design alongside all metrics.

### 17.3 Missing and invalid values

- Never replace missing values with arbitrary zero.
- Define whether an input can be accepted with some fields missing. The rule and model must agree.
- If a required feature is missing, reject prediction or use an explicitly trained/validated missing-data path.
- Do not impute using statistics calculated on validation/test data.
- Reject values with invalid units or impossible ranges unless a documented validation policy explicitly handles them.
- Distinguish “sensor missing,” “sensor disconnected,” “invalid reading,” and “stale reading” in the UI/API where possible.

### 17.4 Synthetic and rule-derived data

A model trained on rule-generated labels can reproduce the rules; that is useful for software consistency but is not independent evidence that the rules diagnose plant health. For mixed/synthetic data, clearly disclose which features are measured, adjusted, derived, or synthetic. Never market synthetic data as fully measured sensor ground truth.

## 18. Model selection, splits, and evaluation policy

### 18.1 Primary metric

Use **validation macro-F1** as the primary model-selection metric. Macro-F1 gives each class equal weight and helps avoid selecting a model solely because it performs well on a majority class.

### 18.2 Additional required reporting

For each evaluated model, record:

- accuracy;
- macro precision;
- macro recall;
- macro-F1;
- per-class precision, recall, F1, and support;
- confusion matrix;
- train/validation/test counts;
- class counts per split;
- split/grouping protocol;
- source dataset/version and preprocessing version;
- label-generation method;
- model family, hyperparameters, and artifact version;
- evaluation date and software/runtime versions where relevant.

### 18.3 Split rules

- Image target split: 70/15/15 train/validation/test, stratified where appropriate, after duplicate checks.
- Lettuce sensor: preserve the prepared day/experiment-based split; do not replace with a random split.
- Spinach sensor: preserve six-hour time-block separation and the 160-train/40-test-per-class prepared split.
- Tomato sensor: preserve month-based split and inspect the actual month assignments.
- Do not overwrite existing prepared split files or manifests.
- Test data remains untouched until final evaluation.

### 18.4 Cross-validation

For correlated sensor time series, use group-aware or time-aware cross-validation rather than random row-wise folds. For lettuce, the available report recommends grouped five-fold cross-validation by calendar date because same-day/tank readings may be correlated. Record whether this has actually been run; do not state it is complete unless an output report exists.

### 18.5 Metric interpretation

- A high score on synthetic/rule-labelled data can reflect reproduction of the label-generation rules.
- A high score on one split does not guarantee performance on another greenhouse, cultivar, season, camera, lighting setup, or sensor.
- Report the exact evaluation context.
- Do not describe validation results as test results.
- Do not present an internal dataset score as a guarantee of real-world crop-health accuracy.

## 19. Image/sensor evidence and fusion

The image and sensor datasets are not generally paired observations. They have different classes, different data-generation processes, and different semantic meanings.

Required policy:

1. Preserve the image prediction and sensor prediction separately.
2. Store each prediction's modality, timestamp, input provenance, confidence, and model version.
3. Do not train a fusion model as if unrelated image and sensor rows were paired.
4. Only compare outputs using a documented and reviewed mapping that is semantically justified and validated.
5. If there is no valid mapping, show both results separately and explain that they are not directly comparable.
6. Flag a mismatch only when the mapping makes the comparison meaningful.
7. If one modality is unavailable, continue with the available modality and disclose what was not assessed.
8. Never turn an arbitrary image/sensor class difference into a “contradiction” alert.
9. If a future paired dataset is collected, document plant identity, timestamp alignment, crop/cultivar, growing conditions, and pairing rules before training a fusion model.

## 20. Recommendation system

### 20.1 Responsibilities

The recommendation layer should convert supported evidence into understandable, conservative suggestions. It should:

- explain which model output or sensor rule led to the suggestion;
- state uncertainty and missing evidence;
- suggest practical checks or next steps;
- distinguish a predicted condition from a confirmed diagnosis;
- distinguish advice from actions actually performed by a device;
- avoid presenting unsupported nutrient dosage or concentration instructions.

### 20.2 Gemini integration

Gemini is the planned language-model API for recommendation generation. It must not be the only path:

- use a deterministic, rule-based fallback if Gemini is unavailable, times out, or returns unusable content;
- validate the returned structure/content before displaying it;
- treat model-generated text as untrusted;
- do not allow recommendation text to execute shell commands, change code, modify datasets, overwrite model files, or access secrets;
- do not send API keys or unnecessary personal data to the model;
- never claim Gemini output is agronomically validated merely because it is fluent.

### 20.3 Rule-based fallback

The fallback should use documented crop-specific rules and available evidence. It should return a cautious “insufficient information” result when the input does not support a recommendation. Rules must be versioned and tested. Do not invent a crop-specific dosage from a class label alone.

### 20.4 Recommendations and confidence

- Derive any confidence threshold from validation data.
- If confidence is low, say the result is uncertain and recommend further measurement/inspection rather than a strong corrective action.
- If image and sensor evidence cannot be compared, do not combine them into a single severity score.
- Recommendations are advisory and do not replace grower judgment, laboratory testing, or qualified agronomic advice.

## 21. Historical records and campus/device integration

### 21.1 Historical records

Each stored record should identify, where applicable:

- crop;
- input modality;
- observed timestamp and timezone;
- whether the reading was live, manually entered, or imported;
- input values or a safe reference to them;
- predicted class;
- confidence/uncertainty;
- model version;
- preprocessing/rule version;
- recommendation text and rule/model evidence.

Historical readings must retain their true timestamp. Never relabel an old reading as current simply because it is the most recent record.

### 21.2 Campus integration

- Keep campus integration modular so the application works without a device.
- Provide manual input when hardware/device access is unavailable.
- Do not fabricate live sensor values.
- Validate the device payload's schema, units, timestamps, and freshness.
- Treat disconnected, stale, malformed, or unsupported device data as unavailable.
- The adapter must map physical sensors to the exact model feature schema.
- Tomato sensor inference requires extra compatibility checks because of synthetic/adjusted training features.
- Do not assume a hardware action was executed unless the device confirms it. The initial project is a decision-support application, not an automatic nutrient controller.

## 22. Backend and API contracts

The backend default is FastAPI. Exact routes must be derived from the existing code before being treated as final. The following are functional contract requirements, not fixed route names.

### 22.1 Input validation

- Validate crop identifier and analysis mode.
- Validate image MIME/type, file size, and decodability.
- Do not trust client filenames as filesystem paths.
- Validate sensor feature names, order, units, missing values, and numeric ranges.
- Reject unexpected fields unless the API contract intentionally supports versioned extensions.
- Return clear, structured errors for invalid inputs, missing models, failed segmentation, unsupported schema, and inference failures.

### 22.2 Prediction response

A prediction response should, where supported, contain:

- crop and modality;
- predicted class and human-readable display label;
- confidence/uncertainty information with defined semantics;
- model identifier/version;
- timestamp;
- input/preprocessing provenance;
- warnings and limitations;
- whether the result is experimental, uncertain, or suitable for the requested inference mode.

Do not claim exact response JSON field names until the actual Pydantic schemas and routes are inspected.

### 22.3 Failure handling

- Missing/corrupt model artifact → clear service error; do not silently load a different model.
- Feature mismatch → reject prediction and identify schema incompatibility.
- Segmentation failure → warn and offer retry.
- Invalid sensor unit/value → reject or explicitly mark invalid.
- Gemini failure → use rule-based fallback.
- GPU out-of-memory → use a documented smaller batch/CPU fallback where feasible; otherwise return a clear error.
- Database failure → do not claim that history was saved.
- Campus device unavailable → allow manual entry and state that live data is unavailable.

## 23. Frontend/backend security and privacy

- Store secrets in environment variables; never commit real API keys or passwords.
- Maintain `.env.example` with placeholder values only.
- Validate upload type and size; do not trust filenames or paths.
- Avoid logging raw secrets, API tokens, or unnecessary personal data.
- Treat Gemini-generated content as untrusted.
- Restrict model artifact loading to trusted configured paths.
- Avoid executing user-supplied or model-generated code.
- Ensure errors do not expose secret values or internal credentials.
- Keep access controls appropriate to the initial no-login scope; if authentication is later added, document the new threat model and authorization rules.
- Follow the license/terms of each dataset and model weight before redistribution.

## 24. Model registry and artifact management

Use a structured local registry initially, with explicit promotion, versioned artifacts, metadata, and rollback.

### 24.1 Metadata to save with every model

- crop and task;
- modality (image, sensor, or feature-based image);
- model family/architecture;
- training code commit/version;
- dataset manifest/version/hash where feasible;
- class names and display mapping;
- split protocol and counts;
- ordered feature list and units for sensor/feature models;
- image size/normalization/channel convention for image models;
- segmentation and feature-extraction configuration;
- label-generation/rule version;
- preprocessing version;
- hyperparameters;
- validation metrics;
- final test metrics, if final evaluation has been completed;
- confidence calibration/threshold settings;
- dependency/runtime versions;
- expected input schema/shape;
- training date and artifact version.

### 24.2 Promotion and rollback

- An experiment does not become the active model merely because its filename is newest.
- Promote a model explicitly after reviewing validation performance, limitations, and compatibility.
- Verify schema compatibility at startup and inference.
- Keep the previous known-good artifact available for rollback.
- Do not overwrite a deployed artifact without preserving version and metadata.
- If a feature schema changes, mark old artifacts incompatible until they are retrained or explicitly migrated.

## 25. GPU, CPU, and resource policy

The user's personal laptop is an HP Victus 16 with an NVIDIA RTX 3050, running Windows 11 and Ubuntu. This is the intended GPU target **when the active environment has compatible hardware drivers, CUDA/PyTorch builds, and available memory**.

- Check GPU availability at runtime rather than assuming it.
- Keep a CPU fallback.
- Use adaptive batch sizes and memory-aware inference/training.
- Handle CUDA out-of-memory gracefully.
- Avoid unnecessarily large model downloads when a smaller model meets the task.
- Record device/runtime details in training reports.
- Do not claim GPU acceleration if the process actually ran on CPU.
- The campus/lab environment may not have a GPU; code must not depend on GPU-only execution.

## 26. Logging, observability, and reproducibility

- Use structured logs for major stages: validation, segmentation, feature extraction, inference, recommendation fallback, and persistence.
- Include model/version identifiers in logs where helpful.
- Do not log secrets.
- Avoid logging unnecessary personal data or full uploaded content.
- Log failed schema checks and segmentation failures clearly.
- Record dataset counts, feature schema, split manifest, random seeds, configuration, and dependency versions for training runs.
- Keep preprocessing deterministic where possible.
- If stochastic operations are used, record random seeds and relevant configuration.
- A run is reproducible only when the code, data version, configuration, and split assignments are recoverable.

## 27. Testing strategy

### 27.1 Unit tests

Test at least:

- class-name normalization and display mapping;
- sensor feature schema/order validation;
- unit/range validation;
- missing-value handling;
- label-rule boundary conditions;
- multi-rule priority order;
- each OpenCV feature formula and edge case;
- zero/invalid contour perimeter handling;
- GLCM configuration;
- `droop_angle` calculation if retained;
- segmentation mask validation;
- confidence/uncertainty threshold logic;
- rule-based recommendation fallback;
- model metadata and compatibility validation.

### 27.2 Integration tests

Test at least:

- image upload → segmentation/preprocessing → model prediction;
- individual-leaf feature extraction → expected feature vector → model prediction;
- sensor payload → schema validation → prediction;
- unavailable Gemini → fallback recommendation;
- unavailable campus device → manual-input workflow;
- save and retrieve timestamped history;
- missing/corrupt model artifact behaviour;
- GPU unavailable or out-of-memory fallback behaviour where testable.

### 27.3 Data and leakage tests

Test at least:

- class counts before/after preprocessing;
- no overlap of source image identities across image splits;
- all crops from the same original lettuce image remain in one split;
- sensor split/group/time separation is preserved;
- prohibited leakage fields are absent from model inputs;
- target column is not present among features;
- train/validation/test files are not silently regenerated;
- no NaN/Inf values reach models without a documented policy;
- generated feature column order exactly matches artifact metadata.

### 27.4 Acceptance conditions

Do not mark a feature complete until the implementation, tests, and/or artifacts demonstrate the behaviour. A UI mockup or written plan is not proof that the backend integration, model, or hardware path works.

## 28. Known limitations and required disclosures

1. Some sensor labels are generated from thresholds rather than observed plant-health outcomes.
2. Lettuce sensor labels reflect project-defined reference bands; the available report describes 726 real chemistry readings, not thousands of independent chemistry observations.
3. The lettuce training set was oversampled; the held-out test set retained real rows and original held-out distribution.
4. Spinach sensor data mixes real air readings with adjusted humidity and synthetic/derived water chemistry features.
5. Tomato sensor data uses transformed features and synthetically derived water temperature.
6. Tomato source units/provenance are not all independently verified.
7. Tomato source values appear to have fixed-range structure, which suggests simulation but does not prove it.
8. Class balance in prepared sensor datasets may be artificial and not representative of field prevalence.
9. Image and sensor records are not generally paired.
10. Different image and sensor taxonomies prevent automatic direct comparison without a validated mapping.
11. Whole-plant and individual-leaf image inputs are not interchangeable.
12. A model trained on rule-derived labels may reproduce the rules without learning biological causation.
13. Performance depends on the documented dataset, split, preprocessing, and label generation; it is not a guarantee of field accuracy.
14. Dataset and model licenses/redistribution terms must be checked.
15. `droop_angle` is conditional on a defensible reproducible implementation; otherwise it must be removed/replaced and all dependent features/models rebuilt.
16. Campus/live tomato sensor inference is not validated until transformed training features are mapped to actual device measurements.
17. A model confidence score is not automatically a calibrated probability.
18. Recommendations are advisory and must remain conservative.

## 29. Required repository verification checklist

These are explicit verification tasks, not permission to guess.

### 29.1 Tomato OpenCV extractor

- [ ] Locate the actual tomato OpenCV feature-extraction source.
- [ ] Inspect the `droop_angle` formula, reference axis, mask/contour assumptions, and tests.
- [ ] Decide whether the feature is reproducible and defensible.
- [ ] If not, remove or replace it.
- [ ] Regenerate the tomato feature dataset if the schema changes.
- [ ] Retrain all dependent models.
- [ ] Re-evaluate using the preserved split protocol.
- [ ] Update feature schema, model metadata, manifests, tests, and this document.

### 29.2 Tomato sensor preparation script

Inspect `build_tomato_dataset.py` and the generated manifest. Record:

- [ ] exact eight feature names and order;
- [ ] exact water-temperature formula and any noise/seed/clipping;
- [ ] high-water-temperature threshold;
- [ ] precise label priority when several faults match;
- [ ] exact dropped columns;
- [ ] missing-value handling;
- [ ] exact calendar-month split assignments and row counts;
- [ ] final class counts;
- [ ] output paths and reproducibility settings;
- [ ] units and whether each value is measured, transformed, or synthetic.

### 29.3 Other dataset checks

- [ ] Confirm current lettuce sensor files and compare their counts with the training report.
- [ ] Run or verify grouped date-based cross-validation for lettuce if not already complete.
- [ ] Verify spinach preprocessing rule priority.
- [ ] Verify spinach time-block split assignments.
- [ ] Verify image dataset counts, duplicates, near-duplicates, and split manifests.
- [ ] Confirm source versions and license/terms for all datasets.
- [ ] Confirm the exact canonical source URL for the spinach source file.
- [ ] Confirm all inference preprocessing matches training preprocessing.

### 29.4 Application/runtime checks

- [ ] Verify the actual frontend/backend folder structure and API contracts.
- [ ] Verify PostgreSQL persistence and timestamp semantics.
- [ ] Verify Gemini fallback behaviour.
- [ ] Verify campus adapter behaviour with no device connected.
- [ ] Verify model registry metadata and rollback.
- [ ] Verify runtime GPU detection and CPU fallback.
- [ ] Verify `.env.example` contains placeholders only and no real secrets.
- [ ] Verify layered tests pass.
- [ ] Update this document from actual evidence rather than assumptions.

## 30. Instructions for AI coding assistants and contributors

When working on this repository, follow these rules:

1. Read `PROJECT_OVERVIEW.md` before changing architecture, dataset preparation, feature extraction, training, evaluation, or application behaviour.
2. Treat documented decisions as defaults, but inspect actual code before claiming an implementation detail is confirmed.
3. Do not ask the user to repeat decisions already documented here.
4. If a required fact is absent, inspect the repository, scripts, manifests, reports, and artifacts first.
5. Ask a user question only when the answer cannot reasonably be determined from the available project files and materially affects a decision.
6. Prefer safe, reversible defaults and explain consequential changes.
7. Never invent a formula, threshold, class priority, feature order, metric, split count, endpoint, or successful test result.
8. Do not overwrite raw datasets or prepared split files.
9. Preserve time/group separation and avoid leakage.
10. When changing a feature definition, regenerate affected features and retrain/evaluate dependent models.
11. Do not use test data to tune models or thresholds.
12. Keep measured, adjusted, derived, synthetic, and rule-generated data clearly distinguished.
13. Keep image and sensor outputs separate unless a validated mapping supports comparison.
14. Do not silently fill missing sensor values with zero.
15. Keep recommendations advisory, explainable, and backed by evidence.
16. Keep secrets out of source control and logs.
17. Do not claim GPU use without runtime evidence.
18. Add or update tests for changed behaviour.
19. Update this document whenever a verified project contract changes.
20. When implementation cannot be verified, state the uncertainty and add a specific TODO rather than fabricating certainty.
21. Do not mark a task complete merely because code was generated; run relevant tests or report clearly what could not be run.
22. When reporting results, state the data source, split, metric, and limitations.
23. Preserve previous working artifacts when promoting a new model.
24. Keep the project understandable for future teammates: use clear names, documented schemas, reproducible scripts, and explicit error messages.

## 31. Decision log — defaults already established

| Topic | Canonical decision |
|---|---|
| Crops | Lettuce, spinach, tomato |
| Frontend | Next.js + React |
| Backend | FastAPI |
| Database | PostgreSQL |
| Image deep learning | Custom CNN and EfficientNetB0 comparison using PyTorch |
| Traditional ML | scikit-learn |
| Segmentation | SAM 2.1 initial approach; validate masks and allow retry |
| Sensor models | Decision Tree, Random Forest, SVM, and suitable alternatives |
| Model-selection metric | Validation macro-F1 |
| Other evaluation | Accuracy, macro precision/recall, per-class metrics, confusion matrix, support |
| Test set | Untouched until final evaluation |
| Image split target | 70/15/15 stratified where appropriate |
| Existing sensor splits | Preserve; do not recreate or overwrite |
| Lettuce crop leakage | Keep all crops from one original image in one split |
| Sensor missing values | Never arbitrary zero fill |
| Fusion | Separate outputs unless validated mapping exists |
| Recommendations | Gemini with deterministic rule fallback |
| Authentication | Not required initially |
| Campus integration | Modular, manual fallback; never fabricate live values |
| History | Timestamped; old readings are not live |
| Storage | Local files/artifacts initially; do not assume cloud storage |
| Model registry | Structured, versioned, explicit promotion and rollback |
| Hardware | RTX 3050 when compatible, CPU fallback, adaptive resources |
| Tomato source status | Downloaded/inspected/used; provenance/units/license remain separate questions |
| Tomato sensor live use | Experimental until feature compatibility is validated |
| Tomato `droop_angle` | Retain only if current calculation is reproducible and defensible; otherwise remove/replace, regenerate features, retrain |
| Dataset labels | Disclose rule-derived labels and synthetic/derived inputs |
| Unknown implementation facts | Inspect repository; never guess |

## 32. Change-management rule

This document should evolve with the implementation.

When a decision or contract changes:

1. identify the affected source code, data, model artifacts, UI, and tests;
2. verify the current behaviour;
3. document the new contract and reason;
4. update schemas/manifests and tests;
5. regenerate features/retrain models if required;
6. evaluate using the approved split protocol;
7. preserve prior artifacts for rollback;
8. update this overview and any downstream documentation;
9. report any remaining uncertainty explicitly.

**Final rule:** the project should be reproducible, transparent about limitations, and safe against unsupported claims. When evidence is missing, document the gap; never fill it with an invented fact.
