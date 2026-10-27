"""Class-label registry, taxonomy definitions, and deterministic display mapping (T2-016).

Adheres to TECH_RULES and PROJECT_OVERVIEW Section 6:
- Label lists strictly equal the folder names on disk under image_data/ (21 image classes total).
- Exact source spelling preserved (including spaces in 'Tomato_Spider_mites Two-spotted_spider_mite').
- Deterministic display mapping: 'healthy' or 'unhealthy/<original class>'.
- Unknown labels raise a clear error; never infer a class from a filename.
"""

from __future__ import annotations

from typing import Final, Literal

CropType = Literal["lettuce", "spinach", "tomato"]
ModalityType = Literal["image", "sensor"]

# ==============================================================================
# Exact Folder Names on Disk under image_data/ (21 Image Classes Total)
# ==============================================================================

LETTUCE_IMAGE_CLASSES: Final[tuple[str, ...]] = (
    "Healthier",
    "NitrogenDeficiency",
    "PhosphorusDeficiency",
    "PotassiumDeficiency",
)

SPINACH_IMAGE_CLASSES: Final[tuple[str, ...]] = (
    "Spinach_Healthy",
    "Spinach_Non fresh",
    "Spinach_anthracnose",
    "Spinach_bacterial spot",
    "Spinach_downy mildew",
    "Spinach_pest damage",
)

TOMATO_IMAGE_CLASSES: Final[tuple[str, ...]] = (
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Powdery_Mildew",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites Two-spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
)

ALL_IMAGE_CLASSES: Final[tuple[str, ...]] = (
    LETTUCE_IMAGE_CLASSES + SPINACH_IMAGE_CLASSES + TOMATO_IMAGE_CLASSES
)

# ==============================================================================
# Sensor Class Lists from Prepared Datasets
# ==============================================================================

LETTUCE_SENSOR_CLASSES: Final[tuple[str, ...]] = (
    "EC_High",
    "EC_Low",
    "Humidity_Low",
    "Multiple_Stress",
    "Normal",
    "WaterTemp_High",
    "WaterTemp_Low",
    "pH_High",
    "pH_Low",
)

SPINACH_SENSOR_CLASSES: Final[tuple[str, ...]] = (
    "EC_High_Toxicity",
    "EC_Low_Deficiency",
    "Humidity_High_Mildew_Risk",
    "Humidity_Low_Stress",
    "Normal",
    "Water_Temp_High",
    "pH_Imbalance",
)

TOMATO_SENSOR_CLASSES: Final[tuple[str, ...]] = (
    "EC_High_Toxicity",
    "EC_Low_Deficiency",
    "Humidity_High_Mildew_Risk",
    "Humidity_Low_Stress",
    "K_Deficiency",
    "N_Deficiency",
    "Normal",
    "P_Deficiency",
    "Water_Temp_High",
    "pH_Imbalance",
)

# Live-compatible tomato sensor variant (T4-020):
# Excludes N/P/K deficiencies because Modbus N/P/K hardware sensors are disabled/deleted on the campus gateway (ADR 0008).
TOMATO_LIVE_SENSOR_CLASSES: Final[tuple[str, ...]] = (
    "EC_High_Toxicity",
    "EC_Low_Deficiency",
    "Humidity_High_Mildew_Risk",
    "Humidity_Low_Stress",
    "Normal",
    "Water_Temp_High",
    "pH_Imbalance",
)

# Registry Maps
IMAGE_CLASSES: Final[dict[str, tuple[str, ...]]] = {
    "lettuce": LETTUCE_IMAGE_CLASSES,
    "spinach": SPINACH_IMAGE_CLASSES,
    "tomato": TOMATO_IMAGE_CLASSES,
}

SENSOR_CLASSES: Final[dict[str, tuple[str, ...]]] = {
    "lettuce": LETTUCE_SENSOR_CLASSES,
    "spinach": SPINACH_SENSOR_CLASSES,
    "tomato": TOMATO_SENSOR_CLASSES,
    "tomato_live": TOMATO_LIVE_SENSOR_CLASSES,
}

# Healthy label representations across modalities
HEALTHY_IMAGE_LABELS: Final[frozenset[str]] = frozenset(
    {
        "Healthier",
        "Spinach_Healthy",
        "Tomato_Healthy",
        "Healthy",
        "healthy",
    }
)

HEALTHY_SENSOR_LABELS: Final[frozenset[str]] = frozenset(
    {
        "Normal",
    }
)


# ==============================================================================
# Custom Exceptions
# ==============================================================================


class UnknownCropError(ValueError):
    """Raised when an unknown crop is specified."""


class UnknownLabelError(ValueError):
    """Raised when an unknown class label is encountered."""


class UnknownModalityError(ValueError):
    """Raised when an unknown modality is requested."""


# ==============================================================================
# Label Validation & Registry Access
# ==============================================================================


def get_classes(crop: str, modality: str = "image") -> tuple[str, ...]:
    """Return the registered class labels for a crop and modality."""
    crop_lower = crop.lower()
    mod_lower = modality.lower()

    if mod_lower == "image":
        if crop_lower not in IMAGE_CLASSES:
            raise UnknownCropError(
                f"Unknown crop '{crop}'. Registered image crops: {sorted(IMAGE_CLASSES.keys())}"
            )
        return IMAGE_CLASSES[crop_lower]

    if mod_lower == "sensor":
        if crop_lower not in SENSOR_CLASSES:
            raise UnknownCropError(
                f"Unknown crop '{crop}'. Registered sensor crops: {sorted(SENSOR_CLASSES.keys())}"
            )
        return SENSOR_CLASSES[crop_lower]

    raise UnknownModalityError(
        f"Unknown modality '{modality}'. Expected 'image' or 'sensor'."
    )


def validate_label(crop: str, label: str, modality: str = "image") -> str:
    """Validate that label belongs to the registered taxonomy for crop and modality.

    Raises UnknownLabelError if the label is unrecognised.
    Returns the exact valid label.
    """
    valid_classes = get_classes(crop, modality)
    if label not in valid_classes:
        raise UnknownLabelError(
            f"Unknown {modality} label '{label}' for crop '{crop}'. "
            f"Expected one of: {list(valid_classes)}"
        )
    return label


def get_display_label(crop: str, label: str, modality: str = "image") -> str:
    """Deterministic display mapping adhering to Overview Section 6.

    Maps:
    - Healthy class -> 'healthy'
    - Any stress / deficiency / disease class -> 'unhealthy/<original class>'

    Raises UnknownLabelError if the label is unknown.
    """
    valid_label = validate_label(crop, label, modality)

    mod_lower = modality.lower()
    if mod_lower == "image":
        if valid_label in HEALTHY_IMAGE_LABELS:
            return "healthy"
        return f"unhealthy/{valid_label}"

    if mod_lower == "sensor":
        if valid_label in HEALTHY_SENSOR_LABELS:
            return "healthy"
        return f"unhealthy/{valid_label}"

    raise UnknownModalityError(f"Unknown modality '{modality}'")


def get_class_index(crop: str, label: str, modality: str = "image") -> int:
    """Return 0-indexed integer class position for model output mapping."""
    valid_label = validate_label(crop, label, modality)
    return get_classes(crop, modality).index(valid_label)


def get_class_from_index(crop: str, index: int, modality: str = "image") -> str:
    """Return label corresponding to 0-indexed integer position."""
    classes = get_classes(crop, modality)
    if index < 0 or index >= len(classes):
        raise IndexError(
            f"Index {index} out of bounds for crop '{crop}' ({len(classes)} classes)."
        )
    return classes[index]


def num_classes(crop: str, modality: str = "image") -> int:
    """Return total number of classes for crop and modality."""
    return len(get_classes(crop, modality))
