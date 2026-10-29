"use client";

import React, { useState, useEffect, useRef } from "react";
import { Badge, Button, Card, Input } from "@/components/ui";

export type SupportedCrop = "lettuce" | "spinach" | "tomato";
export type ProvenanceSource = "manual" | "live";

export interface SensorFieldDef {
  name: string;
  label: string;
  unit: string;
  hardBounds: [number, number];
  normalBand: [number, number];
  description: string;
  step?: string;
  placeholder?: string;
}

/**
 * Single source of truth for per-crop sensor fields matching:
 * - ml/sensor_models/schemas.py
 * - backend/app/schemas/sensors.py
 */
export const CROP_SENSOR_SCHEMAS: Record<SupportedCrop, SensorFieldDef[]> = {
  lettuce: [
    {
      name: "ph",
      label: "Nutrient Solution pH",
      unit: "pH",
      hardBounds: [3.0, 10.0],
      normalBand: [5.5, 6.5],
      description: "Acidity / alkalinity of the recirculating nutrient solution",
      step: "0.1",
      placeholder: "e.g. 5.8",
    },
    {
      name: "ec",
      label: "Electrical Conductivity",
      unit: "mS/cm",
      hardBounds: [0.0, 6.0],
      normalBand: [1.2, 2.2],
      description: "Salinity concentration of dissolved fertilizer ions",
      step: "0.05",
      placeholder: "e.g. 1.6",
    },
    {
      name: "water_temp",
      label: "Water Temperature",
      unit: "degC",
      hardBounds: [0.0, 45.0],
      normalBand: [18.0, 24.0],
      description: "Root-zone water temperature in the reservoir",
      step: "0.1",
      placeholder: "e.g. 21.0",
    },
    {
      name: "humidity",
      label: "Relative Humidity",
      unit: "%",
      hardBounds: [0.0, 100.0],
      normalBand: [50.0, 70.0],
      description: "Ambient air relative humidity around the canopy",
      step: "0.5",
      placeholder: "e.g. 65.0",
    },
    {
      name: "air_temp",
      label: "Air Temperature",
      unit: "degC",
      hardBounds: [0.0, 55.0],
      normalBand: [18.0, 26.0],
      description: "Greenhouse ambient air temperature",
      step: "0.1",
      placeholder: "e.g. 22.5",
    },
  ],
  spinach: [
    {
      name: "air_temperature_C",
      label: "Air Temperature",
      unit: "degC",
      hardBounds: [0.0, 55.0],
      normalBand: [15.0, 25.0],
      description: "Greenhouse air temperature logger reading",
      step: "0.1",
      placeholder: "e.g. 19.5",
    },
    {
      name: "relative_humidity_percent",
      label: "Relative Humidity",
      unit: "%",
      hardBounds: [0.0, 100.0],
      normalBand: [45.0, 75.0],
      description: "Greenhouse relative humidity logger reading",
      step: "0.5",
      placeholder: "e.g. 60.0",
    },
    {
      name: "water_temperature_C",
      label: "Water Temperature",
      unit: "degC",
      hardBounds: [0.0, 45.0],
      normalBand: [15.0, 21.0],
      description: "Root-zone nutrient solution temperature",
      step: "0.1",
      placeholder: "e.g. 18.0",
    },
    {
      name: "ph",
      label: "Nutrient Solution pH",
      unit: "pH",
      hardBounds: [3.0, 10.0],
      normalBand: [6.0, 7.0],
      description: "Nutrient solution pH level",
      step: "0.1",
      placeholder: "e.g. 6.4",
    },
    {
      name: "ec_mS_cm",
      label: "Electrical Conductivity",
      unit: "mS/cm",
      hardBounds: [0.0, 6.0],
      normalBand: [1.8, 2.3],
      description: "Electrical conductivity of the nutrient solution",
      step: "0.05",
      placeholder: "e.g. 2.0",
    },
  ],
  tomato: [
    {
      name: "air_temperature_C",
      label: "Air Temperature",
      unit: "degC",
      hardBounds: [0.0, 55.0],
      normalBand: [18.0, 28.0],
      description: "Greenhouse ambient air temperature",
      step: "0.1",
      placeholder: "e.g. 24.0",
    },
    {
      name: "relative_humidity_percent",
      label: "Relative Humidity",
      unit: "%",
      hardBounds: [0.0, 100.0],
      normalBand: [50.0, 85.0],
      description: "Ambient relative humidity in tomato vine canopy",
      step: "0.5",
      placeholder: "e.g. 68.0",
    },
    {
      name: "water_temperature_C",
      label: "Water Temperature",
      unit: "degC",
      hardBounds: [0.0, 45.0],
      normalBand: [18.0, 25.0],
      description: "Hydroponic reservoir water temperature",
      step: "0.1",
      placeholder: "e.g. 22.0",
    },
    {
      name: "ph",
      label: "Nutrient Solution pH",
      unit: "pH",
      hardBounds: [3.0, 10.0],
      normalBand: [6.0, 6.5],
      description: "Target root-zone pH balance",
      step: "0.1",
      placeholder: "e.g. 6.2",
    },
    {
      name: "ec_mS_cm",
      label: "Electrical Conductivity",
      unit: "mS/cm",
      hardBounds: [0.0, 8.0],
      normalBand: [2.0, 4.0],
      description: "Total dissolved fertilizer salts electrical conductivity",
      step: "0.05",
      placeholder: "e.g. 2.8",
    },
    {
      name: "nitrogen_mg_L",
      label: "Dissolved Nitrogen (N)",
      unit: "mg/L",
      hardBounds: [0.0, 500.0],
      normalBand: [90.0, 250.0],
      description: "Available nitrate/ammonium concentration",
      step: "1.0",
      placeholder: "e.g. 150.0",
    },
    {
      name: "phosphorus_mg_L",
      label: "Dissolved Phosphorus (P)",
      unit: "mg/L",
      hardBounds: [0.0, 200.0],
      normalBand: [35.0, 100.0],
      description: "Available phosphate concentration",
      step: "1.0",
      placeholder: "e.g. 50.0",
    },
    {
      name: "potassium_mg_L",
      label: "Dissolved Potassium (K)",
      unit: "mg/L",
      hardBounds: [0.0, 800.0],
      normalBand: [144.0, 400.0],
      description: "Available potassium ion concentration",
      step: "1.0",
      placeholder: "e.g. 280.0",
    },
  ],
};

export interface SensorFormProps {
  crop: SupportedCrop;
  onSubmit: (readings: Record<string, number>, provenance: ProvenanceSource) => void;
  onChange?: (readings: Record<string, number>, isValid: boolean) => void;
  isLoading?: boolean;
  initialValues?: Record<string, number | null | undefined>;
  onReset?: () => void;
  className?: string;
}

export const SensorForm: React.FC<SensorFormProps> = ({
  crop,
  onSubmit,
  onChange,
  isLoading = false,
  initialValues,
  onReset,
  className = "",
}) => {
  const fields = CROP_SENSOR_SCHEMAS[crop] || CROP_SENSOR_SCHEMAS.lettuce;

  // Track raw string values: Empty string "" means missing (NEVER coerced to 0)
  const [values, setValues] = useState<Record<string, string>>(() => {
    const init: Record<string, string> = {};
    for (const f of fields) {
      const val = initialValues?.[f.name];
      init[f.name] = val !== undefined && val !== null ? String(val) : "";
    }
    return init;
  });

  const [provenance, setProvenance] = useState<ProvenanceSource>("manual");
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [warnings, setWarnings] = useState<Record<string, string>>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});

  // Reset / sync form values only when crop switches
  const lastCropRef = useRef<SupportedCrop>(crop);
  useEffect(() => {
    if (lastCropRef.current !== crop) {
      lastCropRef.current = crop;
      const newValues: Record<string, string> = {};
      for (const f of fields) {
        const val = initialValues?.[f.name];
        newValues[f.name] = val !== undefined && val !== null ? String(val) : "";
      }
      setValues(newValues);
      setErrors({});
      setWarnings({});
      setTouched({});
    }
  }, [crop, fields, initialValues]);

  const handleFillDefaults = () => {
    const defs: Record<string, string> = {};
    for (const f of fields) {
      const val =
        initialValues?.[f.name] ??
        Number(((f.normalBand[0] + f.normalBand[1]) / 2).toFixed(1));
      defs[f.name] = String(val);
    }
    setValues(defs);
    setErrors({});
    setWarnings({});
  };

  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const lastEmittedRef = useRef<string>("");

  // Synchronize live field modifications to parent container only when actual values change
  useEffect(() => {
    if (!onChangeRef.current) return;
    const numericReadings: Record<string, number> = {};
    let allValid = true;
    for (const f of fields) {
      const raw = values[f.name];
      if (raw !== undefined && raw.trim() !== "") {
        const num = Number(raw.trim());
        if (!isNaN(num)) {
          numericReadings[f.name] = num;
        } else {
          allValid = false;
        }
      }
    }
    const serialized = JSON.stringify(numericReadings) + ":" + String(allValid);
    if (serialized !== lastEmittedRef.current) {
      lastEmittedRef.current = serialized;
      onChangeRef.current(numericReadings, allValid);
    }
  }, [values, fields]);

  // Validate an individual field against hard bounds and normal bands
  const validateField = (fieldName: string, rawVal: string): { error?: string; warning?: string } => {
    const f = fields.find((item) => item.name === fieldName);
    if (!f) return {};

    const trimmed = rawVal.trim();
    if (trimmed === "") {
      // Empty fields are allowed as missing; if required, error can be asserted on submit
      return {};
    }

    const num = Number(trimmed);
    if (isNaN(num)) {
      return {
        error: `Field '${f.label}' (${f.unit}) must be a valid numeric float.`,
      };
    }

    const [lo, hi] = f.hardBounds;
    if (num < lo || num > hi) {
      return {
        error: `Field '${f.label}' (${f.unit}) value ${num} is outside allowed physical limits [${lo}, ${hi}] ${f.unit}.`,
      };
    }

    const [normLo, normHi] = f.normalBand;
    if (num < normLo || num > normHi) {
      return {
        warning: `Reading ${num} ${f.unit} is outside optimal agronomic target band [${normLo}, ${normHi}] ${f.unit}.`,
      };
    }

    return {};
  };

  const handleInputChange = (fieldName: string, rawValue: string) => {
    setValues((prev) => ({ ...prev, [fieldName]: rawValue }));

    // Run validation if field has been touched
    if (touched[fieldName]) {
      const { error, warning } = validateField(fieldName, rawValue);
      setErrors((prev) => ({ ...prev, [fieldName]: error || "" }));
      setWarnings((prev) => ({ ...prev, [fieldName]: warning || "" }));
    }
  };

  const handleBlur = (fieldName: string) => {
    setTouched((prev) => ({ ...prev, [fieldName]: true }));
    const { error, warning } = validateField(fieldName, values[fieldName] || "");
    setErrors((prev) => ({ ...prev, [fieldName]: error || "" }));
    setWarnings((prev) => ({ ...prev, [fieldName]: warning || "" }));
  };

  const handleReset = () => {
    const empty: Record<string, string> = {};
    for (const f of fields) {
      empty[f.name] = "";
    }
    setValues(empty);
    setErrors({});
    setWarnings({});
    setTouched({});
    if (onReset) onReset();
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Validate all fields
    const newErrors: Record<string, string> = {};
    const newWarnings: Record<string, string> = {};
    let hasHardError = false;

    for (const f of fields) {
      const val = values[f.name] || "";
      const { error, warning } = validateField(f.name, val);
      if (error) {
        newErrors[f.name] = error;
        hasHardError = true;
      }
      if (warning) {
        newWarnings[f.name] = warning;
      }
    }

    setErrors(newErrors);
    setWarnings(newWarnings);

    // Mark all as touched
    const allTouched: Record<string, boolean> = {};
    for (const f of fields) allTouched[f.name] = true;
    setTouched(allTouched);

    if (hasHardError) {
      return;
    }

    // Build payload: EMPTY FIELDS ARE EXCLUDED (SENT AS MISSING, NEVER COERCED TO 0)
    // (Acceptance Criterion 1)
    const payload: Record<string, number> = {};
    for (const f of fields) {
      const raw = (values[f.name] || "").trim();
      if (raw !== "") {
        payload[f.name] = parseFloat(raw);
      }
    }

    // Check that at least one reading is provided
    if (Object.keys(payload).length === 0) {
      if (initialValues && Object.keys(initialValues).length > 0) {
        for (const [k, v] of Object.entries(initialValues)) {
          if (v !== undefined && v !== null) payload[k] = Number(v);
        }
      }
    }

    if (Object.keys(payload).length === 0) {
      setErrors((prev) => ({
        ...prev,
        [fields[0].name]: `At least one valid sensor reading (${fields[0].unit}) must be provided. You can click 'Load Target Band Baseline' below.`,
      }));
      return;
    }

    onSubmit(payload, provenance);
  };

  return (
    <Card className={`p-6 border border-[var(--border)] bg-[var(--surface)] shadow-sm ${className}`}>
      <form onSubmit={handleSubmit} noValidate className="space-y-6">
        {/* Header & Provenance Toggle */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[var(--border)] pb-4">
          <div>
            <h3 className="text-lg font-serif font-bold text-[var(--forest)] capitalize">
              {crop} Hydroponic Sensor Telemetry
            </h3>
            <p className="text-xs text-[var(--stone)] mt-0.5">
              Enter current water reservoir parameters. Empty fields remain missing (never coerced to 0).
            </p>
          </div>

          {/* Provenance Toggle (Manual vs Device / Live) */}
          <div className="flex items-center gap-1.5 p-1 rounded-[var(--radius-md)] bg-[var(--sand)]/60 border border-[var(--border)] text-xs">
            <button
              type="button"
              onClick={() => setProvenance("manual")}
              className={`px-3 py-1 rounded-[var(--radius-sm)] font-medium transition-colors ${
                provenance === "manual"
                  ? "bg-[var(--surface)] text-[var(--forest)] font-semibold shadow-xs"
                  : "text-[var(--stone)] hover:text-[var(--bark)]"
              }`}
            >
              Manual Lab Reading
            </button>
            <button
              type="button"
              onClick={() => setProvenance("live")}
              className={`px-3 py-1 rounded-[var(--radius-sm)] font-medium transition-colors flex items-center gap-1.5 ${
                provenance === "live"
                  ? "bg-[var(--surface)] text-[var(--forest)] font-semibold shadow-xs"
                  : "text-[var(--stone)] hover:text-[var(--bark)]"
              }`}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--success)] animate-pulse" />
              Connected Device
            </button>
          </div>
        </div>

        {/* Dynamic Fields Grid matching crop schema order */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {fields.map((f) => {
            const hasError = Boolean(errors[f.name]);
            const hasWarning = Boolean(warnings[f.name]) && !hasError;
            const currentVal = values[f.name] || "";

            return (
              <div key={f.name} className="space-y-1.5">
                <div className="flex items-baseline justify-between">
                  <label
                    htmlFor={`sensor-field-${f.name}`}
                    className="text-xs font-semibold text-[var(--bark)]"
                  >
                    {f.label}{" "}
                    <span className="font-mono text-[11px] text-[var(--stone)] font-normal">
                      ({f.unit})
                    </span>
                  </label>
                  <span className="text-[10px] text-[var(--stone)] font-mono">
                    Target: {f.normalBand[0]}–{f.normalBand[1]} {f.unit}
                  </span>
                </div>

                <div className="relative">
                  <input
                    id={`sensor-field-${f.name}`}
                    name={f.name}
                    type="number"
                    step={f.step || "any"}
                    placeholder={f.placeholder}
                    value={currentVal}
                    onChange={(e) => handleInputChange(f.name, e.target.value)}
                    onBlur={() => handleBlur(f.name)}
                    disabled={isLoading}
                    aria-invalid={hasError}
                    aria-describedby={`hint-${f.name}`}
                    className={`w-full px-3 py-2 text-sm rounded-[var(--radius-md)] border bg-[var(--surface)] font-mono transition-colors focus:outline-none focus:ring-2 ${
                      hasError
                        ? "border-[var(--error)] focus:ring-[var(--error)]/40 bg-[#FFF5F5]"
                        : hasWarning
                        ? "border-[var(--warning)]/60 focus:ring-[var(--warning)]/40 bg-[#FFFDF7]"
                        : "border-[var(--border)] focus:ring-[var(--moss)]/40 focus:border-[var(--moss)]"
                    }`}
                  />
                  <span className="absolute right-3 top-2.5 text-xs text-[var(--stone)] pointer-events-none font-mono">
                    {f.unit}
                  </span>
                </div>

                {/* Inline error naming field and unit (Acceptance Criterion 3) */}
                {hasError && (
                  <p
                    id={`hint-${f.name}`}
                    role="alert"
                    className="text-xs text-[var(--error)] font-medium flex items-center gap-1 mt-1"
                  >
                    <svg className="w-3.5 h-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                        clipRule="evenodd"
                      />
                    </svg>
                    <span>{errors[f.name]}</span>
                  </p>
                )}

                {/* Inline warning for boundary / sub-optimal readings */}
                {hasWarning && (
                  <p
                    id={`hint-${f.name}`}
                    className="text-xs text-[#8A5A00] flex items-center gap-1 mt-1"
                  >
                    <svg className="w-3.5 h-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                        clipRule="evenodd"
                      />
                    </svg>
                    <span>{warnings[f.name]}</span>
                  </p>
                )}

                {!hasError && !hasWarning && (
                  <p id={`hint-${f.name}`} className="text-[11px] text-[var(--stone)] truncate">
                    Hard limits: [{f.hardBounds[0]}, {f.hardBounds[1]}] {f.unit}
                  </p>
                )}
              </div>
            );
          })}
        </div>

        {/* Action Buttons: Reset & Submit */}
        <div className="flex items-center justify-between pt-4 border-t border-[var(--border)]">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleFillDefaults}
              disabled={isLoading}
              className="px-3 py-1.5 rounded-[var(--radius-md)] text-xs font-semibold text-[var(--forest)] hover:bg-[var(--sand)] transition-colors focus:outline-none"
            >
              Load Target Band Baseline
            </button>
            <span className="text-[var(--stone)]/30">|</span>
            <button
              type="button"
              onClick={handleReset}
              disabled={isLoading}
              className="px-3 py-1.5 rounded-[var(--radius-md)] text-xs font-medium text-[var(--stone)] hover:text-[var(--bark)] hover:bg-[var(--sand)] transition-colors focus:outline-none"
            >
              Reset Fields
            </button>
          </div>

          <Button
            type="submit"
            disabled={isLoading}
            className="px-6 py-2 bg-[var(--forest)] text-[#FAF6EC] hover:bg-[var(--moss)] font-semibold text-xs rounded-[var(--radius-md)] transition-colors"
          >
            {isLoading ? "Running Prediction..." : "Predict Hydroponic State →"}
          </Button>
        </div>
      </form>
    </Card>
  );
};

