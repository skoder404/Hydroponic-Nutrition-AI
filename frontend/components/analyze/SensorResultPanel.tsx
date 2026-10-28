"use client";

import React from "react";
import { Badge } from "../ui/Badge";
import { Card } from "../ui/Card";
import { ConfidenceMeter, ConfidenceThresholds } from "../ui/ConfidenceMeter";
import { UncertaintyBadge } from "../ui/UncertaintyBadge";
import { CROP_SENSOR_SCHEMAS, SupportedCrop } from "./SensorForm";
import { PredictionResponse } from "../../lib/api/types";

export interface EvidenceRow {
  parameter: string;
  label?: string;
  reading: number;
  unit: string;
  referenceBand: [number, number];
  isOutOfBand?: boolean;
}

export interface SensorResultPanelProps {
  crop: SupportedCrop | string;
  prediction?: PredictionResponse | null;
  readings?: Record<string, number | undefined> | null;
  thresholds?: ConfidenceThresholds;
  className?: string;
}

const DEFAULT_THRESHOLDS: ConfidenceThresholds = {
  high: 0.75,
  moderate: 0.50,
};

export const SensorResultPanel: React.FC<SensorResultPanelProps> = ({
  crop,
  prediction,
  readings,
  thresholds = DEFAULT_THRESHOLDS,
  className = "",
}) => {
  const normalizedCrop = (crop || "lettuce").toLowerCase() as SupportedCrop;
  const isTomato = normalizedCrop === "tomato";
  const isExperimental = isTomato || Boolean(prediction?.evidence?.experimental);

  const displayLabel = prediction?.display_label || prediction?.predicted_class || "Sensor Analysis Complete";
  const rawClass = prediction?.predicted_class || "Normal";
  const confidence = prediction?.confidence ?? 0.85;
  const isUncertain = prediction?.is_uncertain || false;
  const modelName = prediction?.model_name || `${normalizedCrop}_sensor_v1.0.0`;

  // Build evidence rows from CROP_SENSOR_SCHEMAS and passed readings
  const schemaFields = CROP_SENSOR_SCHEMAS[normalizedCrop] || CROP_SENSOR_SCHEMAS.lettuce;
  const inputReadings = (readings || prediction?.evidence?.inputs || {}) as Record<string, number | undefined>;

  const evidenceRows: EvidenceRow[] = schemaFields.map((field) => {
    // Attempt match by field.name or variant
    const val =
      inputReadings[field.name] ??
      inputReadings[`${field.name}_reading`] ??
      (prediction?.evidence as Record<string, unknown> | undefined)?.[field.name];

    const numVal = typeof val === "number" ? val : 0;
    const [minBand, maxBand] = field.normalBand;
    const isOutOfBand = numVal < minBand || numVal > maxBand;

    return {
      parameter: field.name,
      label: field.label,
      reading: numVal,
      unit: field.unit,
      referenceBand: field.normalBand,
      isOutOfBand,
    };
  });

  return (
    <Card
      variant="default"
      className={`p-6 bg-[var(--surface)] border border-[var(--border)] shadow-[var(--shadow-warm-sm)] text-left space-y-5 ${className}`}
      data-testid="sensor-result-panel"
    >
      {/* Top Header: Modality + Model Version + Uncertainty */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] pb-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-wider text-[var(--stone)]">
            Sensor Telemetry
          </span>
          <span className="capitalize font-mono text-xs px-2 py-0.5 rounded bg-[var(--sand)] text-[var(--bark)] border border-[var(--border)]">
            {normalizedCrop}
          </span>
          <span className="font-mono text-xs text-[var(--stone)]">
            {modelName}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Acceptance Criterion 1: Tomato shows experimental badge */}
          {isExperimental && (
            <Badge variant="experimental">Experimental Model</Badge>
          )}
          <UncertaintyBadge
            score={confidence}
            thresholds={thresholds}
            isUncertain={isUncertain}
          />
        </div>
      </div>

      {/* Acceptance Criterion 1: Tomato shows the experimental banner */}
      {isTomato && (
        <div
          className="p-3.5 bg-[#FFF4DC] border-l-4 border-[var(--warning)] rounded-[var(--radius-md)] text-xs text-[var(--warning)] shadow-xs"
          data-testid="tomato-experimental-banner"
          role="alert"
        >
          <div className="flex items-start gap-2.5">
            <svg
              className="w-5 h-5 shrink-0 text-[var(--warning)]"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            <div>
              <span className="font-bold block uppercase tracking-wide text-[11px]">
                Tomato Experimental Sensor Model
              </span>
              <p className="mt-0.5 leading-relaxed text-[#7A4B00]">
                Tomato sensor models utilize transformed features and synthetically derived water temperature.
                Campus and live sensor inference is experimental and not field-validated until transformed features
                are mapped to certified physical device measurements (Overview §28.5, §28.16).
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Main Condition Classification */}
      <div className="space-y-2">
        <span className="text-xs uppercase tracking-wider text-[var(--stone)] font-bold">
          Predicted Condition State
        </span>
        <div className="flex flex-wrap items-baseline gap-3">
          <h3 className="text-xl font-bold text-[var(--bark)]">
            {displayLabel}
          </h3>
          {rawClass !== displayLabel && (
            <span className="text-xs font-mono text-[var(--stone)] bg-[var(--sand)] px-2 py-0.5 rounded">
              Raw: {rawClass}
            </span>
          )}
        </div>

        <ConfidenceMeter
          score={confidence}
          thresholds={thresholds}
          isUncertain={isUncertain}
          label="Telemetry Confidence"
        />
      </div>

      {/* Acceptance Criterion 3: Label is described as rule-derived */}
      <div className="p-3 bg-[#F7F4EC] border border-[var(--clay)]/30 rounded-[var(--radius-md)] text-xs text-[var(--clay)] space-y-1">
        <div className="flex items-center gap-1.5 font-bold">
          <Badge variant="rules">Rule-Derived Target</Badge>
          <span>Agronomic Baseline Protocol</span>
        </div>
        <p className="leading-relaxed">
          This foliar and nutrient label is described as rule-derived. The target state was computed from
          threshold reference bands rather than biologically observed tissue pathology (Overview §28.1).
        </p>
      </div>

      {/* Acceptance Criterion 2: Evidence rows show reading, unit, and reference band */}
      <div className="space-y-3 pt-2">
        <h4 className="text-xs font-bold uppercase tracking-wider text-[var(--stone)]">
          Telemetry Evidence &amp; Reference Bands
        </h4>

        <div className="rounded-[var(--radius-md)] border border-[var(--border)] overflow-hidden">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-[var(--sand)]/50 border-b border-[var(--border)] font-semibold text-[var(--stone)]">
              <tr>
                <th className="px-3.5 py-2.5">Parameter</th>
                <th className="px-3 py-2.5 text-right">Observed Reading</th>
                <th className="px-3 py-2.5 text-right">Reference Band</th>
                <th className="px-3.5 py-2.5 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--border)]/60 text-[var(--bark)]">
              {evidenceRows.map((row) => (
                <tr key={row.parameter} className="hover:bg-[var(--sand)]/20">
                  <td className="px-3.5 py-2">
                    <span className="font-semibold block">{row.label || row.parameter}</span>
                    <span className="font-mono text-[10px] text-[var(--stone)] uppercase">
                      {row.parameter}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right font-mono font-bold">
                    {row.reading.toFixed(2)} {row.unit}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-[var(--stone)]">
                    {row.referenceBand[0].toFixed(1)} – {row.referenceBand[1].toFixed(1)} {row.unit}
                  </td>
                  <td className="px-3.5 py-2 text-right">
                    {row.isOutOfBand ? (
                      <span className="inline-flex items-center gap-1 font-semibold text-[var(--warning)] bg-[#FFF4DC] px-2 py-0.5 rounded text-[11px]">
                        Deviation
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 font-semibold text-[var(--forest)] bg-[#EBF7EE] px-2 py-0.5 rounded text-[11px]">
                        Optimal
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Card>
  );
};
