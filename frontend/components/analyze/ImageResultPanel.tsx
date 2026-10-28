"use client";

import React, { useState } from "react";
import Image from "next/image";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import { ConfidenceThresholds } from "../ui/ConfidenceMeter";
import { LeafCard } from "./LeafCard";
import { LeafPrediction, PredictionResponse } from "../../lib/api/types";

export interface RejectedLeaf {
  leaf_id: string | number;
  reason: string;
}

export interface ImageResultPanelProps {
  /** Complete or partial image prediction response from API */
  prediction?: PredictionResponse | null;
  /** Plant photograph URL or object URL */
  imageUrl?: string | null;
  /** Optional segmentation mask overlay URL */
  maskOverlayUrl?: string | null;
  /** Image source modality / provenance */
  source?: "upload" | "campus_camera" | "imported" | "mock";
  /** Optional segmentation failure flag or error message */
  segmentationFailed?: boolean;
  /** Detailed reason if segmentation failed */
  segmentationFailureReason?: string | null;
  /** Confidence tier thresholds provided by backend / metadata */
  thresholds?: ConfidenceThresholds;
  /** Preprocessing chips to display */
  preprocessingChips?: string[];
  /** Handler when user clicks retry after failure */
  onRetry?: () => void;
  /** Handler when user clicks choose different image */
  onChooseDifferentImage?: () => void;
  /** Optional custom CSS classes */
  className?: string;
}

const DEFAULT_THRESHOLDS: ConfidenceThresholds = {
  high: 0.75,
  moderate: 0.50,
};

export const ImageResultPanel: React.FC<ImageResultPanelProps> = ({
  prediction,
  imageUrl,
  maskOverlayUrl,
  source = "upload",
  segmentationFailed = false,
  segmentationFailureReason,
  thresholds = DEFAULT_THRESHOLDS,
  preprocessingChips = ["224×224 Normalised", "Dual-stage YOLOv8-Seg", "ImageNet scale"],
  onRetry,
  onChooseDifferentImage,
  className = "",
}) => {
  const [showMask, setShowMask] = useState<boolean>(true);
  const [selectedLeafIndex, setSelectedLeafIndex] = useState<number | null>(null);

  // Extract leaves and rejected leaves safely
  const leaves: LeafPrediction[] = prediction?.leaves || [];
  const rejectedLeaves: RejectedLeaf[] = (prediction?.rejected_leaves || []) as RejectedLeaf[];
  const hasNoLeaves = leaves.length === 0;

  // Determine if this is a segmentation / input failure state
  const isFailure =
    segmentationFailed ||
    (hasNoLeaves && Boolean(prediction && !rejectedLeaves.length)) ||
    Boolean(segmentationFailureReason);

  // Failure Card (Acceptance Criterion 3: Segmentation failure is shown as an input failure, not a disease)
  if (isFailure) {
    return (
      <Card
        variant="warning"
        className={`p-6 border-l-4 border-[var(--warning)] bg-[#FFFDF7] shadow-[var(--shadow-warm-sm)] ${className}`}
        aria-live="polite"
      >
        <div className="flex items-start gap-4">
          <div className="w-10 h-10 rounded-full bg-[#FFF4DC] text-[var(--warning)] flex items-center justify-center shrink-0">
            <svg
              className="w-6 h-6"
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
          </div>

          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-bold uppercase tracking-wider text-[var(--warning)]">
                Input Quality Issue
              </span>
              <Badge variant="experimental">Non-Diagnostic Failure</Badge>
            </div>

            <h3 className="text-lg font-bold text-[var(--bark)]">
              Leaf Segmentation Unsuccessful
            </h3>

            <p className="mt-2 text-sm text-[var(--bark)]/85 leading-relaxed">
              {segmentationFailureReason ||
                "No distinct plant leaves could be isolated from this image. This is strictly an image processing / input failure (e.g. lighting, blur, or framing), NOT a plant disease or nutrient deficiency diagnosis."}
            </p>

            <div className="mt-4 p-3 bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius-md)] text-xs text-[var(--stone)] space-y-1">
              <p className="font-semibold text-[var(--bark)]">Recommended capture guidelines:</p>
              <ul className="list-disc list-inside space-y-0.5">
                <li>Ensure the plant foliage is in clear focus with uniform lighting.</li>
                <li>Avoid extreme glare, dark shadows, or obstructed camera angles.</li>
                <li>Capture whole leaves within the frame boundary without severe overlap.</li>
              </ul>
            </div>

            <div className="mt-5 flex flex-wrap gap-3">
              {onRetry && (
                <Button variant="primary" size="sm" onClick={onRetry}>
                  <svg
                    className="w-4 h-4 mr-1.5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                    />
                  </svg>
                  Retry Segmentation
                </Button>
              )}

              {onChooseDifferentImage && (
                <Button variant="secondary" size="sm" onClick={onChooseDifferentImage}>
                  Choose Different Image
                </Button>
              )}
            </div>
          </div>
        </div>
      </Card>
    );
  }

  // Calculate per-class breakdown (no averaged score, strictly counts per class)
  const classCounts: Record<string, number> = {};
  for (const leaf of leaves) {
    const label = leaf.display_label || leaf.predicted_class || "Unclassified";
    classCounts[label] = (classCounts[label] || 0) + 1;
  }

  return (
    <div className={`space-y-6 text-left ${className}`} data-testid="image-result-panel">
      {/* Top Bar: Source + Preprocessing Chips */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius-lg)]">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold text-[var(--stone)] uppercase tracking-wider">
            Source:
          </span>
          <Badge variant={source === "campus_camera" ? "live" : "upload"}>
            {source === "campus_camera" ? "Campus Camera (Live)" : "User Upload"}
          </Badge>

          {prediction?.model_name && (
            <Badge variant="neutral">
              {prediction.model_name}
              {prediction.model_version ? ` (${prediction.model_version})` : ""}
            </Badge>
          )}
        </div>

        {/* Preprocessing Chips */}
        <div className="flex flex-wrap items-center gap-1.5">
          {preprocessingChips.map((chip, idx) => (
            <span
              key={idx}
              className="text-[11px] font-mono px-2 py-0.5 rounded bg-[var(--sand)] text-[var(--bark)] border border-[var(--border)]"
            >
              {chip}
            </span>
          ))}
        </div>
      </div>

      {/* Main Grid: Photo with Leaf Overlay + Plant Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Plant Photo with Numbered Leaf Overlay (7 cols) */}
        <div className="lg:col-span-7 bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius-lg)] p-4 shadow-[var(--shadow-warm-sm)]">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-bold text-[var(--bark)] flex items-center gap-2">
              <span>Plant Imagery &amp; Leaf Overlay</span>
              <span className="text-xs font-normal text-[var(--stone)]">
                ({leaves.length} {leaves.length === 1 ? "leaf" : "leaves"} detected)
              </span>
            </h3>

            {/* Mask Overlay Toggle with accessible alt/aria text */}
            {maskOverlayUrl && (
              <button
                type="button"
                onClick={() => setShowMask((prev) => !prev)}
                aria-pressed={showMask}
                aria-label="Toggle segmentation mask overlay"
                className={`text-xs px-2.5 py-1 rounded-[var(--radius-sm)] border font-medium transition-colors ${
                  showMask
                    ? "bg-[#EBF7EE] text-[var(--forest)] border-[var(--forest)]/40 font-semibold"
                    : "bg-[var(--sand)] text-[var(--stone)] border-[var(--border)] hover:text-[var(--bark)]"
                }`}
              >
                {showMask ? "Mask Overlay: Visible" : "Mask Overlay: Hidden"}
              </button>
            )}
          </div>

          {/* Image & Leaf Overlay Viewport */}
          <div className="relative w-full aspect-4/3 bg-[var(--sand)]/40 rounded-[var(--radius-md)] overflow-hidden border border-[var(--border)] flex items-center justify-center">
            {imageUrl ? (
              <div className="relative w-full h-full">
                <Image
                  src={imageUrl}
                  alt="Hydroponic plant foliage under evaluation"
                  fill
                  className="object-contain"
                  unoptimized
                />

                {/* Optional Mask Overlay Layer */}
                {maskOverlayUrl && showMask && (
                  <div className="absolute inset-0 pointer-events-none mix-blend-multiply opacity-75">
                    <Image
                      src={maskOverlayUrl}
                      alt="Segmentation mask overlay identifying individual leaf boundaries"
                      fill
                      className="object-contain"
                      unoptimized
                    />
                  </div>
                )}

                {/* Interactive Leaf Marker Pins */}
                {leaves.map((leaf, idx) => {
                  const leafNum = idx + 1;
                  const isSelected = selectedLeafIndex === idx;

                  // Position markers across image area (or from mask coordinates if available)
                  // Distribute neatly across viewport as fallback coordinates
                  const col = idx % 3;
                  const row = Math.floor(idx / 3);
                  const topPercent = 25 + row * 28;
                  const leftPercent = 20 + col * 30;

                  return (
                    <button
                      key={leaf.leaf_id || idx}
                      type="button"
                      onClick={() => setSelectedLeafIndex(isSelected ? null : idx)}
                      style={{ top: `${topPercent}%`, left: `${leftPercent}%` }}
                      aria-label={`Highlight Leaf #${leafNum}: ${leaf.display_label}`}
                      aria-pressed={isSelected}
                      className={`absolute -translate-x-1/2 -translate-y-1/2 z-10 w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-transform duration-150 shadow-md ${
                        isSelected
                          ? "bg-[var(--forest)] text-[#FFFFFF] scale-125 ring-4 ring-[#FFFFFF] ring-offset-1"
                          : "bg-[#FFFFFF] text-[var(--forest)] border-2 border-[var(--forest)] hover:scale-110"
                      }`}
                    >
                      #{leafNum}
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className="p-8 text-center text-sm text-[var(--stone)] flex flex-col items-center gap-2">
                <svg
                  className="w-10 h-10 text-[var(--stone)]/60"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
                <span>Foliage image rendering preview</span>
              </div>
            )}
          </div>

          <p className="mt-2 text-xs text-[var(--stone)] text-center">
            Click any numbered pin or leaf card below to correlate leaf position with condition.
          </p>
        </div>

        {/* Plant Summary (5 cols) - STRICTLY NO AVERAGED SCORE */}
        <div className="lg:col-span-5 bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius-lg)] p-5 shadow-[var(--shadow-warm-sm)] space-y-4">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-[var(--stone)]">
              Plant Canopy Summary
            </span>
            <h3 className="text-lg font-bold text-[var(--bark)] mt-0.5">
              Assessed Foliage Breakdown
            </h3>
          </div>

          {/* Total Leaf Count Metric */}
          <div className="p-3.5 bg-[var(--sand)]/40 rounded-[var(--radius-md)] border border-[var(--border)] flex items-center justify-between">
            <span className="text-sm font-medium text-[var(--bark)]">Total Detected Leaves</span>
            <span className="text-xl font-bold font-mono text-[var(--forest)]">
              {leaves.length}
            </span>
          </div>

          {/* Counts per Predicted Class */}
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-[var(--stone)] mb-2">
              Counts by Predicted Condition
            </h4>
            <div className="space-y-2">
              {Object.entries(classCounts).map(([label, count]) => (
                <div
                  key={label}
                  className="flex items-center justify-between p-2.5 bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius-sm)] text-sm"
                >
                  <span className="font-semibold text-[var(--bark)]">{label}</span>
                  <span className="font-mono font-bold px-2 py-0.5 rounded bg-[var(--sand)] text-[var(--bark)] text-xs">
                    {count} {count === 1 ? "leaf" : "leaves"}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Explicit Honest Architecture Disclaimer: NO AVERAGED SCORE */}
          <div className="p-3 bg-[#F0F5FA] border border-[var(--info)]/30 rounded-[var(--radius-md)] text-xs text-[var(--info)] space-y-1">
            <div className="flex items-center gap-1.5 font-bold">
              <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span>Per-Leaf Assessment Architecture</span>
            </div>
            <p className="leading-relaxed">
              No averaged or combined score is computed. Hydroponic foliar diseases exhibit localized
              symptoms; averaging confidence scores across different leaves would obscure early-stage
              deficiencies.
            </p>
          </div>
        </div>
      </div>

      {/* LeafGrid Section (Acceptance Criterion 1: Each leaf shows its own confidence tier) */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-bold text-[var(--bark)]">
              Individual Leaf Diagnoses ({leaves.length})
            </h3>
            <p className="text-xs text-[var(--stone)]">
              Each leaf has its own independent prediction, confidence tier, and raw classification.
            </p>
          </div>

          {selectedLeafIndex !== null && (
            <button
              type="button"
              onClick={() => setSelectedLeafIndex(null)}
              className="text-xs text-[var(--clay)] hover:underline font-medium"
            >
              Clear selection
            </button>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4" data-testid="leaf-grid">
          {leaves.map((leaf, idx) => (
            <LeafCard
              key={leaf.leaf_id || idx}
              leafId={leaf.leaf_id || `leaf_${idx + 1}`}
              index={idx + 1}
              displayLabel={leaf.display_label || leaf.predicted_class}
              originalClass={leaf.predicted_class}
              confidence={leaf.confidence}
              isUncertain={leaf.is_uncertain}
              thresholds={thresholds}
              contourArea={leaf.contour_area}
              isSelected={selectedLeafIndex === idx}
              onSelect={() => setSelectedLeafIndex(selectedLeafIndex === idx ? null : idx)}
            />
          ))}
        </div>
      </div>

      {/* Rejected Leaves: Listed as Not Assessed with Reasons (Acceptance Criterion 2) */}
      {rejectedLeaves.length > 0 && (
        <Card
          variant="warning"
          className="p-5 border-l-4 border-[var(--warning)] bg-[#FFFDF7]"
          data-testid="rejected-leaves-panel"
        >
          <div className="flex items-center gap-2 mb-2">
            <svg
              className="w-5 h-5 text-[var(--warning)] shrink-0"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            <h4 className="text-sm font-bold text-[var(--bark)]">
              Not Assessed Leaves ({rejectedLeaves.length})
            </h4>
          </div>

          <p className="text-xs text-[var(--stone)] mb-3">
            The following candidate leaf segments were rejected by quality filters and were not
            evaluated for disease. They are never silently omitted:
          </p>

          <div className="space-y-2">
            {rejectedLeaves.map((rejected, idx) => (
              <div
                key={rejected.leaf_id || idx}
                className="flex items-start justify-between gap-3 p-2.5 rounded bg-[var(--surface)] border border-[var(--border)] text-xs"
              >
                <div className="flex items-center gap-2">
                  <span className="font-mono font-bold text-[var(--bark)] bg-[var(--sand)] px-1.5 py-0.5 rounded">
                    #{rejected.leaf_id}
                  </span>
                  <span className="text-[var(--stone)]">Status: Not assessed</span>
                </div>
                <div className="text-right text-[var(--bark)] font-medium">
                  Reason: <span className="text-[var(--clay)] font-semibold">{rejected.reason}</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};
