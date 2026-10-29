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
  const [showAnnotations, setShowAnnotations] = useState<boolean>(true);
  const [showConfidence, setShowConfidence] = useState<boolean>(true);
  const [showMask, setShowMask] = useState<boolean>(true);
  const [selectedLeafIndex, setSelectedLeafIndex] = useState<number | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [aspectRatio, setAspectRatio] = useState<number | null>(null);

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
  let deficientCount = 0;
  let healthyCount = 0;

  for (const leaf of leaves) {
    const label = leaf.display_label || leaf.predicted_class || "Unclassified";
    classCounts[label] = (classCounts[label] || 0) + 1;
    const isLeafHealthy =
      (leaf.predicted_class || "").toLowerCase().includes("healthy") ||
      (leaf.display_label || "").toLowerCase().includes("healthy");
    if (isLeafHealthy) {
      healthyCount += 1;
    } else {
      deficientCount += 1;
    }
  }

  // Fallback bounding box layout if bbox coordinates are not returned by the backend
  const getFallbackBox = (index: number, total: number) => {
    if (total === 1) {
      return { top: 12, left: 14, width: 72, height: 74 };
    }
    if (total === 2) {
      if (index === 0) return { top: 15, left: 10, width: 38, height: 68 };
      return { top: 18, left: 52, width: 38, height: 68 };
    }
    if (total === 3) {
      if (index === 0) return { top: 14, left: 14, width: 38, height: 40 };
      if (index === 1) return { top: 14, left: 52, width: 38, height: 42 };
      return { top: 54, left: 28, width: 44, height: 38 };
    }
    const cols = Math.min(3, total);
    const col = index % cols;
    const row = Math.floor(index / cols);
    const boxW = Math.max(22, Math.floor(80 / cols));
    const boxH = Math.max(22, 34);
    return {
      top: 12 + row * (boxH + 6),
      left: 10 + col * (boxW + 6),
      width: boxW,
      height: boxH,
    };
  };

  return (
    <div className={`space-y-6 text-left w-full ${className}`} data-testid="image-result-panel">
      {/* 1. TOP BAR: Provenance, Model, Interactive Overlays Toggle */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius-lg)] shadow-[var(--shadow-warm-sm)]">
        <div className="flex flex-wrap items-center gap-2.5">
          <span className="text-xs font-bold text-[var(--stone)] uppercase tracking-wider">
            Image Source:
          </span>
          <Badge variant={source === "campus_camera" ? "live" : "upload"}>
            {source === "campus_camera" ? "Campus Camera (Live Stream)" : "User Foliar Upload"}
          </Badge>

          {prediction?.model_name && (
            <Badge variant="neutral">
              {prediction.model_name}
              {prediction.model_version ? ` · v${prediction.model_version}` : ""}
            </Badge>
          )}
        </div>

        {/* View Controls & Preprocessing Badges */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Toggle Bounding Boxes & In-Picture Labels */}
          <button
            type="button"
            onClick={() => setShowAnnotations((prev) => !prev)}
            aria-pressed={showAnnotations}
            className={`text-xs px-3 py-1.5 rounded-[var(--radius-md)] border font-semibold transition-all duration-150 flex items-center gap-1.5 ${
              showAnnotations
                ? "bg-[var(--forest)] text-[#FFFFFF] border-[var(--forest)] shadow-xs"
                : "bg-[var(--sand)] text-[var(--stone)] border-[var(--border)] hover:text-[var(--bark)]"
            }`}
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 12h16M4 18h7"
              />
            </svg>
            <span>{showAnnotations ? "In-Picture Tags: ON" : "In-Picture Tags: OFF"}</span>
          </button>

          {/* Toggle Confidence % in Labels */}
          {showAnnotations && (
            <button
              type="button"
              onClick={() => setShowConfidence((prev) => !prev)}
              aria-pressed={showConfidence}
              className={`text-xs px-2.5 py-1.5 rounded-[var(--radius-md)] border font-medium transition-colors ${
                showConfidence
                  ? "bg-[#EBF7EE] text-[var(--forest)] border-[var(--forest)]/40 font-semibold"
                  : "bg-[var(--surface)] text-[var(--stone)] border-[var(--border)] hover:text-[var(--bark)]"
              }`}
            >
              {showConfidence ? "Confidence %: Visible" : "Confidence %: Hidden"}
            </button>
          )}

          {/* Optional Segmentation Mask Toggle */}
          {maskOverlayUrl && (
            <button
              type="button"
              onClick={() => setShowMask((prev) => !prev)}
              aria-pressed={showMask}
              className={`text-xs px-2.5 py-1.5 rounded-[var(--radius-md)] border font-medium transition-colors ${
                showMask
                  ? "bg-[#EBF7EE] text-[var(--forest)] border-[var(--forest)]/40 font-semibold"
                  : "bg-[var(--sand)] text-[var(--stone)] border-[var(--border)]"
              }`}
            >
              Mask: {showMask ? "Visible" : "Hidden"}
            </button>
          )}
        </div>
      </div>

      {/* 2. LARGE EXPANSIVE PLANT PHOTO & BOUNDING BOX VISUALIZATION */}
      <div className="bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius-xl)] p-5 sm:p-6 shadow-[var(--shadow-warm-md)] space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] pb-3">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-[var(--clay)]">
              Foliar Diagnostic Studio
            </span>
            <h3 className="text-lg sm:text-xl font-bold text-[var(--bark)] mt-0.5">
              Whole Plant Canopy &amp; Localized Deficiencies
            </h3>
            <p className="text-xs text-[var(--stone)] mt-0.5">
              Identified leaves and deficient patches are framed with bounding boxes, calibrated confidence, and pathology labels directly on the photo.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-[var(--sand)] text-[var(--bark)] border border-[var(--border)]">
              <span className="w-2 h-2 rounded-full bg-[var(--forest)] animate-pulse" />
              {leaves.length} {leaves.length === 1 ? "Leaf" : "Leaves"} Assessed
            </span>
            {deficientCount > 0 && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold bg-[#FEE2E2] text-[#B91C1C] border border-[#FCA5A5]">
                ⚠ {deficientCount} Deficiency Detected
              </span>
            )}
          </div>
        </div>

        {/* Hero Image Container: Large, Centered, High Contrast */}
        <div className="relative w-full min-h-[460px] sm:min-h-[560px] lg:min-h-[640px] max-h-[820px] bg-[#0E1511] rounded-[var(--radius-lg)] overflow-hidden border border-[#233528] flex items-center justify-center p-3 sm:p-4">
          {imageUrl ? (
            <div
              className="relative mx-auto rounded-[var(--radius-md)] overflow-hidden shadow-2xl transition-all duration-300"
              style={{
                aspectRatio: aspectRatio ? `${aspectRatio}` : "4/3",
                maxHeight: "760px",
                width: "100%",
                maxWidth: aspectRatio && aspectRatio < 0.9 ? "540px" : "920px",
              }}
            >
              {/* Plant Image Element */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={imageUrl}
                alt="Hydroponic plant foliage evaluated for nutrient deficiency"
                className="w-full h-full object-contain block select-none"
                onLoad={(e) => {
                  const img = e.currentTarget;
                  if (img.naturalWidth && img.naturalHeight) {
                    setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
                    setAspectRatio(img.naturalWidth / img.naturalHeight);
                  }
                }}
              />

              {/* Optional Mask Overlay Layer */}
              {maskOverlayUrl && showMask && (
                <div className="absolute inset-0 pointer-events-none mix-blend-screen opacity-70">
                  <Image
                    src={maskOverlayUrl}
                    alt="Segmentation mask overlay"
                    fill
                    className="object-contain"
                    unoptimized
                  />
                </div>
              )}

              {/* BOUNDING BOXES & IN-PICTURE LABELS OVERLAY */}
              {showAnnotations &&
                leaves.map((leaf, idx) => {
                  const leafNum = idx + 1;
                  const isSelected = selectedLeafIndex === idx;

                  const isHealthy =
                    (leaf.predicted_class || "").toLowerCase().includes("healthy") ||
                    (leaf.display_label || "").toLowerCase().includes("healthy");
                  const isUncertain = leaf.is_uncertain || leaf.uncertain || false;
                  const isDeficient = !isHealthy && !isUncertain;

                  // Resolve coordinates
                  let boxTop = 0;
                  let boxLeft = 0;
                  let boxWidth = 0;
                  let boxHeight = 0;

                  if (leaf.bbox && leaf.bbox.length === 4) {
                    const [x1, y1, x2, y2] = leaf.bbox;
                    if (x2 <= 1.0 && y2 <= 1.0) {
                      boxLeft = Math.max(0, x1 * 100);
                      boxTop = Math.max(0, y1 * 100);
                      boxWidth = Math.min(100 - boxLeft, (x2 - x1) * 100);
                      boxHeight = Math.min(100 - boxTop, (y2 - y1) * 100);
                    } else if (naturalSize && naturalSize.width > 0 && naturalSize.height > 0) {
                      boxLeft = Math.max(0, (x1 / naturalSize.width) * 100);
                      boxTop = Math.max(0, (y1 / naturalSize.height) * 100);
                      boxWidth = Math.min(100 - boxLeft, ((x2 - x1) / naturalSize.width) * 100);
                      boxHeight = Math.min(100 - boxTop, ((y2 - y1) / naturalSize.height) * 100);
                    } else {
                      const fb = getFallbackBox(idx, leaves.length);
                      boxTop = fb.top;
                      boxLeft = fb.left;
                      boxWidth = fb.width;
                      boxHeight = fb.height;
                    }
                  } else {
                    const fb = getFallbackBox(idx, leaves.length);
                    boxTop = fb.top;
                    boxLeft = fb.left;
                    boxWidth = fb.width;
                    boxHeight = fb.height;
                  }

                  // Border and tag color styling
                  const borderClasses = isHealthy
                    ? isSelected
                      ? "border-emerald-400 bg-emerald-500/20 shadow-[0_0_20px_rgba(16,185,129,0.7)] ring-4 ring-emerald-300/40"
                      : "border-emerald-500 bg-emerald-500/10 shadow-[0_0_12px_rgba(16,185,129,0.45)] hover:border-emerald-300 hover:bg-emerald-500/20"
                    : isUncertain
                    ? isSelected
                      ? "border-purple-400 bg-purple-500/25 shadow-[0_0_20px_rgba(139,92,246,0.7)] ring-4 ring-purple-300/40"
                      : "border-purple-500 bg-purple-500/15 shadow-[0_0_12px_rgba(139,92,246,0.45)] hover:border-purple-300 hover:bg-purple-500/20"
                    : isSelected
                    ? "border-rose-400 bg-rose-600/30 shadow-[0_0_25px_rgba(244,63,94,0.85)] ring-4 ring-rose-400/50"
                    : "border-rose-500 bg-rose-600/15 shadow-[0_0_15px_rgba(239,68,68,0.55)] hover:border-rose-400 hover:bg-rose-600/25";

                  const tagBg = isHealthy
                    ? "bg-emerald-600 text-[#FFFFFF] border-emerald-400"
                    : isUncertain
                    ? "bg-purple-700 text-[#FFFFFF] border-purple-400"
                    : "bg-rose-700 text-[#FFFFFF] border-rose-400";

                  const confPercent =
                    leaf.confidence !== null && leaf.confidence !== undefined
                      ? (leaf.confidence * 100).toFixed(1)
                      : null;
                  const isNearTop = boxTop < 7;

                  return (
                    <div
                      key={leaf.leaf_id || idx}
                      onClick={() => setSelectedLeafIndex(isSelected ? null : idx)}
                      style={{
                        top: `${boxTop}%`,
                        left: `${boxLeft}%`,
                        width: `${boxWidth}%`,
                        height: `${boxHeight}%`,
                      }}
                      className={`absolute border-2 sm:border-3 rounded-lg sm:rounded-xl cursor-pointer transition-all duration-200 z-10 ${borderClasses}`}
                    >
                      {/* IN-PICTURE IDENTIFIED CLASS & CONFIDENCE BADGE */}
                      <div
                        className={`absolute ${
                          isNearTop ? "top-1.5 left-1.5" : "-top-7 sm:-top-8 left-0"
                        } flex items-center gap-1.5 px-2.5 py-1 rounded-md sm:rounded-lg text-[11px] sm:text-xs font-bold border shadow-xl backdrop-blur-md z-30 whitespace-nowrap transition-transform duration-150 ${tagBg} ${
                          isSelected ? "scale-105 ring-2 ring-white" : ""
                        }`}
                      >
                        <span className="font-mono bg-black/30 px-1 rounded text-[10px] sm:text-xs">
                          #{leafNum}
                        </span>

                        <span>
                          {isHealthy ? "✓ " : isDeficient ? "⚠ " : "? "}
                          {leaf.display_label || leaf.predicted_class}
                        </span>

                        {showConfidence && confPercent !== null && (
                          <span className="font-mono bg-black/40 px-1.5 py-0.5 rounded text-[10px] sm:text-[11px] font-semibold text-white/95">
                            {confPercent}%
                          </span>
                        )}
                      </div>

                      {/* Corner Target Markers */}
                      <div className="absolute top-1 left-1 w-2.5 h-2.5 border-t-2 border-l-2 border-white/80 pointer-events-none" />
                      <div className="absolute top-1 right-1 w-2.5 h-2.5 border-t-2 border-r-2 border-white/80 pointer-events-none" />
                      <div className="absolute bottom-1 left-1 w-2.5 h-2.5 border-b-2 border-l-2 border-white/80 pointer-events-none" />
                      <div className="absolute bottom-1 right-1 w-2.5 h-2.5 border-b-2 border-r-2 border-white/80 pointer-events-none" />
                    </div>
                  );
                })}
            </div>
          ) : (
            <div className="p-12 text-center text-sm text-[var(--stone)] flex flex-col items-center gap-3">
              <svg
                className="w-12 h-12 text-[var(--stone)]/60"
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
              <span className="font-medium">No foliar image available for preview</span>
            </div>
          )}
        </div>

        {/* 3. ALIGNED CANOPY SUMMARY CARDS & METRICS STRIP */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5 pt-2">
          {/* Card 1: Total Detected */}
          <div className="p-3.5 bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius-lg)] shadow-xs">
            <span className="text-[11px] font-bold uppercase tracking-wider text-[var(--stone)] block">
              Leaves Detected
            </span>
            <div className="flex items-baseline justify-between mt-1">
              <span className="text-2xl font-bold font-mono text-[var(--forest)]">
                {leaves.length}
              </span>
              <span className="text-xs text-[var(--stone)]">
                {leaves.length === 1 ? "Single leaf" : "Multi-leaf canopy"}
              </span>
            </div>
          </div>

          {/* Card 2: Deficient / Pathology Findings */}
          <div
            className={`p-3.5 border rounded-[var(--radius-lg)] shadow-xs ${
              deficientCount > 0
                ? "bg-[#FEF2F2] border-[#FCA5A5]"
                : "bg-[var(--surface)] border-[var(--border)]"
            }`}
          >
            <span
              className={`text-[11px] font-bold uppercase tracking-wider block ${
                deficientCount > 0 ? "text-[#B91C1C]" : "text-[var(--stone)]"
              }`}
            >
              Deficiencies Detected
            </span>
            <div className="flex items-baseline justify-between mt-1">
              <span
                className={`text-2xl font-bold font-mono ${
                  deficientCount > 0 ? "text-[#DC2626]" : "text-[var(--forest)]"
                }`}
              >
                {deficientCount}
              </span>
              <span className="text-xs font-semibold text-[var(--stone)]">
                {deficientCount === 0 ? "Foliage optimal" : "Requires attention"}
              </span>
            </div>
          </div>

          {/* Card 3: Healthy Foliage */}
          <div className="p-3.5 bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius-lg)] shadow-xs">
            <span className="text-[11px] font-bold uppercase tracking-wider text-[var(--stone)] block">
              Healthy Leaves
            </span>
            <div className="flex items-baseline justify-between mt-1">
              <span className="text-2xl font-bold font-mono text-emerald-600">
                {healthyCount}
              </span>
              <span className="text-xs text-[var(--stone)]">
                {leaves.length > 0 ? `${((healthyCount / leaves.length) * 100).toFixed(0)}% optimal` : "0%"}
              </span>
            </div>
          </div>

          {/* Card 4: Dominant Diagnosis */}
          <div className="p-3.5 bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius-lg)] shadow-xs">
            <span className="text-[11px] font-bold uppercase tracking-wider text-[var(--stone)] block">
              Primary Condition
            </span>
            <div className="mt-1">
              <span className="text-sm font-bold text-[var(--bark)] truncate block">
                {prediction?.display_label || prediction?.predicted_class || "Inspection Complete"}
              </span>
              <span className="text-[11px] text-[var(--stone)] font-mono">
                {prediction?.crop ? `${prediction.crop.toUpperCase()} · No score averaging` : ""}
              </span>
            </div>
          </div>
        </div>

        {/* Non-Averaging Agronomic Notice */}
        <div className="p-3 bg-[#F0F5FA] border border-[var(--info)]/30 rounded-[var(--radius-md)] text-xs text-[var(--info)] flex items-start gap-2">
          <svg className="w-4 h-4 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <span className="leading-relaxed">
            <strong>Rule 16 / Overview §19 Non-Averaging Guarantee:</strong> Foliar pathologies occur locally on specific leaves. Confidence scores and classifications are preserved independently for each detected leaf to ensure localized nutrient deficiencies are never diluted or obscured.
          </span>
        </div>
      </div>

      {/* 4. INDIVIDUAL LEAF DIAGNOSES GRID */}
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] pb-2.5">
          <div>
            <h3 className="text-base font-bold text-[var(--bark)] flex items-center gap-2">
              <span>Individual Leaf Diagnostics</span>
              <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded-full bg-[var(--sand)] text-[var(--bark)]">
                {leaves.length} {leaves.length === 1 ? "Leaf" : "Leaves"}
              </span>
            </h3>
            <p className="text-xs text-[var(--stone)]">
              Click any leaf card below to correlate and spotlight its bounding box on the photograph above.
            </p>
          </div>

          {selectedLeafIndex !== null && (
            <button
              type="button"
              onClick={() => setSelectedLeafIndex(null)}
              className="text-xs text-[var(--clay)] hover:underline font-bold"
            >
              Clear spotlight selection
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
              isUncertain={leaf.is_uncertain || leaf.uncertain}
              thresholds={thresholds}
              contourArea={leaf.contour_area}
              isSelected={selectedLeafIndex === idx}
              onSelect={() => setSelectedLeafIndex(selectedLeafIndex === idx ? null : idx)}
            />
          ))}
        </div>
      </div>

      {/* 5. REJECTED LEAVES PANEL (Non-Diagnostic Input Rejection) */}
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
              Quality-Filtered Leaf Masks ({rejectedLeaves.length})
            </h4>
          </div>

          <p className="text-xs text-[var(--stone)] mb-3 leading-relaxed">
            The following candidate segments were excluded by morphological quality filters (e.g. edge contact, dust, or tiny contour noise). They represent image processing rejections and are strictly NOT disease diagnoses:
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
                  <span className="text-[var(--stone)]">Status: Excluded from diagnosis</span>
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
