"use client";

import React from "react";
import { Card } from "../ui/Card";
import { ConfidenceMeter, ConfidenceThresholds } from "../ui/ConfidenceMeter";
import { UncertaintyBadge } from "../ui/UncertaintyBadge";

export interface LeafCardProps {
  /** Leaf identifier, e.g. "leaf_01" or "1" */
  leafId: string | number;
  /** Index number for numbered leaf overlay synchronization (1-based) */
  index?: number;
  /** User-friendly display label, e.g. "Healthy Leaf", "Anthracnose" */
  displayLabel: string;
  /** Original unmapped class name from model output, e.g. "Spinach_anthracnose" */
  originalClass?: string;
  /** Model prediction confidence score in [0, 1] */
  confidence: number;
  /** Whether this leaf triggered the model uncertainty rejection flag */
  isUncertain?: boolean;
  /** Thresholds for confidence tiers (supplied by API/metadata) */
  thresholds?: ConfidenceThresholds;
  /** Contour area in pixels or percentage, if provided */
  contourArea?: number;
  /** Whether this leaf card is currently selected / highlighted */
  isSelected?: boolean;
  /** Callback when card is clicked or selected via keyboard */
  onSelect?: () => void;
  /** Optional custom CSS classes */
  className?: string;
}

const DEFAULT_THRESHOLDS: ConfidenceThresholds = {
  high: 0.75,
  moderate: 0.50,
};

export const LeafCard: React.FC<LeafCardProps> = ({
  leafId,
  index,
  displayLabel,
  originalClass,
  confidence,
  isUncertain = false,
  thresholds = DEFAULT_THRESHOLDS,
  contourArea,
  isSelected = false,
  onSelect,
  className = "",
}) => {
  const displayNumber = index !== undefined ? index : leafId;

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelect?.();
    }
  };

  return (
    <Card
      role="button"
      tabIndex={0}
      aria-pressed={isSelected}
      aria-label={`Leaf ${displayNumber}: ${displayLabel}, confidence ${(confidence * 100).toFixed(0)}%`}
      variant={isUncertain ? "muted-uncertain" : "default"}
      interactive
      onClick={onSelect}
      onKeyDown={handleKeyDown}
      className={`relative p-5 transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--moss)] focus-visible:ring-offset-2 ${
        isSelected
          ? "border-2 border-[var(--forest)] ring-2 ring-[var(--forest)]/20 shadow-[var(--shadow-warm-md)] bg-[#F8FAF7]"
          : "border-[var(--border)] hover:border-[var(--border-strong)]"
      } ${className}`}
    >
      {/* Top Header: Leaf Number Badge + Uncertainty Badge */}
      <div className="flex items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold transition-colors ${
              isSelected
                ? "bg-[var(--forest)] text-[#FFFFFF]"
                : "bg-[var(--sand)] text-[var(--bark)] border border-[var(--border-strong)]"
            }`}
          >
            #{displayNumber}
          </span>
          <span className="text-xs font-semibold tracking-wider text-[var(--stone)] uppercase">
            Leaf {displayNumber}
          </span>
        </div>

        <UncertaintyBadge
          score={confidence}
          thresholds={thresholds}
          isUncertain={isUncertain}
        />
      </div>

      {/* Main Condition Display */}
      <div className="mb-4">
        <h4 className="text-base font-semibold text-[var(--bark)] leading-snug">
          {displayLabel}
        </h4>

        {originalClass && originalClass !== displayLabel && (
          <details className="mt-1 text-xs text-[var(--stone)] group">
            <summary className="cursor-pointer list-none hover:text-[var(--bark)] focus-visible:outline-none inline-flex items-center gap-1 font-mono">
              <span className="underline decoration-dotted">Raw model class:</span>
              <span className="bg-[var(--sand)]/80 px-1.5 py-0.5 rounded text-[11px]">
                {originalClass}
              </span>
            </summary>
          </details>
        )}
      </div>

      {/* Per-Leaf ConfidenceMeter */}
      <div className="pt-2 border-t border-[var(--border)]/60">
        <ConfidenceMeter
          score={confidence}
          thresholds={thresholds}
          isUncertain={isUncertain}
          compact
          label={`Leaf #${displayNumber} Confidence`}
        />
      </div>

      {/* Footer Metrics (Contour Area if available) */}
      {contourArea !== undefined && contourArea > 0 && (
        <div className="mt-3 pt-2 border-t border-dashed border-[var(--border)]/40 flex items-center justify-between text-xs text-[var(--stone)]">
          <span>Segment Area</span>
          <span className="font-mono text-[var(--bark)] font-medium">
            {contourArea.toLocaleString()} px
          </span>
        </div>
      )}
    </Card>
  );
};
