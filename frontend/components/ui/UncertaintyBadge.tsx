"use client";

import React, { useState, useId } from "react";
import { ConfidenceTier, ConfidenceThresholds, getConfidenceTier } from "./ConfidenceMeter";

export interface UncertaintyBadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  /** Confidence score in [0, 1] */
  score?: number;
  /** Tier thresholds supplied by API/registry (never hardcoded constants) */
  thresholds?: ConfidenceThresholds;
  /** Explicit uncertainty flag */
  isUncertain?: boolean;
  /** Explicit tier override if already derived by parent */
  tier?: ConfidenceTier;
  /** Custom class name */
  className?: string;
}

export const UncertaintyBadge: React.FC<UncertaintyBadgeProps> = ({
  score,
  thresholds,
  isUncertain = false,
  tier: explicitTier,
  className = "",
  ...props
}) => {
  const [showTooltip, setShowTooltip] = useState(false);
  const tooltipId = useId();

  // Resolve tier: either explicit, or derived from score + API thresholds
  const resolvedTier: ConfidenceTier = explicitTier
    ? explicitTier
    : score !== undefined && thresholds !== undefined
    ? getConfidenceTier(score, thresholds, isUncertain)
    : isUncertain
    ? "uncertain"
    : "moderate";

  // Tier design tokens: accessible icon + text + pattern/border; never colour alone
  // Uncertain tier is visually lighter than higher-confidence tier
  const tierConfig: Record<
    ConfidenceTier,
    {
      label: string;
      bgStyle: string;
      textStyle: string;
      borderStyle: string;
      icon: React.ReactNode;
    }
  > = {
    higher: {
      label: "Higher Confidence",
      bgStyle: "bg-[#EBF7EE] shadow-xs",
      textStyle: "text-[#2E6B3A] font-semibold",
      borderStyle: "border-[#2E6B3A] border-solid",
      icon: (
        <svg
          className="w-3.5 h-3.5 text-[#2E6B3A] shrink-0"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2.5}
            d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
          />
        </svg>
      ),
    },
    moderate: {
      label: "Moderate Confidence",
      bgStyle: "bg-[#FFF9EC]",
      textStyle: "text-[#8A5A00] font-medium",
      borderStyle: "border-[#8A5A00]/70 border-dashed",
      icon: (
        <svg
          className="w-3.5 h-3.5 text-[#8A5A00] shrink-0"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <circle cx="12" cy="12" r="9" strokeWidth={2} />
          <path strokeLinecap="round" strokeWidth={2} d="M12 8v4m0 4h.01" />
        </svg>
      ),
    },
    uncertain: {
      label: "Uncertain Score",
      // Visually lighter than higher-confidence tier (subdued sand tint, reduced opacity, dotted line)
      bgStyle: "bg-[var(--sand)]/40 opacity-80",
      textStyle: "text-[var(--stone)] font-normal",
      borderStyle: "border-[var(--border-strong)] border-dotted",
      icon: (
        <svg
          className="w-3.5 h-3.5 text-[var(--clay)] shrink-0"
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
      ),
    },
  };

  const current = tierConfig[resolvedTier];

  return (
    <span className="relative inline-flex items-center" {...props}>
      <button
        type="button"
        onClick={() => setShowTooltip((prev) => !prev)}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        onFocus={() => setShowTooltip(true)}
        onBlur={() => setShowTooltip(false)}
        aria-describedby={tooltipId}
        aria-expanded={showTooltip}
        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-[var(--radius-full)] text-xs border select-none transition-all duration-150 cursor-help focus:outline-none focus:ring-2 focus:ring-[var(--moss)] ${current.bgStyle} ${current.textStyle} ${current.borderStyle} ${className}`}
      >
        {current.icon}
        <span>{current.label}</span>
        {score !== undefined && (
          <span className="font-mono text-[11px] opacity-80">
            ({Math.round(score * 100)}%)
          </span>
        )}
      </button>

      {/* Accessible tooltip with mandatory wording */}
      {showTooltip && (
        <span
          id={tooltipId}
          role="tooltip"
          className="absolute left-1/2 -translate-x-1/2 bottom-full mb-1.5 z-40 w-52 p-2 rounded-md bg-[var(--bark)] text-[#FAF6EC] text-[11px] leading-tight shadow-lg border border-[var(--border-strong)] pointer-events-none text-center"
        >
          <span className="block font-medium text-[#FAF6EC]">
            Model score, not a calibrated probability.
          </span>
          <span className="block text-[10px] text-[#D9CDB1] mt-0.5">
            Tap or hover to inspect uncertainty tier guidelines.
          </span>
        </span>
      )}
    </span>
  );
};
