"use client";

import React, { useState, useId } from "react";

export type ConfidenceTier = "higher" | "moderate" | "uncertain";

export interface ConfidenceThresholds {
  /** Score at or above which the tier is classified as 'higher' (e.g. 0.75 from model metadata) */
  high: number;
  /** Score at or above which the tier is classified as 'moderate' (e.g. 0.50 from model metadata) */
  moderate: number;
}

export interface ConfidenceMeterProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Normalized confidence score in [0, 1] */
  score: number;
  /** Tier thresholds supplied by API or model registry (never hardcoded constants) */
  thresholds: ConfidenceThresholds;
  /** Optional explicit override if model uncertainty rejection flag was triggered */
  isUncertain?: boolean;
  /** Label describing the prediction target, e.g. 'Leaf #2' or 'Foliar condition' */
  label?: string;
  /** Whether to show compact meter layout */
  compact?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * Determines confidence tier from API-provided thresholds.
 */
export function getConfidenceTier(
  score: number,
  thresholds: ConfidenceThresholds,
  isUncertain?: boolean
): ConfidenceTier {
  if (isUncertain) return "uncertain";
  if (score >= thresholds.high) return "higher";
  if (score >= thresholds.moderate) return "moderate";
  return "uncertain";
}

export const ConfidenceMeter: React.FC<ConfidenceMeterProps> = ({
  score,
  thresholds,
  isUncertain = false,
  label,
  compact = false,
  className = "",
  ...props
}) => {
  const [showTooltip, setShowTooltip] = useState(false);
  const tooltipId = useId();

  // Clamp score between 0 and 1
  const clampedScore = Math.max(0, Math.min(1, Number.isFinite(score) ? score : 0));
  const percentage = Math.round(clampedScore * 100);

  const tier = getConfidenceTier(clampedScore, thresholds, isUncertain);

  // Accessible tier configuration: distinct icon, text, and CSS pattern (never color alone)
  const tierConfig: Record<
    ConfidenceTier,
    {
      name: string;
      subtext: string;
      colorClass: string;
      bgClass: string;
      borderClass: string;
      barPattern: string;
      barBg: string;
      icon: React.ReactNode;
    }
  > = {
    higher: {
      name: "Higher Confidence",
      subtext: `Score ≥ ${(thresholds.high * 100).toFixed(0)}%`,
      colorClass: "text-[var(--forest)]",
      bgClass: "bg-[#EBF7EE] shadow-sm",
      borderClass: "border-[#2E6B3A] border-solid",
      barPattern: "bg-[#2E6B3A]",
      barBg: "bg-[#D5ECD9]",
      icon: (
        <svg
          className="w-4 h-4 text-[#2E6B3A] shrink-0"
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
      name: "Moderate Confidence",
      subtext: `Score ${(thresholds.moderate * 100).toFixed(0)}%–${(thresholds.high * 100).toFixed(0)}%`,
      colorClass: "text-[#8A5A00]",
      bgClass: "bg-[#FFF9EC]",
      borderClass: "border-[#8A5A00]/60 border-dashed",
      barPattern:
        "bg-[repeating-linear-gradient(45deg,#8A5A00,#8A5A00_6px,#B27900_6px,#B27900_12px)]",
      barBg: "bg-[#F7EED8]",
      icon: (
        <svg
          className="w-4 h-4 text-[#8A5A00] shrink-0"
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
      name: "Uncertain Score",
      subtext: `Score < ${(thresholds.moderate * 100).toFixed(0)}%`,
      // Uncertain tier is visually lighter than higher-confidence tier
      colorClass: "text-[var(--stone)]",
      bgClass: "bg-[var(--sand)]/40 opacity-85",
      borderClass: "border-[var(--border-strong)] border-dotted",
      barPattern:
        "bg-[repeating-linear-gradient(-45deg,#665C4D,#665C4D_4px,transparent_4px,transparent_8px)]",
      barBg: "bg-[var(--sand)]/60",
      icon: (
        <svg
          className="w-4 h-4 text-[var(--clay)] shrink-0"
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

  const current = tierConfig[tier];

  return (
    <div
      className={`rounded-lg p-3 border transition-all duration-200 ${current.bgClass} ${current.borderClass} ${className}`}
      role="region"
      aria-label={label ? `Confidence meter for ${label}` : "Model confidence meter"}
      {...props}
    >
      {/* Header with Title, Tier Badge, and Touch/Click Tooltip Toggle */}
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-1.5 min-w-0">
          {current.icon}
          <span className={`text-xs font-semibold tracking-tight ${current.colorClass}`}>
            {current.name}
          </span>
          <span className="text-[11px] text-[var(--stone)] hidden sm:inline" aria-hidden="true">
            • {percentage}%
          </span>
        </div>

        {/* Info/Disclaimer trigger (works on touch and hover) */}
        <div className="relative inline-block">
          <button
            type="button"
            onClick={() => setShowTooltip((prev) => !prev)}
            onMouseEnter={() => setShowTooltip(true)}
            onMouseLeave={() => setShowTooltip(false)}
            onFocus={() => setShowTooltip(true)}
            onBlur={() => setShowTooltip(false)}
            aria-describedby={tooltipId}
            aria-expanded={showTooltip}
            className="p-1 rounded-full text-[var(--stone)] hover:text-[var(--bark)] hover:bg-[var(--sand)] focus:outline-none focus:ring-2 focus:ring-[var(--moss)] transition-colors"
            title="Model score disclaimer"
          >
            <span className="sr-only">Disclaimer information</span>
            <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                clipRule="evenodd"
              />
            </svg>
          </button>

          {/* Accessible Tooltip: Wording strictly says 'model score, not a calibrated probability' */}
          {showTooltip && (
            <div
              id={tooltipId}
              role="tooltip"
              className="absolute right-0 bottom-full mb-1 z-30 w-56 p-2 rounded-md bg-[var(--bark)] text-[#FAF6EC] text-[11px] leading-tight shadow-lg border border-[var(--border-strong)] pointer-events-none"
            >
              <p className="font-medium text-[#FAF6EC]">
                Model score, not a calibrated probability.
              </p>
              <p className="text-[10px] text-[#D9CDB1] mt-1">
                Reflects classifier output distance/softmax heuristic. Thresholds calibrated on holdout validation data.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Visual meter bar with distinct hatch/fill patterns */}
      <div className="w-full">
        <div
          className={`h-2.5 w-full rounded-full overflow-hidden ${current.barBg} border border-black/5`}
          role="progressbar"
          aria-valuenow={percentage}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuetext={`${percentage}% (${current.name})`}
        >
          <div
            className={`h-full transition-all duration-500 ease-out ${current.barPattern}`}
            style={{ width: `${percentage}%` }}
          />
        </div>

        {/* Legend row showing numerical score and threshold context */}
        {!compact && (
          <div className="flex items-center justify-between text-[11px] text-[var(--stone)] mt-1.5 font-mono">
            <span>Score: {clampedScore.toFixed(3)}</span>
            <span>Thresholds: ≥{(thresholds.high * 100).toFixed(0)}% / ≥{(thresholds.moderate * 100).toFixed(0)}%</span>
          </div>
        )}
      </div>
    </div>
  );
};
