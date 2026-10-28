"use client";

import React from "react";
import { Button } from "../ui/Button";

export type AnalysisStep = "validating" | "segmenting" | "classifying" | "recommending" | "complete";

export interface VineProgressIndicatorProps {
  currentStep: AnalysisStep;
  onCancel?: () => void;
  className?: string;
}

const STEPS: Array<{ key: AnalysisStep; label: string; description: string }> = [
  { key: "validating", label: "Validating", description: "Verifying input schema and bounds" },
  { key: "segmenting", label: "Segmenting", description: "Isolating individual leaf contours" },
  { key: "classifying", label: "Classifying", description: "Evaluating neural & classical models" },
  { key: "recommending", label: "Recommending", description: "Synthesizing agronomic advice" },
];

export const VineProgressIndicator: React.FC<VineProgressIndicatorProps> = ({
  currentStep,
  onCancel,
  className = "",
}) => {
  const getStepStatus = (stepKey: AnalysisStep) => {
    const order: AnalysisStep[] = ["validating", "segmenting", "classifying", "recommending", "complete"];
    const currentIndex = order.indexOf(currentStep);
    const stepIndex = order.indexOf(stepKey);

    if (stepIndex < currentIndex) return "done";
    if (stepIndex === currentIndex) return "active";
    return "pending";
  };

  return (
    <div
      className={`p-6 bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius-lg)] shadow-[var(--shadow-warm-md)] text-left space-y-6 ${className}`}
      data-testid="vine-progress-indicator"
      aria-live="polite"
    >
      <div className="flex items-center justify-between border-b border-[var(--border)] pb-3">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-[var(--forest)] block">
            Pipeline In Progress
          </span>
          <h3 className="text-base font-bold text-[var(--bark)]">
            Executing Multi-Stage Plant Health Pipeline
          </h3>
        </div>

        {onCancel && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onCancel}
            aria-label="Cancel active analysis request"
            data-testid="cancel-analysis-button"
            className="text-[var(--error)] hover:bg-[#FDF3F2]"
          >
            Cancel Analysis
          </Button>
        )}
      </div>

      {/* 4-Step Growing Vine Stepper */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 relative">
        {STEPS.map((s, idx) => {
          const status = getStepStatus(s.key);
          const isDone = status === "done";
          const isActive = status === "active";

          return (
            <div key={s.key} className="space-y-2 relative">
              <div className="flex items-center gap-2.5">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-all duration-300 ${
                    isDone
                      ? "bg-[var(--forest)] text-[#FFFFFF]"
                      : isActive
                      ? "bg-[#EBF7EE] text-[var(--forest)] border-2 border-[var(--forest)] ring-4 ring-[var(--forest)]/15 animate-pulse"
                      : "bg-[var(--sand)] text-[var(--stone)] border border-[var(--border)]"
                  }`}
                >
                  {isDone ? (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <span>{idx + 1}</span>
                  )}
                </div>

                <span
                  className={`text-xs font-bold uppercase tracking-wider ${
                    isActive ? "text-[var(--forest)]" : isDone ? "text-[var(--bark)]" : "text-[var(--stone)]"
                  }`}
                >
                  {s.label}
                </span>
              </div>

              <p className="text-[11px] text-[var(--stone)] pl-10 sm:pl-0 leading-tight">
                {s.description}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
};
