"use client";

import React from "react";
import { Badge } from "../ui/Badge";
import { Card } from "../ui/Card";
import { RecommendationAction, RecommendationResponse } from "../../lib/api/types";

export interface RecommendationCardProps {
  recommendation?: RecommendationResponse | null;
  crop?: string;
  className?: string;
}

export const RecommendationCard: React.FC<RecommendationCardProps> = ({
  recommendation,
  crop = "crop",
  className = "",
}) => {
  if (!recommendation) {
    return (
      <Card
        variant="default"
        className={`p-6 bg-[var(--surface)] border border-[var(--border)] text-left shadow-[var(--shadow-warm-sm)] ${className}`}
        data-testid="recommendation-card-empty"
      >
        <span className="text-xs uppercase tracking-wider text-[var(--stone)] font-bold block mb-1">
          Agronomic Guidance
        </span>
        <h3 className="text-base font-bold text-[var(--bark)]">
          No Advisory Recommendations Generated
        </h3>
        <p className="mt-1 text-xs text-[var(--stone)] leading-relaxed">
          Nutrient telemetry and foliar conditions are within nominal project baseline bounds.
        </p>
      </Card>
    );
  }

  // Acceptance Criterion 1: Source chip reflects source from the API
  const isGemini = recommendation.source === "gemini";
  const content = (recommendation as any).content || {};
  const summaryText =
    (typeof recommendation.summary === "string" && recommendation.summary.trim()) ||
    (typeof content.summary === "string" && content.summary.trim()) ||
    (typeof content === "string" && content.trim() ? content : "") ||
    "Nutrient telemetry and foliar conditions evaluated. No acute imbalance detected.";

  const rawActions: any[] =
    (recommendation.actions && recommendation.actions.length > 0
      ? recommendation.actions
      : content.actions) || [];

  const actions = rawActions.map((act: any, idx: number) => {
    if (typeof act === "string") {
      return {
        parameter: `Intervention #${idx + 1}`,
        action_type: "Adjustment",
        adjustment_amount: 0,
        unit: "",
        urgency: "immediate" as const,
        rationale: act,
        target_range: null,
      };
    }
    return act;
  });

  const checks: string[] = (content.checks || []).map((c: any) => (typeof c === "string" ? c : String(c)));
  const evidence: string[] =
    recommendation.evidence_citations ||
    ((recommendation as any).evidence?.visual_symptoms as string[]) ||
    [];

  const rawDate = recommendation.generated_at || (recommendation as any).created_at;
  let timeFormatted = "";
  if (rawDate) {
    const parsed = new Date(rawDate);
    if (!isNaN(parsed.getTime())) {
      timeFormatted = parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }
  }
  if (!timeFormatted) {
    timeFormatted = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  return (
    <Card
      variant="default"
      className={`p-6 bg-[var(--surface)] border border-[var(--border)] shadow-[var(--shadow-warm-sm)] text-left space-y-5 ${className}`}
      data-testid="recommendation-card"
    >
      {/* Top Header: Title + Acceptance Criterion 1 (Source chip reflects API source) */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] pb-3">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-[var(--stone)] block mb-0.5">
            Actionable Advisory
          </span>
          <h3 className="text-lg font-bold text-[var(--bark)]">
            Agronomic Recommendations &amp; Corrective Actions
          </h3>
        </div>

        <div className="flex items-center gap-2">
          {/* Source chip reflects API source */}
          <Badge variant={isGemini ? "gemini" : "rules"}>
            Source: {isGemini ? "Gemini AI" : "Agronomic Rules"}
          </Badge>
          <span className="font-mono text-xs text-[var(--stone)]">
            {timeFormatted}
          </span>
        </div>
      </div>

      {/* Summary Narrative */}
      <div className="p-4 bg-[var(--sand)]/30 rounded-[var(--radius-md)] border border-[var(--border)]/70">
        <h4 className="text-xs font-bold uppercase tracking-wider text-[var(--bark)] mb-1">
          Executive Summary
        </h4>
        <p className="text-sm text-[var(--bark)] leading-relaxed">
          {summaryText}
        </p>
      </div>

      {/* Actionable Diagnostic Checks (if provided by Gemini AI) */}
      {checks.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs font-bold uppercase tracking-wider text-[var(--stone)]">
            Targeted Field Inspection Checks ({checks.length})
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            {checks.map((chk, idx) => (
              <div
                key={idx}
                className="p-3 rounded-[var(--radius-md)] bg-[var(--surface)] border border-[var(--border)] text-xs text-[var(--bark)] flex items-start gap-2 shadow-xs"
              >
                <span className="font-mono font-bold text-[var(--forest)] shrink-0">#{idx + 1}</span>
                <span className="leading-snug">{chk}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Suggested Checks / Corrective Actions */}
      {actions.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-xs font-bold uppercase tracking-wider text-[var(--stone)]">
            Suggested Interventions &amp; Remediation ({actions.length})
          </h4>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {actions.map((act, idx) => (
              <div
                key={idx}
                className="p-3.5 rounded-[var(--radius-md)] bg-[var(--surface)] border border-[var(--border)] space-y-1.5 text-xs shadow-xs"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-bold text-[var(--bark)] capitalize">
                    {act.parameter?.replace(/_/g, " ") || `Action #${idx + 1}`}
                  </span>
                  <span
                    className={`text-[10px] uppercase font-mono px-2 py-0.5 rounded font-bold ${
                      act.urgency === "immediate"
                        ? "bg-[#FCE8E6] text-[var(--error)]"
                        : act.urgency === "scheduled"
                        ? "bg-[#FFF4DC] text-[var(--warning)]"
                        : "bg-[#EDF6F8] text-[var(--water)]"
                    }`}
                  >
                    {act.urgency || "standard"}
                  </span>
                </div>

                <div className="text-[11px] font-mono text-[var(--stone)]">
                  Action: <span className="font-semibold text-[var(--bark)]">{act.action_type}</span>{" "}
                  {act.adjustment_amount > 0 && (
                    <span>
                      by {act.adjustment_amount} {act.unit}
                    </span>
                  )}
                  {act.target_range && (
                    <span>
                      {" "}
                      (Target: {act.target_range[0]} - {act.target_range[1]} {act.unit})
                    </span>
                  )}
                </div>

                <p className="text-[11px] text-[var(--stone)] leading-relaxed pt-1 border-t border-[var(--border)]/40">
                  {act.rationale}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Evidence Citations */}
      {evidence.length > 0 && (
        <div className="pt-2 border-t border-[var(--border)]/60 text-xs">
          <span className="font-bold uppercase tracking-wider text-[var(--stone)] text-[11px] block mb-1.5">
            Agronomic Evidence &amp; Rule References
          </span>
          <ul className="list-disc list-inside space-y-1 text-[var(--bark)]/85">
            {evidence.map((ev, idx) => (
              <li key={idx} className="font-mono text-[11px]">
                {ev}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Mandatory Honest Architecture Statement: Acceptance Criterion 3: No combined score is shown anywhere */}
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
          <span>Non-Fused Advisory Guidance</span>
        </div>
        <p className="leading-relaxed">
          Recommendations are conservative and advisory. No combined score is computed across image
          and sensor modalities to prevent artificial certainty. In accordance with TECH_RULES §15,
          image and sensor assessments maintain separate evidence bases.
        </p>
      </div>
    </Card>
  );
};
