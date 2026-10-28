"use client";

import React from "react";
import Link from "next/link";
import { Badge } from "../ui/Badge";
import { Card } from "../ui/Card";
import { UncertaintyBadge } from "../ui/UncertaintyBadge";
import { RecordRead } from "../../lib/api/types";

export interface HistoryCardProps {
  record: RecordRead;
  className?: string;
}

export const HistoryCard: React.FC<HistoryCardProps> = ({ record, className = "" }) => {
  // Format observed date and timezone
  const observedDateStr = new Date(record.observed_at).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  // Extract main prediction information if available
  const mainPrediction = Array.isArray(record.predictions)
    ? record.predictions[0]
    : record.predictions;

  const displayResult =
    mainPrediction?.display_label ||
    mainPrediction?.predicted_class ||
    "Inspected";

  const confidenceScore = mainPrediction?.confidence;
  const isUncertain = mainPrediction?.is_uncertain || false;
  const modelVersion = mainPrediction?.model_version || "v1.0.0";

  return (
    <Card
      variant="default"
      interactive
      className={`p-4 border border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-warm-sm)] hover:border-[var(--forest)]/50 transition-colors ${className}`}
      data-testid="history-card"
    >
      <div className="flex items-start justify-between gap-2 mb-2.5">
        {/* Timestamp + Timezone */}
        <div>
          <span className="text-xs font-bold text-[var(--bark)] block">
            {observedDateStr}
          </span>
          <span className="text-[11px] font-mono text-[var(--stone)] block">
            TZ: {record.observed_tz || "UTC"}
          </span>
        </div>

        {/* Source Chip: 'Live' appears ONLY for records flagged live */}
        <div className="flex items-center gap-1.5 shrink-0">
          {record.source === "live" ? (
            <Badge variant="live" pulse>
              Live
            </Badge>
          ) : record.source === "manual" ? (
            <Badge variant="manual">Manual</Badge>
          ) : (
            <Badge variant="imported">Imported</Badge>
          )}

          {/* Historical Tag: Mandatory on every record per DESIGN S4 */}
          <Badge variant="historical">Historical</Badge>
        </div>
      </div>

      {/* Crop + Modality */}
      <div className="flex items-center gap-2 mb-3">
        <span className="capitalize font-semibold text-sm text-[var(--bark)]">
          {record.crop}
        </span>
        <span className="text-xs px-2 py-0.5 rounded bg-[var(--sand)] text-[var(--bark)] border border-[var(--border)] uppercase tracking-wider font-mono">
          {record.mode}
        </span>
      </div>

      {/* Prediction Result & Confidence */}
      <div className="p-2.5 bg-[var(--sand)]/30 rounded-[var(--radius-md)] border border-[var(--border)]/70 mb-3 flex items-center justify-between">
        <div>
          <span className="text-[11px] uppercase tracking-wider text-[var(--stone)] block font-medium">
            Result
          </span>
          <span className="text-xs font-bold text-[var(--bark)]">
            {displayResult}
          </span>
        </div>

        {confidenceScore !== undefined ? (
          <UncertaintyBadge
            score={confidenceScore}
            isUncertain={isUncertain}
          />
        ) : (
          <span className="text-xs text-[var(--stone)]">Unrated</span>
        )}
      </div>

      {/* Model Version + Action Link */}
      <div className="flex items-center justify-between pt-2 border-t border-[var(--border)]/60 text-xs">
        <span className="text-[var(--stone)] font-mono text-[11px]">
          Model: {modelVersion}
        </span>

        <Link
          href={`/history/${record.id}`}
          className="font-semibold text-[var(--forest)] hover:underline inline-flex items-center gap-1 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--moss)] rounded"
        >
          <span>View Details</span>
          <span aria-hidden="true">&rarr;</span>
        </Link>
      </div>
    </Card>
  );
};

