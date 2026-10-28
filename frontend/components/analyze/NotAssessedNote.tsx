"use client";

import React from "react";
import { Card } from "../ui/Card";

export interface NotAssessedNoteProps {
  missingModality: "image" | "sensor";
  crop?: string;
  className?: string;
}

export const NotAssessedNote: React.FC<NotAssessedNoteProps> = ({
  missingModality,
  crop = "crop",
  className = "",
}) => {
  const isMissingImage = missingModality === "image";

  return (
    <Card
      variant="default"
      className={`p-5 bg-[var(--surface)] border border-dashed border-[var(--border)] shadow-[var(--shadow-warm-sm)] text-left ${className}`}
      data-testid="not-assessed-note"
    >
      <div className="flex items-start gap-3">
        <div className="w-8 h-8 rounded-full bg-[var(--sand)] text-[var(--stone)] flex items-center justify-center shrink-0">
          <svg
            className="w-4 h-4 text-[var(--stone)]"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        </div>

        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-bold text-[var(--bark)]">
              Not Assessed: {isMissingImage ? "Foliar Imagery" : "Water & Telemetry Sensors"}
            </h4>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[var(--sand)] text-[var(--stone)] uppercase tracking-wider font-semibold">
              Omitted Modality
            </span>
          </div>

          <p className="text-xs text-[var(--stone)] leading-relaxed">
            {isMissingImage
              ? `No plant photograph was provided for this inspection. Analysis was performed strictly on water chemistry sensor inputs. Foliar leaf health is not assessed.`
              : `No sensor telemetry was provided for this inspection. Analysis was performed strictly on leaf imagery. Root-zone water chemistry and nutrient bands are not assessed.`}
          </p>
        </div>
      </div>
    </Card>
  );
};
