"use client";

import React from "react";
import Link from "next/link";
import { Badge } from "../ui/Badge";
import { UncertaintyBadge } from "../ui/UncertaintyBadge";
import { HistoryCard } from "./HistoryCard";
import { RecordRead } from "../../lib/api/types";

export interface HistoryTableProps {
  records: RecordRead[];
  className?: string;
}

export const HistoryTable: React.FC<HistoryTableProps> = ({ records, className = "" }) => {
  return (
    <div className={`space-y-4 ${className}`} data-testid="history-table-container">
      {/* 1. Mobile Stacked Cards (< 768px) */}
      <div className="md:hidden space-y-3" data-testid="history-mobile-cards">
        {records.map((rec) => (
          <HistoryCard key={rec.id} record={rec} />
        ))}
      </div>

      {/* 2. Desktop Table Container (≥ 768px) with internal horizontal scrolling */}
      {/* Acceptance Criterion 3: Table scrolls inside its own container on small screens */}
      <div className="hidden md:block rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-warm-sm)] overflow-hidden">
        <div className="w-full overflow-x-auto max-h-[70vh] relative">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-[var(--sand)]/50 border-b border-[var(--border)] font-semibold text-[var(--stone)] uppercase tracking-wider sticky top-0 z-10 backdrop-blur-sm">
              <tr>
                <th scope="col" className="px-4 py-3.5 whitespace-nowrap">
                  Observed Time &amp; TZ
                </th>
                <th scope="col" className="px-3 py-3.5 whitespace-nowrap">
                  Tags
                </th>
                <th scope="col" className="px-3 py-3.5 whitespace-nowrap">
                  Crop
                </th>
                <th scope="col" className="px-3 py-3.5 whitespace-nowrap">
                  Modality
                </th>
                <th scope="col" className="px-3 py-3.5 whitespace-nowrap">
                  Source
                </th>
                <th scope="col" className="px-4 py-3.5 whitespace-nowrap">
                  Predicted Class / Result
                </th>
                <th scope="col" className="px-4 py-3.5 whitespace-nowrap">
                  Confidence Tier
                </th>
                <th scope="col" className="px-3 py-3.5 whitespace-nowrap">
                  Model Version
                </th>
                <th scope="col" className="px-4 py-3.5 text-right whitespace-nowrap">
                  Action
                </th>
              </tr>
            </thead>

            <tbody className="divide-y divide-[var(--border)]/70 text-[var(--bark)]">
              {records.map((rec) => {
                const observedDateStr = new Date(rec.observed_at).toLocaleString(undefined, {
                  year: "numeric",
                  month: "short",
                  day: "numeric",
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                });

                const mainPrediction = Array.isArray(rec.predictions)
                  ? rec.predictions[0]
                  : rec.predictions;

                const displayResult =
                  mainPrediction?.display_label ||
                  mainPrediction?.predicted_class ||
                  "Complete";

                const confidenceScore = mainPrediction?.confidence;
                const isUncertain = mainPrediction?.is_uncertain || false;
                const modelVersion = mainPrediction?.model_version || "v1.0.0";

                return (
                  <tr
                    key={rec.id}
                    className="hover:bg-[var(--sand)]/25 transition-colors duration-150"
                    data-testid={`history-row-${rec.id}`}
                  >
                    {/* Timestamp + TZ */}
                    <td className="px-4 py-3 whitespace-nowrap">
                      <div className="font-semibold text-[var(--bark)]">{observedDateStr}</div>
                      <div className="text-[11px] font-mono text-[var(--stone)]">
                        TZ: {rec.observed_tz || "UTC"}
                      </div>
                    </td>

                    {/* Historical Tag: Mandatory on every row per DESIGN S4 */}
                    <td className="px-3 py-3 whitespace-nowrap">
                      <Badge variant="historical">Historical</Badge>
                    </td>

                    {/* Crop */}
                    <td className="px-3 py-3 whitespace-nowrap capitalize font-medium">
                      {rec.crop}
                    </td>

                    {/* Modality */}
                    <td className="px-3 py-3 whitespace-nowrap">
                      <span className="font-mono text-[11px] uppercase px-1.5 py-0.5 rounded bg-[var(--sand)] text-[var(--bark)] border border-[var(--border)]">
                        {rec.mode}
                      </span>
                    </td>

                    {/* Source Chip: 'Live' appears ONLY for records flagged live */}
                    <td className="px-3 py-3 whitespace-nowrap">
                      {rec.source === "live" ? (
                        <Badge variant="live" pulse>
                          Live
                        </Badge>
                      ) : rec.source === "manual" ? (
                        <Badge variant="manual">Manual</Badge>
                      ) : (
                        <Badge variant="imported">Imported</Badge>
                      )}
                    </td>

                    {/* Predicted Class / Result */}
                    <td className="px-4 py-3 whitespace-nowrap font-medium text-[var(--bark)]">
                      {displayResult}
                    </td>

                    {/* Confidence Tier */}
                    <td className="px-4 py-3 whitespace-nowrap">
                      {confidenceScore !== undefined ? (
                        <UncertaintyBadge
                          score={confidenceScore}
                          isUncertain={isUncertain}
                        />
                      ) : (
                        <span className="text-[var(--stone)] italic">N/A</span>
                      )}
                    </td>

                    {/* Model Version */}
                    <td className="px-3 py-3 whitespace-nowrap font-mono text-[11px] text-[var(--stone)]">
                      {modelVersion}
                    </td>

                    {/* Action */}
                    <td className="px-4 py-3 whitespace-nowrap text-right">
                      <Link
                        href={`/history/${rec.id}`}
                        className="text-[var(--forest)] hover:text-[#243B28] hover:underline font-semibold focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--moss)] rounded px-1"
                        aria-label={`View record details for inspection on ${observedDateStr}`}
                      >
                        View Details &rarr;
                      </Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
