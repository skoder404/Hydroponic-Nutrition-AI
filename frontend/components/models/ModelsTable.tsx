"use client";

import React from "react";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { Card } from "../ui/Card";
import { ModelSummary } from "../../lib/api/types";

export interface ModelsTableProps {
  models: ModelSummary[];
  onSelectModel: (model: ModelSummary) => void;
  className?: string;
}

export const ModelsTable: React.FC<ModelsTableProps> = ({
  models,
  onSelectModel,
  className = "",
}) => {
  // Check if there are promoted models
  const promotedModels = models.filter((m) => m.status === "promoted");
  const hasPromotedModels = promotedModels.length > 0;

  // Acceptance Criterion 3: Empty state explains no promoted models
  if (!hasPromotedModels && models.length === 0) {
    return (
      <Card
        className="p-8 text-center bg-[var(--surface)] border border-[var(--border)] shadow-[var(--shadow-warm-sm)]"
        data-testid="no-promoted-models-empty-state"
      >
        <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-[#FFF4DC] text-[var(--warning)] flex items-center justify-center">
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

        <h3 className="text-base font-bold text-[var(--bark)] mb-1">
          No promoted models
        </h3>

        <p className="text-xs text-[var(--stone)] max-w-md mx-auto leading-relaxed">
          No active promoted models are currently registered in the system. Real-time inference,
          sensor evaluation, and foliar diagnosis remain unavailable until champion models are trained
          and promoted in the registry.
        </p>
      </Card>
    );
  }

  return (
    <div className={`space-y-4 ${className}`} data-testid="models-table-container">
      {/* Required Section 18 / S6 Statement */}
      <div className="p-3 bg-[var(--sand)]/40 border border-[var(--border)] rounded-[var(--radius-md)] text-xs text-[var(--stone)] flex items-center justify-between">
        <span>
          <strong className="text-[var(--bark)]">Benchmark Standard:</strong> All evaluation
          scores mean agreement with project rules/labels on a documented split.
        </span>
        <span className="font-mono text-[11px] text-[var(--bark)] font-semibold shrink-0 ml-2">
          {models.length} Model{models.length === 1 ? "" : "s"}
        </span>
      </div>

      {/* Responsive Table Container: scrolls inside its own container */}
      <div className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-warm-sm)] overflow-hidden">
        <div className="w-full overflow-x-auto max-h-[70vh] relative">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-[var(--sand)]/50 border-b border-[var(--border)] font-semibold text-[var(--stone)] uppercase tracking-wider sticky top-0 z-10 backdrop-blur-sm">
              <tr>
                <th scope="col" className="px-4 py-3.5 whitespace-nowrap">
                  Crop
                </th>
                <th scope="col" className="px-3 py-3.5 whitespace-nowrap">
                  Modality
                </th>
                <th scope="col" className="px-4 py-3.5 whitespace-nowrap">
                  Architecture / Family
                </th>
                <th scope="col" className="px-3 py-3.5 whitespace-nowrap">
                  Version
                </th>
                <th scope="col" className="px-3 py-3.5 whitespace-nowrap">
                  Status
                </th>
                <th scope="col" className="px-4 py-3.5 whitespace-nowrap">
                  Val Macro-F1
                </th>
                <th scope="col" className="px-3 py-3.5 whitespace-nowrap">
                  Type / Flag
                </th>
                <th scope="col" className="px-4 py-3.5 text-right whitespace-nowrap">
                  Action
                </th>
              </tr>
            </thead>

            <tbody className="divide-y divide-[var(--border)]/70 text-[var(--bark)]">
              {models.map((m, idx) => {
                const macroF1 =
                  typeof m.validation_macro_f1 === "number"
                    ? m.validation_macro_f1.toFixed(4)
                    : typeof (m.metrics as Record<string, unknown> | undefined)?.macro_f1 === "number"
                    ? ((m.metrics as Record<string, unknown>).macro_f1 as number).toFixed(4)
                    : "N/A";

                return (
                  <tr
                    key={`${m.crop}-${m.modality}-${m.version}-${idx}`}
                    className="hover:bg-[var(--sand)]/25 transition-colors duration-150"
                  >
                    {/* Crop */}
                    <td className="px-4 py-3 whitespace-nowrap capitalize font-bold text-[var(--bark)]">
                      {m.crop}
                    </td>

                    {/* Modality */}
                    <td className="px-3 py-3 whitespace-nowrap">
                      <span className="font-mono text-[11px] uppercase px-1.5 py-0.5 rounded bg-[var(--sand)] text-[var(--bark)] border border-[var(--border)]">
                        {m.modality}
                      </span>
                    </td>

                    {/* Architecture Family */}
                    <td className="px-4 py-3 whitespace-nowrap font-medium text-[var(--bark)]">
                      {m.family}
                    </td>

                    {/* Version */}
                    <td className="px-3 py-3 whitespace-nowrap font-mono text-[11px] text-[var(--stone)]">
                      {m.version}
                    </td>

                    {/* Status */}
                    <td className="px-3 py-3 whitespace-nowrap">
                      <Badge
                        variant={
                          m.status === "promoted"
                            ? "live"
                            : m.status === "MODEL_UNAVAILABLE"
                            ? "stale"
                            : "neutral"
                        }
                      >
                        {m.status}
                      </Badge>
                    </td>

                    {/* Validation Macro-F1 */}
                    <td className="px-4 py-3 whitespace-nowrap font-mono font-bold text-[var(--forest)]">
                      {macroF1}
                    </td>

                    {/* Acceptance Criterion 1: Experimental models are labelled */}
                    <td className="px-3 py-3 whitespace-nowrap">
                      {m.experimental ? (
                        <Badge variant="experimental">Experimental</Badge>
                      ) : (
                        <span className="text-[11px] text-[var(--stone)] font-mono">Standard</span>
                      )}
                    </td>

                    {/* Action */}
                    <td className="px-4 py-3 whitespace-nowrap text-right">
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => onSelectModel(m)}
                        aria-label={`View detailed metrics for ${m.crop} ${m.family}`}
                      >
                        View Metrics
                      </Button>
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

