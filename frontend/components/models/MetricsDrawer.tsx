"use client";

import React, { useEffect } from "react";
import { Badge } from "../ui/Badge";
import { Button } from "../ui/Button";
import { ModelSummary } from "../../lib/api/types";

export interface MetricsDrawerProps {
  model: ModelSummary | null;
  isOpen: boolean;
  onClose: () => void;
}

export const MetricsDrawer: React.FC<MetricsDrawerProps> = ({
  model,
  isOpen,
  onClose,
}) => {
  // Handle ESC key press
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !model) return null;

  const metrics = (model.metrics || {}) as Record<string, unknown>;
  const perClass = (metrics.per_class || {}) as Record<
    string,
    { precision?: number; recall?: number; f1?: number; support?: number }
  >;
  const classOrder = (model.classes ||
    (metrics.class_order as string[]) ||
    Object.keys(perClass)) as string[];
  const confusionMatrix = (metrics.confusion_matrix_list ||
    metrics.confusion_matrix) as number[][] | undefined;

  const macroF1 =
    typeof model.validation_macro_f1 === "number"
      ? model.validation_macro_f1
      : typeof metrics.macro_f1 === "number"
      ? metrics.macro_f1
      : null;

  return (
    <div
      className="fixed inset-0 z-50 overflow-hidden bg-black/40 backdrop-blur-xs flex justify-end transition-opacity duration-200"
      aria-labelledby="metrics-drawer-title"
      role="dialog"
      aria-modal="true"
      data-testid="metrics-drawer"
    >
      {/* Click outside to close backdrop */}
      <div className="absolute inset-0" onClick={onClose} aria-hidden="true" />

      {/* Drawer Panel */}
      <div className="relative w-full max-w-2xl bg-[var(--surface)] border-l border-[var(--border)] shadow-2xl h-full flex flex-col z-10 overflow-hidden text-left">
        {/* Drawer Header */}
        <div className="p-5 border-b border-[var(--border)] bg-[var(--sand)]/30 flex items-start justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2 mb-1.5">
              <span className="capitalize font-mono text-xs px-2 py-0.5 rounded bg-[var(--sand)] text-[var(--bark)] border border-[var(--border)]">
                {model.crop}
              </span>
              <span className="uppercase font-mono text-xs px-2 py-0.5 rounded bg-[var(--sand)] text-[var(--bark)] border border-[var(--border)]">
                {model.modality}
              </span>
              {model.experimental && (
                <Badge variant="experimental">Experimental</Badge>
              )}
              <Badge variant={model.status === "promoted" ? "live" : "neutral"}>
                {model.status}
              </Badge>
            </div>

            <h2
              id="metrics-drawer-title"
              className="text-lg font-bold text-[var(--bark)]"
            >
              {model.family} ({model.version})
            </h2>
          </div>

          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            aria-label="Close metrics drawer"
            className="text-[var(--stone)] hover:text-[var(--bark)]"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </Button>
        </div>

        {/* Drawer Body (Scrollable) */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 text-xs text-[var(--bark)]">
          {/* Section 18.2 / Overview §28 Mandatory Disclosure Statement */}
          <div className="p-3.5 bg-[#FFFDF7] border-l-4 border-[var(--clay)] rounded-[var(--radius-md)] text-xs text-[var(--clay)] font-medium leading-relaxed shadow-xs">
            <span className="font-bold uppercase tracking-wider block text-[11px] mb-0.5">
              Disclosure Notice
            </span>
            Scores mean agreement with project rules/labels on a documented split.
            They are empirical benchmark evaluations and do not certify veterinary or biological causation.
          </div>

          {/* Core Metrics Overview */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3 rounded-[var(--radius-md)] bg-[var(--sand)]/40 border border-[var(--border)]">
              <span className="text-[11px] font-semibold text-[var(--stone)] block uppercase">
                Macro-F1
              </span>
              <span className="text-base font-bold text-[var(--forest)] font-mono">
                {macroF1 !== null ? macroF1.toFixed(4) : "N/A"}
              </span>
            </div>

            <div className="p-3 rounded-[var(--radius-md)] bg-[var(--sand)]/40 border border-[var(--border)]">
              <span className="text-[11px] font-semibold text-[var(--stone)] block uppercase">
                Accuracy
              </span>
              <span className="text-base font-bold text-[var(--bark)] font-mono">
                {typeof metrics.accuracy === "number"
                  ? `${(metrics.accuracy * 100).toFixed(1)}%`
                  : "N/A"}
              </span>
            </div>

            <div className="p-3 rounded-[var(--radius-md)] bg-[var(--sand)]/40 border border-[var(--border)]">
              <span className="text-[11px] font-semibold text-[var(--stone)] block uppercase">
                Classes
              </span>
              <span className="text-base font-bold text-[var(--bark)] font-mono">
                {classOrder.length}
              </span>
            </div>

            <div className="p-3 rounded-[var(--radius-md)] bg-[var(--sand)]/40 border border-[var(--border)]">
              <span className="text-[11px] font-semibold text-[var(--stone)] block uppercase">
                Eval Samples
              </span>
              <span className="text-base font-bold text-[var(--bark)] font-mono">
                {metrics.total_samples !== undefined
                  ? String(metrics.total_samples)
                  : "Holdout"}
              </span>
            </div>
          </div>

          {/* Split Protocol */}
          <div className="space-y-1.5 p-3.5 bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius-md)]">
            <h3 className="font-bold text-xs uppercase tracking-wider text-[var(--stone)]">
              Split Protocol
            </h3>
            <p className="text-xs text-[var(--bark)] font-mono">
              {model.split_protocol ||
                "Stratified group holdout split; test partition preserved untouched."}
            </p>
          </div>

          {/* Per-Class Performance Breakdown & Support */}
          {Object.keys(perClass).length > 0 && (
            <div>
              <h3 className="font-bold text-xs uppercase tracking-wider text-[var(--stone)] mb-2">
                Per-Class Performance &amp; Support
              </h3>
              <div className="border border-[var(--border)] rounded-[var(--radius-md)] overflow-hidden">
                <table className="w-full text-left border-collapse text-xs">
                  <thead className="bg-[var(--sand)]/40 border-b border-[var(--border)] text-[var(--stone)] font-semibold">
                    <tr>
                      <th className="px-3 py-2">Class Label</th>
                      <th className="px-2 py-2 text-right">Precision</th>
                      <th className="px-2 py-2 text-right">Recall</th>
                      <th className="px-2 py-2 text-right">F1-Score</th>
                      <th className="px-3 py-2 text-right">Support</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--border)]/60">
                    {classOrder.map((cls) => {
                      const data = perClass[cls] || {};
                      return (
                        <tr key={cls} className="hover:bg-[var(--sand)]/20">
                          <td className="px-3 py-1.5 font-mono text-[11px] text-[var(--bark)]">
                            {cls}
                          </td>
                          <td className="px-2 py-1.5 text-right font-mono">
                            {data.precision !== undefined
                              ? data.precision.toFixed(3)
                              : "-"}
                          </td>
                          <td className="px-2 py-1.5 text-right font-mono">
                            {data.recall !== undefined ? data.recall.toFixed(3) : "-"}
                          </td>
                          <td className="px-2 py-1.5 text-right font-mono font-semibold text-[var(--forest)]">
                            {data.f1 !== undefined ? data.f1.toFixed(3) : "-"}
                          </td>
                          <td className="px-3 py-1.5 text-right font-mono text-[var(--stone)]">
                            {data.support !== undefined ? data.support : "-"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Confusion Matrix Table */}
          {confusionMatrix && confusionMatrix.length > 0 && (
            <div>
              <h3 className="font-bold text-xs uppercase tracking-wider text-[var(--stone)] mb-1">
                Confusion Matrix
              </h3>
              <p className="text-[11px] text-[var(--stone)] mb-2">
                Rows indicate ground truth; Columns indicate predicted class.
              </p>
              <div className="overflow-x-auto border border-[var(--border)] rounded-[var(--radius-md)] p-2 bg-[var(--surface)]">
                <table className="text-[10px] font-mono border-collapse">
                  <thead>
                    <tr>
                      <th className="p-1 text-left text-[var(--stone)]">True \ Pred</th>
                      {classOrder.map((lbl, idx) => (
                        <th
                          key={idx}
                          title={lbl}
                          className="p-1 text-center text-[var(--stone)] max-w-16 truncate"
                        >
                          {idx + 1}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {confusionMatrix.map((row, rIdx) => (
                      <tr key={rIdx} className="hover:bg-[var(--sand)]/30">
                        <td
                          title={classOrder[rIdx]}
                          className="p-1 font-semibold text-[var(--bark)] whitespace-nowrap max-w-28 truncate"
                        >
                          {rIdx + 1}. {classOrder[rIdx]}
                        </td>
                        {row.map((val, cIdx) => (
                          <td
                            key={cIdx}
                            className={`p-1 text-center ${
                              rIdx === cIdx
                                ? "bg-[#EBF7EE] text-[var(--forest)] font-bold"
                                : val > 0
                                ? "bg-[#FFF4DC] text-[var(--warning)]"
                                : "text-[var(--stone)]/40"
                            }`}
                          >
                            {val}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Ordered Features */}
          {model.ordered_features && model.ordered_features.length > 0 && (
            <div>
              <h3 className="font-bold text-xs uppercase tracking-wider text-[var(--stone)] mb-2">
                Input Features ({model.ordered_features.length})
              </h3>
              <div className="flex flex-wrap gap-1.5">
                {model.ordered_features.map((feat, idx) => (
                  <span
                    key={idx}
                    className="font-mono text-[11px] px-2 py-0.5 rounded bg-[var(--sand)] text-[var(--bark)] border border-[var(--border)]"
                  >
                    {feat}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Drawer Footer */}
        <div className="p-4 border-t border-[var(--border)] bg-[var(--sand)]/20 flex justify-end">
          <Button variant="secondary" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>
      </div>
    </div>
  );
};
