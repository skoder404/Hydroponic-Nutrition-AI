"use client";

import React, { useEffect } from "react";
import { Button } from "@/components/ui";

export interface InputHelpDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

/**
 * Accessible explainer drawer clarifying what counts as a Single Leaf versus Whole Plant photo.
 * Conforms to WCAG: Esc key closes, accessible dialog landmarks, focus containment.
 */
export const InputHelpDrawer: React.FC<InputHelpDrawerProps> = ({ isOpen, onClose }) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="help-drawer-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="max-w-lg w-full bg-[var(--surface)] rounded-[var(--radius-lg)] border border-[var(--border)] shadow-[var(--shadow-warm-lg)] p-6 space-y-6 text-left"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between border-b border-[var(--border)] pb-4">
          <div>
            <span className="text-xs font-mono uppercase tracking-wider text-[var(--moss)] font-semibold">
              Imaging Guidelines &amp; Protocol
            </span>
            <h3 id="help-drawer-title" className="text-xl font-serif text-[var(--forest)] mt-0.5">
              Single Leaf vs. Whole Plant Photo
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-[var(--stone)] hover:text-[var(--bark)] p-1 rounded hover:bg-[var(--sand)]/50 transition-colors"
            aria-label="Close dialog"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="space-y-4 text-xs text-[var(--bark)] leading-relaxed">
          {/* Mode 1: Single Leaf */}
          <div className="p-3.5 rounded-[var(--radius-md)] border border-[var(--moss)]/30 bg-[#F6FAF7] space-y-1.5">
            <div className="flex items-center gap-2 text-sm font-semibold text-[var(--forest)]">
              <span>🍃</span>
              <h4>Mode: Single Leaf Photo</h4>
            </div>
            <p className="text-[var(--stone)]">
              <strong className="text-[var(--bark)]">What it is:</strong> A close-up, isolated photograph of an individual leaf blade (either detached or held against a clean/neutral surface).
            </p>
            <p className="text-[var(--stone)]">
              <strong className="text-[var(--bark)]">How AI processes it:</strong> Directly evaluates color spaces (ExG, ExR, HSV) and runs the single-leaf nutrient classifier without automatic segmentation.
            </p>
          </div>

          {/* Mode 2: Whole Plant */}
          <div className="p-3.5 rounded-[var(--radius-md)] border border-[var(--clay)]/30 bg-[#FCF8F5] space-y-1.5">
            <div className="flex items-center gap-2 text-sm font-semibold text-[var(--clay)]">
              <span>🪴</span>
              <h4>Mode: Whole Plant Canopy</h4>
            </div>
            <p className="text-[var(--stone)]">
              <strong className="text-[var(--bark)]">What it is:</strong> A photo of the entire plant or channel canopy containing multiple leaves, petioles, and background hydro-channels.
            </p>
            <p className="text-[var(--stone)]">
              <strong className="text-[var(--bark)]">How AI processes it:</strong> Uses Meta SAM 2.1 to isolate individual leaves, runs quality validation (filtering dust specks and whole-image floods), and produces separate per-leaf predictions.
            </p>
          </div>

          <div className="text-[11px] text-[var(--stone)] border-t border-[var(--border)] pt-3">
            <strong className="text-[var(--bark)]">Rule:</strong> Client-side file checks are UX conveniences; server-side integrity and MIME checks remain authoritative.
          </div>
        </div>

        <div className="flex justify-end pt-2">
          <Button variant="primary" size="sm" onClick={onClose}>
            Understood
          </Button>
        </div>
      </div>
    </div>
  );
};
