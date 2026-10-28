"use client";

import React, { useState } from "react";
import { InputHelpDrawer } from "./InputHelpDrawer";

export type InputMode = "single_leaf" | "plant";

export interface InputModeToggleProps {
  value: InputMode | null;
  onChange: (mode: InputMode) => void;
  error?: string;
  disabled?: boolean;
}

export const InputModeToggle: React.FC<InputModeToggleProps> = ({
  value,
  onChange,
  error,
  disabled = false,
}) => {
  const [isHelpOpen, setIsHelpOpen] = useState(false);

  return (
    <div className="w-full space-y-2 text-left">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold text-[var(--bark)] uppercase tracking-wide flex items-center gap-1.5">
          <span>Input Mode</span>
          <span className="text-[10px] font-normal text-[var(--stone)]">(Always Explicit)</span>
        </label>
        <button
          type="button"
          onClick={() => setIsHelpOpen(true)}
          className="text-xs text-[var(--moss)] hover:underline flex items-center gap-1 font-medium cursor-pointer"
          aria-label="Open imaging guidelines"
        >
          <span className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-[var(--sand)] text-[10px] font-mono">
            ?
          </span>
          <span>Which should I choose?</span>
        </button>
      </div>

      <div
        role="radiogroup"
        aria-label="Image Input Mode"
        className="grid grid-cols-2 gap-2 p-1 rounded-[var(--radius-md)] bg-[var(--sand)]/50 border border-[var(--border)]"
      >
        <button
          type="button"
          role="radio"
          aria-checked={value === "single_leaf"}
          disabled={disabled}
          onClick={() => onChange("single_leaf")}
          className={`flex items-center justify-center gap-2 py-2.5 px-3 rounded-[var(--radius-sm)] text-xs font-medium transition-all duration-150 cursor-pointer disabled:cursor-not-allowed ${
            value === "single_leaf"
              ? "bg-[var(--surface)] text-[var(--forest)] font-semibold shadow-[var(--shadow-warm-sm)] border border-[var(--border)]"
              : "text-[var(--stone)] hover:text-[var(--bark)] hover:bg-[var(--sand)]/80"
          }`}
        >
          <span className="text-sm">🍃</span>
          <span>Single Leaf Photo</span>
        </button>

        <button
          type="button"
          role="radio"
          aria-checked={value === "plant"}
          disabled={disabled}
          onClick={() => onChange("plant")}
          className={`flex items-center justify-center gap-2 py-2.5 px-3 rounded-[var(--radius-sm)] text-xs font-medium transition-all duration-150 cursor-pointer disabled:cursor-not-allowed ${
            value === "plant"
              ? "bg-[var(--surface)] text-[var(--forest)] font-semibold shadow-[var(--shadow-warm-sm)] border border-[var(--border)]"
              : "text-[var(--stone)] hover:text-[var(--bark)] hover:bg-[var(--sand)]/80"
          }`}
        >
          <span className="text-sm">🪴</span>
          <span>Whole Plant Canopy</span>
        </button>
      </div>

      {error && (
        <p role="alert" className="text-xs text-[var(--error)] font-medium mt-1">
          {error}
        </p>
      )}

      <InputHelpDrawer isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} />
    </div>
  );
};
