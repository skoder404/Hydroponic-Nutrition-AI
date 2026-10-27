"use client";

import React, { useEffect, useState } from "react";

export type ToastVariant = "success" | "info" | "warning" | "error";

export interface ToastProps {
  id?: string;
  variant?: ToastVariant;
  title: string;
  message?: string;
  onDismiss?: () => void;
  autoDismissMs?: number;
}

export const Toast: React.FC<ToastProps> = ({
  variant = "info",
  title,
  message,
  onDismiss,
  autoDismissMs = variant === "error" ? 0 : 5000,
}) => {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    if (autoDismissMs > 0) {
      const timer = setTimeout(() => {
        setIsVisible(false);
        if (onDismiss) onDismiss();
      }, autoDismissMs);
      return () => clearTimeout(timer);
    }
  }, [autoDismissMs, onDismiss]);

  if (!isVisible) return null;

  const variantStyles: Record<ToastVariant, { border: string; bg: string; text: string; icon: string }> = {
    success: {
      border: "border-[var(--success)]",
      bg: "bg-[#F3FAF5]",
      text: "text-[var(--success)]",
      icon: "✓",
    },
    info: {
      border: "border-[var(--info)]",
      bg: "bg-[#F0F5FA]",
      text: "text-[var(--info)]",
      icon: "ℹ",
    },
    warning: {
      border: "border-[var(--warning)]",
      bg: "bg-[#FFF9EC]",
      text: "text-[var(--warning)]",
      icon: "⚠",
    },
    error: {
      border: "border-[var(--error)]",
      bg: "bg-[#FDF3F2]",
      text: "text-[var(--error)]",
      icon: "✕",
    },
  };

  const current = variantStyles[variant];

  return (
    <div
      role={variant === "error" ? "alert" : "status"}
      aria-live={variant === "error" ? "assertive" : "polite"}
      className={`max-w-md w-full rounded-[var(--radius-md)] border-l-4 ${current.border} ${current.bg} p-4 shadow-[var(--shadow-warm-md)] flex items-start gap-3 transition-all duration-200 text-left`}
    >
      <span className={`text-base font-bold ${current.text} shrink-0 mt-0.5`}>
        {current.icon}
      </span>
      <div className="flex-1 min-w-0">
        <h4 className="text-sm font-semibold text-[var(--bark)]">{title}</h4>
        {message && <p className="text-xs text-[var(--stone)] mt-0.5">{message}</p>}
      </div>
      {onDismiss && (
        <button
          type="button"
          onClick={() => {
            setIsVisible(false);
            onDismiss();
          }}
          className="text-[var(--stone)] hover:text-[var(--bark)] p-1 rounded hover:bg-[var(--sand)]/50 transition-colors"
          aria-label="Dismiss toast"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </div>
  );
};
