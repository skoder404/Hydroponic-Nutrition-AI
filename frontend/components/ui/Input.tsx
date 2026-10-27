import React from "react";

export interface InputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "size"> {
  label?: string;
  unit?: string;
  error?: string;
  missing?: boolean;
  helperText?: string;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  (
    {
      label,
      unit,
      error,
      missing = false,
      helperText,
      disabled = false,
      id,
      className = "",
      type = "text",
      ...props
    },
    ref
  ) => {
    const inputId = id || (label ? label.toLowerCase().replace(/\s+/g, "-") : undefined);

    let borderStateStyles =
      "border-[var(--border-strong)] focus:border-[var(--moss)] focus:ring-2 focus:ring-[var(--moss)] focus:ring-offset-1 focus:ring-offset-[var(--bg)]";

    if (error) {
      borderStateStyles =
        "border-[var(--error)] text-[var(--error)] focus:border-[var(--error)] focus:ring-2 focus:ring-[var(--error)]";
    } else if (missing) {
      borderStateStyles =
        "border-[var(--warning)] bg-[#FFFDF7] focus:border-[var(--warning)] focus:ring-2 focus:ring-[var(--warning)]";
    }

    return (
      <div className="w-full flex flex-col gap-1.5 text-left">
        {label && (
          <div className="flex items-center justify-between">
            <label
              htmlFor={inputId}
              className="text-xs font-semibold text-[var(--bark)] uppercase tracking-wide flex items-center gap-1.5"
            >
              {label}
              {missing && (
                <span className="text-[10px] font-medium text-[var(--warning)] bg-[#FFF4DC] px-1.5 py-0.5 rounded-[var(--radius-sm)] border border-[var(--warning)]/30">
                  Required / Missing
                </span>
              )}
            </label>
            {unit && (
              <span className="text-xs font-medium text-[var(--stone)] font-mono">
                [{unit}]
              </span>
            )}
          </div>
        )}

        <div className="relative flex items-center">
          <input
            ref={ref}
            id={inputId}
            type={type}
            disabled={disabled}
            aria-invalid={Boolean(error)}
            aria-describedby={error ? `${inputId}-error` : helperText ? `${inputId}-help` : undefined}
            className={`w-full rounded-[var(--radius-md)] bg-[var(--surface)] px-3.5 py-2 text-sm text-[var(--bark)] shadow-[var(--shadow-warm-sm)] outline-none transition-all duration-150 border disabled:opacity-50 disabled:bg-[var(--sand)]/40 disabled:cursor-not-allowed placeholder-[var(--stone)]/60 ${unit ? "pr-14" : ""} ${borderStateStyles} ${className}`}
            {...props}
          />

          {unit && (
            <div className="absolute right-3 pointer-events-none text-xs font-mono font-medium text-[var(--stone)]">
              {unit}
            </div>
          )}
        </div>

        {error && (
          <p
            id={`${inputId}-error`}
            role="alert"
            className="text-xs text-[var(--error)] font-medium flex items-center gap-1 mt-0.5"
          >
            <svg
              className="h-3.5 w-3.5 shrink-0"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              aria-hidden="true"
            >
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
            {error}
          </p>
        )}

        {!error && helperText && (
          <p id={`${inputId}-help`} className="text-xs text-[var(--stone)] mt-0.5">
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = "Input";
