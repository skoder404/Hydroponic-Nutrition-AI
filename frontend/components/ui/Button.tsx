import React from "react";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost" | "destructive";
  size?: "sm" | "md" | "lg";
  loading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      children,
      variant = "primary",
      size = "md",
      loading = false,
      disabled = false,
      className = "",
      ...props
    },
    ref
  ) => {
    const baseStyles =
      "inline-flex items-center justify-center font-medium transition-all duration-150 select-none " +
      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--moss)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--bg)] " +
      "active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none disabled:active:scale-100 cursor-pointer disabled:cursor-not-allowed";

    const sizeStyles = {
      sm: "text-xs px-3 py-1.5 rounded-[var(--radius-md)] gap-1.5",
      md: "text-sm px-4 py-2 rounded-[var(--radius-md)] gap-2",
      lg: "text-base px-6 py-3 rounded-[var(--radius-md)] gap-2.5",
    }[size];

    const variantStyles = {
      primary:
        "bg-[var(--forest)] text-[#FFFFFF] shadow-[var(--shadow-warm-sm)] hover:bg-[#243B28] hover:shadow-[var(--shadow-warm-md)] active:bg-[#1D3020]",
      secondary:
        "bg-transparent border border-[var(--clay)] text-[var(--clay)] hover:bg-[var(--clay)] hover:text-[#FFFFFF] active:bg-[#8B4224]",
      ghost:
        "bg-transparent text-[var(--stone)] hover:bg-[var(--sand)] hover:text-[var(--bark)] active:bg-[#E3D8BF]",
      destructive:
        "bg-[var(--error)] text-[#FFFFFF] shadow-[var(--shadow-warm-sm)] hover:bg-[#85221B] active:bg-[#6D1B16]",
    }[variant];

    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        aria-busy={loading}
        className={`${baseStyles} ${sizeStyles} ${variantStyles} ${className}`}
        {...props}
      >
        {loading && (
          <svg
            className="animate-spin -ml-0.5 h-4 w-4 text-current"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        )}
        <span>{children}</span>
      </button>
    );
  }
);

Button.displayName = "Button";
