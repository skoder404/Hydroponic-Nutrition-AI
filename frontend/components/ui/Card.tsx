import React from "react";

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "result" | "warning" | "muted-uncertain";
  interactive?: boolean;
  skeleton?: boolean;
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  (
    {
      children,
      variant = "default",
      interactive = false,
      skeleton = false,
      className = "",
      ...props
    },
    ref
  ) => {
    if (skeleton) {
      return (
        <div
          ref={ref}
          aria-hidden="true"
          className={`rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow-warm-sm)] animate-pulse ${className}`}
          {...props}
        >
          <div className="h-5 w-1/3 bg-[var(--sand)] rounded-[var(--radius-sm)] mb-4" />
          <div className="space-y-2.5">
            <div className="h-3.5 w-full bg-[var(--sand)]/70 rounded-[var(--radius-sm)]" />
            <div className="h-3.5 w-4/5 bg-[var(--sand)]/50 rounded-[var(--radius-sm)]" />
          </div>
        </div>
      );
    }

    const baseStyles =
      "rounded-[var(--radius-lg)] border p-6 transition-all duration-200 text-[var(--bark)] text-left";

    const interactiveStyles = interactive
      ? "cursor-pointer hover:-translate-y-1 hover:shadow-[var(--shadow-warm-md)] active:translate-y-0"
      : "shadow-[var(--shadow-warm-sm)]";

    const variantStyles = {
      default: "bg-[var(--surface)] border-[var(--border)]",
      result: "bg-[var(--surface)] border-2 border-[var(--moss)] shadow-[var(--shadow-warm-md)]",
      warning: "bg-[#FFFDF7] border-[var(--warning)] border-l-4",
      "muted-uncertain":
        "bg-[var(--sand)]/40 border-[var(--border)] opacity-85 hover:opacity-100",
    }[variant];

    return (
      <div
        ref={ref}
        className={`${baseStyles} ${interactiveStyles} ${variantStyles} ${className}`}
        {...props}
      >
        {children}
      </div>
    );
  }
);

Card.displayName = "Card";
