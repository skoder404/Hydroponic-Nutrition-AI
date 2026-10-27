import React from "react";

export type BadgeVariant =
  | "live"
  | "manual"
  | "upload"
  | "imported"
  | "mock"
  | "gemini"
  | "rules"
  | "experimental"
  | "historical"
  | "stale"
  | "neutral";

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  pulse?: boolean;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = "neutral",
  pulse = false,
  className = "",
  ...props
}) => {
  const baseStyles =
    "inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-[var(--radius-full)] text-xs font-medium border select-none transition-colors duration-150";

  const variantStyles: Record<BadgeVariant, string> = {
    live: "bg-[#EBF7EE] text-[var(--success)] border-[var(--success)]/30",
    manual: "bg-[var(--sand)] text-[var(--bark)] border-[var(--border)]",
    upload: "bg-[#EDF6F8] text-[var(--water)] border-[var(--water)]/30",
    imported: "bg-[#F3EFF8] text-[#5C3C8A] border-[#5C3C8A]/30",
    mock: "bg-[var(--sand)]/60 text-[var(--stone)] border-dashed border-[var(--border-strong)]",
    gemini: "bg-[#F0F5FA] text-[var(--info)] border-[var(--info)]/30",
    rules: "bg-[#F7F4EC] text-[var(--clay)] border-[var(--clay)]/30",
    experimental: "bg-[#FFF4DC] text-[var(--warning)] border-[var(--warning)]/40 font-semibold",
    historical: "bg-[var(--sand)]/50 text-[var(--stone)] border-[var(--border)]",
    stale: "bg-[#F5F2EC] text-[var(--stone)] border-[var(--border-strong)] opacity-80",
    neutral: "bg-[var(--sand)] text-[var(--bark)] border-[var(--border)]",
  };

  return (
    <span className={`${baseStyles} ${variantStyles[variant]} ${className}`} {...props}>
      {pulse && (
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-current" />
        </span>
      )}
      {children}
    </span>
  );
};
