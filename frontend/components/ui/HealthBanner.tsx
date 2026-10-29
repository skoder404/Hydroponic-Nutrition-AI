"use client";

import React, { useEffect, useState, useCallback } from "react";

export interface HealthBannerProps {
  initialDown?: boolean;
  healthEndpoint?: string;
  pollIntervalMs?: number;
}

export function HealthBanner({
  initialDown = false,
  healthEndpoint = "/api/v1/health",
  pollIntervalMs = 15000,
}: HealthBannerProps) {
  const [isDown, setIsDown] = useState(initialDown);
  const [isRetrying, setIsRetrying] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  const checkHealth = useCallback(async () => {
    try {
      // Try relative endpoint first, then direct backend port 8000
      let res = await fetch(healthEndpoint, {
        method: "GET",
        headers: { Accept: "application/json" },
        cache: "no-store",
      }).catch(() => null);

      if (!res || !res.ok) {
        res = await fetch("http://127.0.0.1:8000/api/v1/health", {
          method: "GET",
          headers: { Accept: "application/json" },
          cache: "no-store",
        }).catch(() => null);
      }

      if (res && res.ok) {
        const data = await res.json();
        if (data.status === "ok") {
          setIsDown(false);
          setDismissed(false);
          return;
        }
      }
      setIsDown(true);
    } catch {
      setIsDown(true);
    }
  }, [healthEndpoint]);

  useEffect(() => {
    if (initialDown) return;
    checkHealth();
    const interval = setInterval(checkHealth, pollIntervalMs);
    return () => clearInterval(interval);
  }, [checkHealth, initialDown, pollIntervalMs]);

  const handleRetry = async () => {
    setIsRetrying(true);
    await checkHealth();
    setIsRetrying(false);
  };

  if (!isDown || dismissed) {
    return null;
  }

  return (
    <aside
      role="alert"
      aria-live="polite"
      className="w-full bg-[#FFF8E7] border-b border-[var(--warning)] px-4 py-2.5 text-xs sm:text-sm text-[var(--bark)] transition-all duration-200"
    >
      <div className="mx-auto max-w-7xl flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[var(--warning)] text-white font-bold text-xs" aria-hidden="true">
            !
          </span>
          <p className="font-medium">
            <strong className="font-semibold text-[var(--forest)]">Backend unreachable</strong> — analysis and telemetry are temporarily unavailable.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleRetry}
            disabled={isRetrying}
            className="rounded px-2.5 py-1 text-xs font-semibold bg-[var(--forest)] text-white hover:bg-[var(--moss)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring-color)] focus-visible:ring-offset-2 transition-colors disabled:opacity-60"
          >
            {isRetrying ? "Checking..." : "Retry Connection"}
          </button>
          <button
            type="button"
            onClick={() => setDismissed(true)}
            aria-label="Dismiss connectivity alert"
            className="rounded p-1 text-[var(--stone)] hover:text-[var(--bark)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring-color)] transition-colors"
          >
            ✕
          </button>
        </div>
      </div>
    </aside>
  );
}
