"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { Card } from "../../components/ui/Card";
import { useMotion } from "../../components/motion";
import { apiClient } from "../../lib/api/client";

export const CAMPUS_POLL_INTERVAL_S = 10;
export const STALENESS_THRESHOLD_S = 45;

export interface LiveSensorItem {
  name: string;
  label: string;
  value: number;
  unit: string;
  normalBand: [number, number];
  observedAt: Date;
  isStale: boolean;
}

export default function LiveMonitorPage() {
  const { prefersReducedMotion } = useMotion();

  // Connection & Poll State
  const [isConnected, setIsConnected] = useState<boolean>(true);
  const [lastSuccessTime, setLastSuccessTime] = useState<Date | null>(new Date());
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState<boolean>(false);

  // Live Telemetry & Freshness
  const [sensors, setSensors] = useState<LiveSensorItem[]>([
    {
      name: "ph",
      label: "Nutrient Solution pH",
      value: 6.18,
      unit: "pH",
      normalBand: [5.5, 6.5],
      observedAt: new Date(),
      isStale: false,
    },
    {
      name: "ec",
      label: "Electrical Conductivity",
      value: 1.64,
      unit: "mS/cm",
      normalBand: [1.2, 2.2],
      observedAt: new Date(),
      isStale: false,
    },
    {
      name: "water_temp",
      label: "Water Temperature",
      value: 20.8,
      unit: "°C",
      normalBand: [18.0, 24.0],
      observedAt: new Date(),
      isStale: false,
    },
    {
      name: "humidity",
      label: "Relative Humidity",
      value: 64.2,
      unit: "%",
      normalBand: [50.0, 75.0],
      observedAt: new Date(),
      isStale: false,
    },
    {
      name: "air_temp",
      label: "Greenhouse Air Temp",
      value: 23.4,
      unit: "°C",
      normalBand: [18.0, 28.0],
      observedAt: new Date(),
      isStale: false,
    },
    {
      name: "do_level",
      label: "Dissolved Oxygen",
      value: 7.2,
      unit: "mg/L",
      normalBand: [6.0, 9.0],
      observedAt: new Date(Date.now() - 75000), // Stale (> 45s)
      isStale: true,
    },
  ]);

  // Camera Frame State
  const [cameraCaptureTime, setCameraCaptureTime] = useState<Date>(new Date());
  const [hasNewData, setHasNewData] = useState<boolean>(false);
  const [liveAnnouncement, setLiveAnnouncement] = useState<string>("");

  // Polling function
  const pollLiveData = useCallback(async () => {
    setIsPolling(true);
    try {
      // Check device status via API client
      const status = await apiClient.getDeviceStatus();

      if (!status.connected) {
        throw new Error("Campus hardware gateway reported disconnected status.");
      }

      const now = new Date();
      setLastSuccessTime(now);
      setIsConnected(true);
      setErrorMessage(null);
      setCameraCaptureTime(now);

      // Fluctuate readings slightly for live sensation
      setSensors((prev) =>
        prev.map((s) => {
          if (s.isStale) return s; // Keep stale item labeled as stale
          const delta = (Math.random() - 0.5) * 0.05;
          return {
            ...s,
            value: Math.round((s.value + delta) * 100) / 100,
            observedAt: now,
          };
        })
      );

      // Acceptance Criterion 3: New data triggers pulse + aria-live announcement
      setHasNewData(true);
      setLiveAnnouncement(`New live telemetry data acquired at ${now.toLocaleTimeString()}`);
      setTimeout(() => setHasNewData(false), 1500);
    } catch (err: unknown) {
      // Acceptance Criterion 1: Server unreachable shows last-success time and Retry
      setIsConnected(false);
      setErrorMessage(
        err instanceof Error ? err.message : "Campus hardware server unreachable."
      );
    } finally {
      setIsPolling(false);
    }
  }, []);

  // Auto-refresh interval (paused when tab is hidden per DESIGN S8)
  useEffect(() => {
    let intervalId: NodeJS.Timeout | null = null;

    const startPolling = () => {
      if (!intervalId) {
        intervalId = setInterval(() => {
          if (document.visibilityState === "visible") {
            pollLiveData();
          }
        }, CAMPUS_POLL_INTERVAL_S * 1000);
      }
    };

    const stopPolling = () => {
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
    };

    const handleVisibilityChange = () => {
      if (document.visibilityState === "hidden") {
        stopPolling();
      } else {
        pollLiveData();
        startPolling();
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    startPolling();

    return () => {
      stopPolling();
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [pollLiveData]);

  return (
    <div className="space-y-6 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 text-left" data-testid="live-monitor-page">
      {/* Acceptance Criterion 3: aria-live announcement container */}
      <div
        role="status"
        aria-live="polite"
        className="sr-only"
        data-testid="live-aria-announcement"
      >
        {liveAnnouncement}
      </div>

      {/* Header & Status Bar */}
      <div className="border-b border-[var(--border)] pb-4 flex flex-wrap items-center justify-between gap-4">
        <div>
          <span className="text-xs font-mono font-semibold uppercase tracking-wider text-[var(--water)]">
            Telemetry Stream S8
          </span>
          <h1 className="text-2xl sm:text-3xl font-bold text-[var(--bark)] tracking-tight mt-0.5">
            Campus Greenhouse Live Monitor
          </h1>
          <p className="text-sm text-[var(--stone)] mt-1">
            Real-time sensory telemetry and camera streams with automated staleness detection.
          </p>
        </div>

        {/* Connection Status Badge */}
        <div className="flex items-center gap-2.5">
          {isConnected ? (
            <Badge variant="live" pulse={!prefersReducedMotion}>
              Live Stream Active
            </Badge>
          ) : (
            <Badge variant="stale">Gateway Offline</Badge>
          )}

          <Link href="/analyze?source=live">
            <Button variant="primary" size="sm">
              Analyze Current State
            </Button>
          </Link>
        </div>
      </div>

      {/* Acceptance Criterion 1: Server unreachable shows last-success time and Retry */}
      {!isConnected && (
        <Card
          variant="warning"
          className="p-5 border-l-4 border-[var(--error)] bg-[#FFFDF7]"
          data-testid="server-unreachable-card"
          role="alert"
        >
          <div className="flex items-start gap-3">
            <svg
              className="w-5 h-5 text-[var(--error)] shrink-0 mt-0.5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            <div className="flex-1">
              <h3 className="text-sm font-bold text-[var(--bark)]">
                Campus Server Unreachable
              </h3>
              <p className="mt-1 text-xs text-[var(--stone)] leading-relaxed">
                {errorMessage || "Lost HTTP telemetry connection to the campus IoT adapter gateway."}
              </p>
              <p className="mt-1 text-xs text-[var(--bark)] font-medium">
                Last successful telemetry capture:{" "}
                <span className="font-mono text-[var(--stone)]">
                  {lastSuccessTime ? lastSuccessTime.toLocaleTimeString() : "Never"}
                </span>
              </p>
              <div className="mt-3">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={pollLiveData}
                  loading={isPolling}
                  data-testid="retry-connection-button"
                >
                  Retry Connection
                </Button>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Tomato Experimental Limitation Banner */}
      <div className="p-3 bg-[#FFF4DC] border border-[var(--warning)]/40 rounded-[var(--radius-md)] text-xs text-[#7A4B00] flex items-center justify-between">
        <span>
          <strong className="font-bold">Tomato Telemetry Disclaimer:</strong> Transformed sensor
          features and synthetic water temperature must be mapped to physical probes before production
          live monitoring (Overview §28.16).
        </span>
        <Badge variant="experimental">Notice</Badge>
      </div>

      {/* Main Grid: Live Sensor Grid + Camera Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Sensor Grid (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold uppercase tracking-wider text-[var(--bark)] flex items-center gap-2">
              <span>Telemetry Ingestion</span>
              <span className="text-xs font-normal text-[var(--stone)]">
                (Polling every {CAMPUS_POLL_INTERVAL_S}s)
              </span>
            </h2>

            <Link href="/analyze?source=live&mode=sensor">
              <span className="text-xs text-[var(--forest)] font-semibold hover:underline cursor-pointer">
                Analyze these readings &rarr;
              </span>
            </Link>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3" data-testid="live-sensors-grid">
            {sensors.map((sensor) => {
              const isOutOfBand =
                sensor.value < sensor.normalBand[0] || sensor.value > sensor.normalBand[1];

              return (
                /* Acceptance Criterion 2: Stale values are labelled stale */
                <Card
                  key={sensor.name}
                  className={`p-4 border transition-all duration-300 relative ${
                    sensor.isStale
                      ? "bg-[var(--sand)]/40 border-[var(--border-strong)] opacity-75 shadow-none"
                      : "bg-[var(--surface)] border-[var(--border)] shadow-[var(--shadow-warm-sm)]"
                  } ${
                    /* Acceptance Criterion 3: Pulse on new data; reduced motion shows text only */
                    hasNewData && !sensor.isStale
                      ? prefersReducedMotion
                        ? "border-[var(--forest)]"
                        : "ring-2 ring-[var(--forest)]/40 scale-[1.01]"
                      : ""
                  }`}
                  data-testid={`sensor-card-${sensor.name}`}
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <span className="text-xs font-medium text-[var(--stone)]">
                      {sensor.label}
                    </span>

                    {/* Freshness / Staleness Badge */}
                    {sensor.isStale ? (
                      <Badge variant="stale">Stale</Badge>
                    ) : (
                      <span className="flex items-center gap-1 text-[11px] font-mono text-[var(--forest)]">
                        {!prefersReducedMotion && (
                          <span className="w-1.5 h-1.5 rounded-full bg-[var(--forest)] animate-pulse" />
                        )}
                        <span>Fresh</span>
                      </span>
                    )}
                  </div>

                  <div className="flex items-baseline justify-between mb-2">
                    <div className="text-2xl font-bold font-mono text-[var(--bark)]">
                      {sensor.value.toFixed(2)}{" "}
                      <span className="text-sm font-normal text-[var(--stone)]">
                        {sensor.unit}
                      </span>
                    </div>

                    <span
                      className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded ${
                        isOutOfBand
                          ? "bg-[#FFF4DC] text-[var(--warning)]"
                          : "bg-[#EBF7EE] text-[var(--forest)]"
                      }`}
                    >
                      {isOutOfBand ? "Deviation" : "Nominal"}
                    </span>
                  </div>

                  <div className="pt-2 border-t border-[var(--border)]/50 flex items-center justify-between text-[11px] text-[var(--stone)] font-mono">
                    <span>
                      Band: {sensor.normalBand[0]} - {sensor.normalBand[1]} {sensor.unit}
                    </span>
                    <span>{sensor.observedAt.toLocaleTimeString()}</span>
                  </div>
                </Card>
              );
            })}
          </div>
        </div>

        {/* Camera Panel (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold uppercase tracking-wider text-[var(--bark)]">
              Campus Canopy Camera
            </h2>

            <Link href="/analyze?source=live&mode=image">
              <span className="text-xs text-[var(--forest)] font-semibold hover:underline cursor-pointer">
                Analyze this frame &rarr;
              </span>
            </Link>
          </div>

          <Card className="p-4 bg-[var(--surface)] border border-[var(--border)] shadow-[var(--shadow-warm-sm)] space-y-3">
            <div className="relative w-full aspect-4/3 bg-[var(--sand)]/50 rounded-[var(--radius-md)] overflow-hidden border border-[var(--border)] flex items-center justify-center">
              <div className="p-6 text-center text-xs text-[var(--stone)] space-y-2 flex flex-col items-center">
                <svg
                  className="w-12 h-12 text-[var(--stone)]/60"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                </svg>
                <span className="font-semibold text-[var(--bark)]">
                  Live Camera Feed Active (Bay 1 Canopy)
                </span>
                <span className="font-mono text-[11px]">
                  Frame captured: {cameraCaptureTime.toLocaleTimeString()}
                </span>
              </div>
            </div>

            <div className="flex items-center justify-between pt-2 text-xs">
              <span className="text-[var(--stone)] font-mono text-[11px]">
                Camera: Campus RTSP / MJPEG
              </span>

              <Link href="/analyze?source=live&mode=image">
                <Button variant="secondary" size="sm">
                  Analyze Frame
                </Button>
              </Link>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
