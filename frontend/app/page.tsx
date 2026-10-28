"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { HeroPlant } from "@/components/home/HeroPlant";
import { CropCard, CropInfo } from "@/components/home/CropCard";
import { MotionFadeUp } from "@/components/motion";

const CROPS: CropInfo[] = [
  {
    id: "lettuce",
    name: "Lettuce",
    latin: "Lactuca sativa",
    icon: "🥬",
    description: "Macronutrient and micronutrient foliar deficiency classification and pH/EC root zone tracking.",
    classesCount: 9,
    supportedInputs: ["Leaf photos", "Sensors (pH, EC, Temp)"],
    experimental: false,
  },
  {
    id: "spinach",
    name: "Spinach",
    latin: "Spinacia oleracea",
    icon: "🍃",
    description: "USDA Ag Data Commons multi-leaf segmentation, color space textural analysis, and environmental telemetry.",
    classesCount: 12,
    supportedInputs: ["Multi-leaf photos", "Sensors (pH, EC, RH, Temp)"],
    experimental: false,
  },
  {
    id: "tomato",
    name: "Tomato",
    latin: "Solanum lycopersicum",
    icon: "🍅",
    description: "Vegetative stage leaf symptom classification and greenhouse sensor monitoring with experimental validation bounds.",
    classesCount: 15,
    supportedInputs: ["Leaf photos", "Live sensors"],
    experimental: true,
  },
];

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(true);
  const [backendError, setBackendError] = useState(false);

  useEffect(() => {
    // S1 Loading state: Skeletons for up to 600ms, then render
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 600);

    // Probe backend health
    fetch("/api/v1/health", { cache: "no-store" })
      .then((res) => {
        if (!res.ok) setBackendError(true);
      })
      .catch(() => {
        // Handled non-blockingly; layout banner also displays
        setBackendError(true);
      });

    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="space-y-12 max-w-5xl mx-auto py-2">
      {/* S1 Hero Section: Growing plant line-art and clear non-alarmist branding */}
      <section className="text-center space-y-4 pt-2 sm:pt-4">
        <HeroPlant className="w-36 h-36 sm:w-44 sm:h-44 mx-auto mb-2" />

        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[var(--sand)] text-[var(--forest)] text-xs font-semibold tracking-wide uppercase">
          Advisory Decision Support
        </div>

        <h1 className="text-3xl sm:text-5xl font-extrabold font-serif text-[var(--forest)] tracking-tight">
          Hydroponic Nutrition AI
        </h1>

        <p className="text-[var(--bark)] text-base sm:text-lg max-w-2xl mx-auto leading-relaxed">
          Evidence-backed foliar deficiency classification and environmental sensor anomaly detection. Designed for greenhouse operators and research cultivators.
        </p>

        {/* Advisory Expectations Banner */}
        <div className="max-w-2xl mx-auto p-4 rounded-xl bg-[var(--sand)]/70 border border-[var(--border)] text-xs sm:text-sm text-[var(--bark)] text-left flex items-start gap-3">
          <span className="text-lg" aria-hidden="true">💡</span>
          <p>
            <strong className="font-semibold text-[var(--forest)]">Advisory Guidance:</strong> Model predictions reflect probabilistic pattern matches from validated datasets. Predictions are never clinical diagnoses and are paired with actionable physical checks.
          </p>
        </div>

        {backendError && (
          <div className="max-w-2xl mx-auto p-3 rounded-lg bg-[#FFF8E7] border border-[var(--warning)] text-xs text-[var(--bark)] flex items-center justify-between">
            <span>Backend is currently unreachable. Predictions may be temporarily offline.</span>
            <button
              onClick={() => window.location.reload()}
              className="font-semibold text-[var(--forest)] underline hover:text-[var(--moss)]"
            >
              Retry
            </button>
          </div>
        )}
      </section>

      {/* S1 Crop Selection Grid following the 3 / 2+1 / 1 column rule */}
      <section className="space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2 border-b border-[var(--border)] pb-3">
          <div>
            <h2 className="text-2xl font-bold font-serif text-[var(--forest)]">Select Crop Cultivar</h2>
            <p className="text-xs text-[var(--stone)]">Choose a supported crop to start computer vision or telemetry analysis</p>
          </div>
          <div className="flex gap-3 text-xs font-semibold">
            <Link href="/models" className="text-[var(--forest)] hover:text-[var(--moss)] underline">
              View Model Metrics &amp; Limits →
            </Link>
          </div>
        </div>

        {/* 
          Layout rule from DESIGN.md S1:
          ≥1024: 3 columns; 640–1023: 2+1; <640: 1 column
        */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 [&>*:last-child]:sm:col-span-2 [&>*:last-child]:lg:col-span-1 gap-6">
          {CROPS.map((crop, idx) => (
            <CropCard
              key={crop.id}
              crop={crop}
              index={idx}
              isLoading={isLoading}
            />
          ))}
        </div>
      </section>

      {/* Bottom Shortcuts / Features */}
      <section className="pt-4 border-t border-[var(--border)]">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <MotionFadeUp index={3}>
            <Link
              href="/history"
              className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--sand)] hover:border-[var(--moss)] transition-colors flex items-center justify-between group block"
            >
              <div>
                <h4 className="font-serif font-bold text-[var(--forest)] text-sm group-hover:text-[var(--moss)]">Historical Records</h4>
                <p className="text-xs text-[var(--stone)]">Review past nutrient observations and recommendations</p>
              </div>
              <span className="text-[var(--forest)] group-hover:translate-x-1 transition-transform" aria-hidden="true">→</span>
            </Link>
          </MotionFadeUp>

          <MotionFadeUp index={4}>
            <Link
              href="/about"
              className="p-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--sand)] hover:border-[var(--moss)] transition-colors flex items-center justify-between group block"
            >
              <div>
                <h4 className="font-serif font-bold text-[var(--forest)] text-sm group-hover:text-[var(--moss)]">Methodology &amp; Architecture</h4>
                <p className="text-xs text-[var(--stone)]">Learn about our two-stage vision models and rule fallback matrix</p>
              </div>
              <span className="text-[var(--forest)] group-hover:translate-x-1 transition-transform" aria-hidden="true">→</span>
            </Link>
          </MotionFadeUp>
        </div>
      </section>
    </div>
  );
}
