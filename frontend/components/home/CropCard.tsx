"use client";

import React from "react";
import Link from "next/link";
import { MotionFadeUp } from "@/components/motion";

export interface CropInfo {
  id: string;
  name: string;
  latin: string;
  icon: string;
  description: string;
  classesCount: number;
  supportedInputs: string[];
  experimental?: boolean;
}

export interface CropCardProps {
  crop: CropInfo;
  index?: number;
  isLoading?: boolean;
}

export function CropCard({ crop, index = 0, isLoading = false }: CropCardProps) {
  if (isLoading) {
    return (
      <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow-warm-sm)] animate-pulse space-y-4">
        <div className="h-10 w-10 bg-[var(--sand)] rounded-lg" />
        <div className="space-y-2">
          <div className="h-6 w-32 bg-[var(--sand)] rounded" />
          <div className="h-4 w-24 bg-[var(--sand)] rounded" />
        </div>
        <div className="h-12 bg-[var(--sand)] rounded" />
        <div className="pt-4 border-t border-[var(--border)] flex justify-between">
          <div className="h-5 w-20 bg-[var(--sand)] rounded" />
          <div className="h-5 w-16 bg-[var(--sand)] rounded" />
        </div>
      </div>
    );
  }

  return (
    <MotionFadeUp index={index}>
      <Link
        href={`/analyze?crop=${crop.id}`}
        aria-label={`Select ${crop.name} for nutrition analysis`}
        className="group relative flex flex-col justify-between h-full rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow-warm-sm)] hover:shadow-[var(--shadow-warm-md)] hover:border-[var(--moss)] hover:-translate-y-[3px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring-color)] focus-visible:ring-offset-2 transition-all duration-200 cursor-pointer text-left"
      >
        <div>
          {/* Header icon with hover tilt (DESIGN.md: 6 degree tilt on hover) */}
          <div className="flex items-start justify-between gap-2 mb-3">
            <span
              className="text-4xl block transform transition-transform duration-200 group-hover:rotate-6 group-hover:scale-105"
              aria-hidden="true"
            >
              {crop.icon}
            </span>
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-[var(--sand)] text-[var(--forest)]">
              {crop.classesCount} Classes
            </span>
          </div>

          <h3 className="text-xl font-bold font-serif text-[var(--forest)] group-hover:text-[var(--moss)] transition-colors">
            {crop.name}
          </h3>
          <p className="text-xs italic text-[var(--stone)] mb-2.5">{crop.latin}</p>
          <p className="text-sm text-[var(--bark)] leading-relaxed mb-4">
            {crop.description}
          </p>

          {/* Supported Inputs Chips */}
          <div className="flex flex-wrap gap-1.5 mb-2">
            {crop.supportedInputs.map((input) => (
              <span
                key={input}
                className="text-xs px-2 py-0.5 rounded-md bg-[var(--sand)] text-[var(--stone)] font-medium"
              >
                {input}
              </span>
            ))}
            {crop.experimental && (
              <span className="text-xs px-2 py-0.5 rounded-md bg-[#FFF3E0] border border-[#FFB74D] text-[#B78103] font-semibold">
                Sensor mode: experimental
              </span>
            )}
          </div>
        </div>

        {/* Action footer */}
        <div className="mt-5 pt-3 border-t border-[var(--border)] flex items-center justify-between text-xs font-semibold text-[var(--forest)] group-hover:text-[var(--moss)] transition-colors">
          <span>Start Analysis</span>
          <span className="transform transition-transform group-hover:translate-x-1" aria-hidden="true">
            →
          </span>
        </div>
      </Link>
    </MotionFadeUp>
  );
}
