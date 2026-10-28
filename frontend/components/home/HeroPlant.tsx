"use client";

import React from "react";
import { MotionGrowSvg } from "@/components/motion";

/**
 * Animated line-art plant hero that draws itself once via SVG stroke-dashoffset animation.
 * Conforms to DESIGN.md S1: ~1.2s grow animation, static drawing under reduced motion.
 */
export function HeroPlant({ className = "w-36 h-36 sm:w-44 sm:h-44 mx-auto" }: { className?: string }) {
  // SVG path for seedling stem and organic leaves
  const plantPaths = [
    // Stem
    "M 50 95 Q 50 65 50 35 Q 50 20 48 10",
    // Left leaf
    "M 50 65 C 35 60 20 45 25 30 C 35 30 45 45 50 50",
    // Right leaf
    "M 50 50 C 65 45 80 30 75 18 C 65 18 55 35 50 35",
    // Top budding leaf
    "M 50 25 C 42 18 45 8 50 5 C 55 8 58 18 50 25",
    // Base soil mound / reservoir line
    "M 25 95 Q 50 90 75 95",
  ];

  return (
    <div className={`relative flex items-center justify-center ${className}`} aria-label="Decorative growing seedling illustration">
      <svg
        viewBox="0 0 100 100"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full drop-shadow-sm"
      >
        {plantPaths.map((d, i) => (
          <MotionGrowSvg
            key={i}
            pathD={d}
            strokeColor="var(--forest)"
            strokeWidth={2.2}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ))}
        {/* Subtle decorative droplet */}
        <circle cx="50" cy="94" r="2" fill="var(--water)" opacity="0.8" />
      </svg>
    </div>
  );
}
