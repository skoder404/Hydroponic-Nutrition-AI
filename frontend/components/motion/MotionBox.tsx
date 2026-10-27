"use client";

import React, { useEffect, useState } from "react";
import { useMotion } from "./MotionProvider";
import { fadeUpVariant, growVariant, getStaggerDelay } from "./variants";

export interface MotionFadeUpProps extends React.HTMLAttributes<HTMLDivElement> {
  index?: number;
  staggerMs?: number;
}

/**
 * Container component that applies the DESIGN.md fadeUp animation (250-300ms, 8px slide).
 * When reduced motion is active, translation is eliminated and only an opacity fade or instant display occurs.
 */
export const MotionFadeUp: React.FC<MotionFadeUpProps> = ({
  children,
  index = 0,
  staggerMs,
  className = "",
  style,
  ...props
}) => {
  const { prefersReducedMotion } = useMotion();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const delay = prefersReducedMotion ? 0 : getStaggerDelay(index, staggerMs);
    const timer = setTimeout(() => setMounted(true), delay);
    return () => clearTimeout(timer);
  }, [index, prefersReducedMotion, staggerMs]);

  const variant = prefersReducedMotion ? fadeUpVariant.reducedMotion : fadeUpVariant;
  const currentStyles = mounted ? variant.animate : variant.initial;

  return (
    <div
      className={className}
      style={{
        ...style,
        opacity: currentStyles.opacity,
        transform: currentStyles.transform,
        transition: currentStyles.transition,
      }}
      {...props}
    >
      {children}
    </div>
  );
};

export interface MotionGrowSvgProps extends React.SVGProps<SVGSVGElement> {
  pathD: string;
  strokeColor?: string;
  strokeWidth?: number;
}

/**
 * SVG Path component that applies the DESIGN.md hero plant grow stroke draw animation (~1200ms).
 * Degrades instantly to static stroke when reduced motion is preferred.
 */
export const MotionGrowSvg: React.FC<MotionGrowSvgProps> = ({
  pathD,
  strokeColor = "var(--forest)",
  strokeWidth = 2,
  className = "",
  ...props
}) => {
  const { prefersReducedMotion } = useMotion();
  const [isDrawn, setIsDrawn] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setIsDrawn(true), 50);
    return () => clearTimeout(timer);
  }, []);

  const variant = prefersReducedMotion ? growVariant.reducedMotion : growVariant;
  const current = isDrawn ? variant.animate : variant.initial;

  return (
    <svg viewBox="0 0 100 100" className={`overflow-visible ${className}`} {...props}>
      <path
        d={pathD}
        fill="none"
        stroke={strokeColor}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{
          strokeDasharray: "200",
          strokeDashoffset: isDrawn || prefersReducedMotion ? "0" : "200",
          opacity: current.opacity,
          transition: prefersReducedMotion ? "none" : current.transition,
        }}
      />
    </svg>
  );
};
