/**
 * Shared motion variants adhering to DESIGN.md Motion System:
 * - Reduced motion transforms movement into a simple fade or nothing.
 * - Animates ONLY transform, opacity, and SVG stroke properties.
 * - Durations derive directly from MOTION_DURATIONS tokens.
 */

import { MOTION_DURATIONS, MOTION_EASINGS } from "./tokens";

export interface MotionStyleState {
  opacity?: number;
  transform?: string;
  strokeDashoffset?: number | string;
  strokeDasharray?: number | string;
  transition?: string;
}

export interface SharedVariant {
  initial: MotionStyleState;
  animate: MotionStyleState;
  exit?: MotionStyleState;
  reducedMotion: {
    initial: MotionStyleState;
    animate: MotionStyleState;
    exit?: MotionStyleState;
  };
}

/**
 * 1. fadeUp: 250-300ms upward entrance.
 * Reduced motion fallback: opacity fade only (no translateY movement).
 */
export const fadeUpVariant: SharedVariant = {
  initial: {
    opacity: 0,
    transform: "translateY(8px)",
    transition: `opacity ${MOTION_DURATIONS.cropCardFadeUp}ms ${MOTION_EASINGS.default}, transform ${MOTION_DURATIONS.cropCardFadeUp}ms ${MOTION_EASINGS.default}`,
  },
  animate: {
    opacity: 1,
    transform: "translateY(0px)",
    transition: `opacity ${MOTION_DURATIONS.cropCardFadeUp}ms ${MOTION_EASINGS.default}, transform ${MOTION_DURATIONS.cropCardFadeUp}ms ${MOTION_EASINGS.default}`,
  },
  exit: {
    opacity: 0,
    transform: "translateY(-8px)",
    transition: `opacity ${MOTION_DURATIONS.cropCardFadeUp}ms ${MOTION_EASINGS.default}`,
  },
  reducedMotion: {
    initial: { opacity: 0, transform: "none", transition: `opacity ${MOTION_DURATIONS.cropCardFadeUp}ms ${MOTION_EASINGS.default}` },
    animate: { opacity: 1, transform: "none", transition: `opacity ${MOTION_DURATIONS.cropCardFadeUp}ms ${MOTION_EASINGS.default}` },
    exit: { opacity: 0, transform: "none", transition: `opacity ${MOTION_DURATIONS.cropCardFadeUp}ms ${MOTION_EASINGS.default}` },
  },
};

/**
 * 2. grow: SVG line art stroke draw for hero plant and leaf paths (~1200ms).
 * Reduced motion fallback: instantly static drawn path.
 */
export const growVariant: SharedVariant = {
  initial: {
    opacity: 0,
    strokeDashoffset: "100%",
    strokeDasharray: "100%",
    transition: `stroke-dashoffset ${MOTION_DURATIONS.heroPlantGrow}ms ${MOTION_EASINGS.default}, opacity 200ms ease-out`,
  },
  animate: {
    opacity: 1,
    strokeDashoffset: "0%",
    strokeDasharray: "100%",
    transition: `stroke-dashoffset ${MOTION_DURATIONS.heroPlantGrow}ms ${MOTION_EASINGS.default}, opacity 200ms ease-out`,
  },
  reducedMotion: {
    initial: { opacity: 1, strokeDashoffset: "0%", strokeDasharray: "none", transition: "none" },
    animate: { opacity: 1, strokeDashoffset: "0%", strokeDasharray: "none", transition: "none" },
  },
};

/**
 * 3. crossFade: 200ms clean opacity transition (e.g. mask overlay toggle).
 * Reduced motion fallback: instant swap (0ms).
 */
export const crossFadeVariant: SharedVariant = {
  initial: {
    opacity: 0,
    transition: `opacity ${MOTION_DURATIONS.maskToggle}ms ease-out`,
  },
  animate: {
    opacity: 1,
    transition: `opacity ${MOTION_DURATIONS.maskToggle}ms ease-out`,
  },
  reducedMotion: {
    initial: { opacity: 1, transition: "none" },
    animate: { opacity: 1, transition: "none" },
  },
};

/**
 * 4. scalePress: interactive micro-interaction for cards and buttons.
 * Hover lifts 3px (150ms); active scales 0.98 (80ms).
 * Reduced motion fallback: no translation or scaling.
 */
export const scalePressVariant = {
  hover: "translateY(-3px)",
  active: "scale(0.98)",
  reducedHover: "none",
  reducedActive: "none",
};

/**
 * Stagger helper: calculates delay for child elements based on index.
 */
export function getStaggerDelay(index: number, staggerMs: number = MOTION_DURATIONS.cropCardStagger): number {
  return index * staggerMs;
}
