/**
 * Centralized motion durations and timing tokens from DESIGN.md Motion System.
 *
 * Invariants:
 * - No component hard-codes duration values.
 * - Only transform, opacity, and SVG stroke properties may be animated.
 * - UI feedback budget: 120–200 ms. Entrances: 250–450 ms.
 */

export const MOTION_DURATIONS = {
  // Entrances and views
  heroPlantGrow: 1200,      // ~1.2s S1 hero SVG plant stroke draw
  cropCardFadeUp: 300,     // 300ms S1 crop cards staggered entrance
  cropCardStagger: 60,     // 60ms stagger between cards
  cropCardHoverLift: 150,  // 150ms hover lift + tilt
  pageTransition: 250,     // 250ms cross-fade + 8px slide
  resultPanelFade: 300,    // 300ms result panels fade-up
  resultPanelOffset: 80,   // 80ms offset between image and sensor panel
  leafCardFade: 250,       // 250ms leaf card entrance
  leafCardStagger: 50,     // 50ms stagger between leaf cards

  // Feedback & micro-interactions
  feedbackPress: 80,       // 80ms scale .98 button press
  hoverLift: 150,          // 150ms card/button hover lift
  maskToggle: 200,         // 200ms cross-fade between image and mask overlay
  saveToastSlide: 300,     // 300ms toast slide + leaf-check draw
  errorFade: 150,          // 150ms inline error fade (no shake)

  // Gauges, progress, and indicators
  confidenceMeterFill: 500,// 500ms ease-out fill for confidence bar
  sensorReadingCount: 400, // 400ms needle/bar count-up into band
  stepProgress: 350,       // per-step vine growth transition
  livePulse: 600,          // 600ms pulse when fresh telemetry arrives

  // Loops (pausable when tab hidden)
  loadingLoop: 1400,       // 1.4s droplet fills leaf outline loop
  decorativeLoop: 40000,   // 40s+ slow drifting leaf background loop
} as const;

export type MotionDurationKey = keyof typeof MOTION_DURATIONS;

/**
 * Whitelist of composited CSS properties permitted to animate under DESIGN.md.
 * Layout properties (top, left, width, height, margin) are strictly forbidden.
 */
export const ALLOWED_ANIMATED_PROPERTIES = [
  "transform",
  "opacity",
  "strokeDashoffset",
  "strokeDasharray",
] as const;

export const MOTION_EASINGS = {
  default: "cubic-bezier(0.16, 1, 0.3, 1)", // Smooth ease-out
  inOut: "cubic-bezier(0.65, 0, 0.35, 1)",
  linear: "linear",
} as const;
