"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { MOTION_DURATIONS, MotionDurationKey } from "./tokens";
import {
  crossFadeVariant,
  fadeUpVariant,
  growVariant,
  SharedVariant,
} from "./variants";

interface MotionContextValue {
  prefersReducedMotion: boolean;
  isTabVisible: boolean;
  simulatedReducedMotion: boolean;
  setSimulatedReducedMotion: (val: boolean) => void;
  shouldAnimate: boolean;
  getDuration: (key: MotionDurationKey) => number;
  getVariant: (name: "fadeUp" | "grow" | "crossFade") => SharedVariant;
}

const MotionContext = createContext<MotionContextValue | undefined>(undefined);

export interface MotionProviderProps {
  children: React.ReactNode;
  initialSimulatedReducedMotion?: boolean;
}

export const MotionProvider: React.FC<MotionProviderProps> = ({
  children,
  initialSimulatedReducedMotion = false,
}) => {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState<boolean>(false);
  const [simulatedReducedMotion, setSimulatedReducedMotion] = useState<boolean>(
    initialSimulatedReducedMotion
  );
  const [isTabVisible, setIsTabVisible] = useState<boolean>(true);

  // 1. Monitor system prefers-reduced-motion media query
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;

    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setPrefersReducedMotion(mediaQuery.matches);

    const handleChange = (e: MediaQueryListEvent) => {
      setPrefersReducedMotion(e.matches);
    };

    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, []);

  // 2. Monitor tab visibility (pause decorative loops when tab is hidden)
  useEffect(() => {
    if (typeof document === "undefined") return;

    const handleVisibility = () => {
      setIsTabVisible(document.visibilityState === "visible");
    };

    document.addEventListener("visibilitychange", handleVisibility);
    return () => document.removeEventListener("visibilitychange", handleVisibility);
  }, []);

  const effectiveReducedMotion = prefersReducedMotion || simulatedReducedMotion;
  const shouldAnimate = !effectiveReducedMotion && isTabVisible;

  const getDuration = (key: MotionDurationKey): number => {
    if (effectiveReducedMotion) return 0;
    return MOTION_DURATIONS[key];
  };

  const getVariant = (name: "fadeUp" | "grow" | "crossFade"): SharedVariant => {
    switch (name) {
      case "fadeUp":
        return fadeUpVariant;
      case "grow":
        return growVariant;
      case "crossFade":
        return crossFadeVariant;
      default:
        return fadeUpVariant;
    }
  };

  return (
    <MotionContext.Provider
      value={{
        prefersReducedMotion: effectiveReducedMotion,
        isTabVisible,
        simulatedReducedMotion,
        setSimulatedReducedMotion,
        shouldAnimate,
        getDuration,
        getVariant,
      }}
    >
      {children}
    </MotionContext.Provider>
  );
};

export function useMotion(): MotionContextValue {
  const context = useContext(MotionContext);
  if (!context) {
    throw new Error("useMotion must be used within a <MotionProvider>");
  }
  return context;
}
