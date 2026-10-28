"use client";

import React from "react";
import { Card } from "../ui/Card";

export const OVERVIEW_SECTION_28_LIMITATIONS = [
  "Some sensor labels are generated from thresholds rather than observed plant-health outcomes.",
  "Lettuce sensor labels reflect project-defined reference bands; the available report describes 726 real chemistry readings, not thousands of independent chemistry observations.",
  "The lettuce training set was oversampled; the held-out test set retained real rows and original held-out distribution.",
  "Spinach sensor data mixes real air readings with adjusted humidity and synthetic/derived water chemistry features.",
  "Tomato sensor data uses transformed features and synthetically derived water temperature.",
  "Tomato source units/provenance are not all independently verified.",
  "Tomato source values appear to have fixed-range structure, which suggests simulation but does not prove it.",
  "Class balance in prepared sensor datasets may be artificial and not representative of field prevalence.",
  "Image and sensor records are not generally paired.",
  "Different image and sensor taxonomies prevent automatic direct comparison without a validated mapping.",
  "Whole-plant and individual-leaf image inputs are not interchangeable.",
  "A model trained on rule-derived labels may reproduce the rules without learning biological causation.",
  "Performance depends on the documented dataset, split, preprocessing, and label generation; it is not a guarantee of field accuracy.",
  "Dataset and model licenses/redistribution terms must be checked.",
  "`droop_angle` is conditional on a defensible reproducible implementation; otherwise it must be removed/replaced and all dependent features/models rebuilt.",
  "Campus/live tomato sensor inference is not validated until transformed training features are mapped to actual device measurements.",
  "A model confidence score is not automatically a calibrated probability.",
  "Recommendations are advisory and must remain conservative.",
];

export const LimitationsView: React.FC = () => {
  return (
    <div className="space-y-6 text-left" data-testid="limitations-view">
      {/* Required Disclosure Banner */}
      <div className="p-4 bg-[#FFFDF7] border-l-4 border-[var(--clay)] rounded-[var(--radius-md)] text-xs text-[var(--clay)] font-medium leading-relaxed shadow-xs">
        <span className="font-bold uppercase tracking-wider block text-[11px] mb-1">
          Architectural Transparency Standard
        </span>
        Scores mean agreement with project rules/labels on a documented split.
        Hydroponic diagnosis models operate within bounded empirical conditions and must not be
        relied upon as certified biological or pathogen determinations.
      </div>

      {/* Main Limitations Card: Section 28 Disclosures */}
      <Card
        variant="warning"
        className="p-6 border border-[var(--warning)]/50 bg-[#FFFDF7] shadow-[var(--shadow-warm-sm)]"
      >
        <div className="flex items-center gap-2.5 mb-3">
          <div className="w-8 h-8 rounded-full bg-[#FFF4DC] text-[var(--warning)] flex items-center justify-center shrink-0">
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>

          <div>
            <h3 className="text-base font-bold text-[var(--bark)]">
              Overview §28: Known Limitations &amp; Required Disclosures
            </h3>
            <p className="text-xs text-[var(--stone)]">
              The following 18 documented caveats govern all models, datasets, and inferences in this project:
            </p>
          </div>
        </div>

        <ol className="list-decimal list-inside space-y-2.5 text-xs text-[var(--bark)]/90 leading-relaxed pt-2 border-t border-[var(--border)]/60">
          {OVERVIEW_SECTION_28_LIMITATIONS.map((limitation, idx) => (
            <li key={idx} className="pl-1">
              <span className="font-medium">{limitation}</span>
            </li>
          ))}
        </ol>
      </Card>

      {/* Boundary Principles Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-4 bg-[var(--surface)] border border-[var(--border)] text-xs space-y-1.5">
          <h4 className="font-bold text-[var(--bark)] flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[var(--clay)]" />
            No Multimodal Fusion
          </h4>
          <p className="text-[var(--stone)] leading-relaxed">
            Image and sensor inputs are collected independently without paired timestamps. Multimodal
            fusion without paired data is mathematically prohibited per TECH_RULES §15.
          </p>
        </Card>

        <Card className="p-4 bg-[var(--surface)] border border-[var(--border)] text-xs space-y-1.5">
          <h4 className="font-bold text-[var(--bark)] flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[var(--warning)]" />
            Empirical Calibration
          </h4>
          <p className="text-[var(--stone)] leading-relaxed">
            Confidence scores reflect softmax activation probabilities on isolated test splits, not
            calibrated real-world Bayesian likelihoods.
          </p>
        </Card>

        <Card className="p-4 bg-[var(--surface)] border border-[var(--border)] text-xs space-y-1.5">
          <h4 className="font-bold text-[var(--bark)] flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[var(--forest)]" />
            Deterministic Fallback
          </h4>
          <p className="text-[var(--stone)] leading-relaxed">
            When external API or LLM services are degraded, deterministic agronomic rules and
            hard limits take absolute precedence.
          </p>
        </Card>
      </div>
    </div>
  );
};
