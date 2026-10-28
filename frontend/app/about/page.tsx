"use client";

import React, { useState, useId } from "react";
import Link from "next/link";
import { Badge, Card, ConfidenceMeter } from "@/components/ui";

interface StepperStep {
  id: string;
  number: number;
  title: string;
  shortDesc: string;
  fullDesc: string;
  technicalDetails: string[];
  icon: React.ReactNode;
}

interface GlossaryEntry {
  term: string;
  category: "Evaluation" | "Vision" | "Methodology" | "Safety";
  summary: string;
  conceptNoteRef: string;
  detailedText: string;
}

export default function AboutHowItWorksPage() {
  const [activeStep, setActiveStep] = useState<number>(0);
  const [glossaryFilter, setGlossaryFilter] = useState<string>("All");
  const [expandedTerm, setExpandedTerm] = useState<string | null>("Confidence");
  const stepperHeadingId = useId();

  const steps: StepperStep[] = [
    {
      id: "step-ingest",
      number: 1,
      title: "Data Acquisition & Telemetry",
      shortDesc: "Real-time water chemistry and whole-plant photography",
      fullDesc:
        "The system collects high-frequency sensor readings (pH, electrical conductivity, water temperature, dissolved oxygen) alongside high-resolution RGB plant photography captured under greenhouse illumination.",
      technicalDetails: [
        "Sensors queried via Modbus RTU / Campus Server adapter every 30 seconds",
        "Client-side image validation enforces minimum resolution (256x256), aspect ratio sanity, and file integrity",
        "Sensors validated against per-crop agronomic bounds (lettuce, spinach, tomato)",
      ],
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
          />
        </svg>
      ),
    },
    {
      id: "step-segment",
      number: 2,
      title: "Per-Leaf Instance Segmentation",
      shortDesc: "Meta SAM 2.1 leaf isolation and mask geometry validation",
      fullDesc:
        "Instead of attempting whole-canopy classification where overlapping leaves distort diagnosis, the vision pipeline isolates individual leaves using Meta SAM 2.1 Hiera-Tiny foundation model, applying strict geometric validation.",
      technicalDetails: [
        "Rejects edge-clipping, sub-threshold areas (<0.5% image), and non-convex mask artifacts",
        "Retains per-leaf source mapping linking each crop back to its parent plant coordinates",
        "Extracts GLCM texture (contrast, entropy) and morphology without destructive spatial pooling",
      ],
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
          />
        </svg>
      ),
    },
    {
      id: "step-classify",
      number: 3,
      title: "Dual-Pipeline Machine Learning",
      shortDesc: "Independent sensor and leaf classifiers with holdout-derived thresholds",
      fullDesc:
        "Water sensor telemetry and individual leaf crops are evaluated through separate, purpose-built models. Image scores are never averaged across the canopy to prevent masking isolated focal deficiencies.",
      technicalDetails: [
        "Sensors evaluated via calibrated classical ensembles (Random Forest, Gradient Boosting, SVM)",
        "Leaf imagery evaluated via custom CNNs and EfficientNet-B0 with per-leaf uncertainty thresholds",
        "Every model evaluated strictly on isolated validation partitions using Macro-F1",
      ],
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
          />
        </svg>
      ),
    },
    {
      id: "step-advise",
      number: 4,
      title: "Advisory Synthesis & Fallback",
      shortDesc: "Deterministic local rules with optional generative synthesis",
      fullDesc:
        "The system generates cautious dosing and environmental adjustments. If generative AI is unavailable or produces out-of-band parameters, deterministic agronomic rules take over instantly with full audit trails.",
      technicalDetails: [
        "Hard validation guards clamp pH titration rates and EC nutrient adjustments to safe agronomic increments",
        "Transparent source badges indicate whether recommendations originated from Rules or Generative AI",
        "Operator confirmation required before executing any pump dosing adjustment",
      ],
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      ),
    },
  ];

  const glossaryEntries: GlossaryEntry[] = [
    {
      term: "Confidence",
      category: "Evaluation",
      summary: "A mathematical classifier score, not a calibrated real-world probability.",
      conceptNoteRef: "decision-tree-hyperparameters-depth-and-overfitting.md",
      detailedText:
        "Confidence is the numerical score generated by a machine learning model's output layer (such as softmax activations or margin distances). It indicates how strongly the model ranked the chosen condition relative to other training classes. Crucially, raw confidence scores are uncalibrated: an output score of 0.85 does NOT mean there is an 85% statistical likelihood of disease in the physical greenhouse. The interface labels tiers as Higher, Moderate, or Uncertain using validation-calibrated thresholds rather than subjective percentages.",
    },
    {
      term: "Macro-F1",
      category: "Evaluation",
      summary: "The unweighted arithmetic average of per-class F1-scores, balancing all conditions equally.",
      conceptNoteRef: "confusion-matrix-precisionrecall-macrof1.md",
      detailedText:
        "Macro-F1 calculates the harmonic mean of precision and recall for each individual class and computes their unweighted mean: Macro-F1 = (1/C) * Σ F1_c. In agricultural datasets where healthy samples often outnumber rare nutritional deficiencies (such as iron or potassium chlorosis), overall accuracy can appear high (>90%) while completely failing on critical deficiencies. Macro-F1 treats every deficiency class as equally vital, preventing majority-class dominance.",
    },
    {
      term: "Rule-Derived Label",
      category: "Methodology",
      summary: "Synthetic ground-truth labels assigned by algorithmic formulas rather than human pathology.",
      conceptNoteRef: "why-rulederived-labels-inflate-scores.md",
      detailedText:
        "In sensor telemetry datasets, nutrient deficiency labels are derived from deterministic rules based on target agronomic EC and pH bands (e.g. EC < 1.0 mS/cm implies low nutrient concentration). When models are trained and tested on rule-derived data, classical classifiers easily learn the boundary conditions, reporting near-perfect scores that do not reflect biological noise. The interface flags sensor predictions as 'Rule-derived' so operators understand the baseline nature of the telemetry.",
    },
    {
      term: "Segmentation",
      category: "Vision",
      summary: "Isolating individual leaf masks from whole-plant images to assess condition leaf-by-leaf.",
      conceptNoteRef: "instance-segmentation-for-perleaf-prediction-sam.md",
      detailedText:
        "Standard plant health models take an entire photograph and output a single classification label. In real hydroponic cultivation, nutrient deficiencies manifest progressively: mobile nutrients (N, P, K, Mg) cause chlorosis on older, lower leaves while new growth remains healthy; immobile nutrients (Fe, Ca) affect young crown leaves first. By segmenting leaves individually with SAM 2.1, our pipeline preserves spatial granularity across the plant canopy.",
    },
    {
      term: "Uncertainty",
      category: "Safety",
      summary: "Calibrated threshold bounds where the model refuses to guess and alerts the operator.",
      conceptNoteRef: "weak-labels-and-instancelevel-perleaf-prediction.md",
      detailedText:
        "Uncertainty thresholds represent empirical safety cutoffs derived strictly from validation partition performance. When a leaf crop exhibits anomalous GLCM texture, unusual aspect ratios, or an ambiguous softmax distribution, the model flags the leaf as 'Uncertain'. Rather than forcing an overconfident incorrect diagnosis, the interface advises human visual inspection and water testing.",
    },
  ];

  const categories = ["All", "Evaluation", "Vision", "Methodology", "Safety"];
  const filteredGlossary =
    glossaryFilter === "All"
      ? glossaryEntries
      : glossaryEntries.filter((g) => g.category === glossaryFilter);

  return (
    <div className="space-y-12 max-w-5xl mx-auto py-4">
      {/* Page Header */}
      <div className="border-b border-[var(--border)] pb-8">
        <div className="flex flex-wrap items-center gap-2 mb-3">
          <Badge variant="live">System Architecture</Badge>
          <Badge variant="rules">Non-Diagnostic Support</Badge>
          <Badge variant="experimental">Research Prototype</Badge>
        </div>
        <h1 className="text-3xl sm:text-4xl font-serif font-bold text-[var(--forest)] tracking-tight">
          About & How It Works
        </h1>
        <p className="text-base text-[var(--stone)] mt-2 max-w-3xl leading-relaxed">
          Hydroponic Nutrition AI is a dual-pipeline decision-support system designed for precision
          hydroponic cultivation. It pairs real-time water chemistry monitoring with per-leaf
          computer vision to detect nutrient imbalances early.
        </p>
      </div>

      {/* Mandatory Non-Diagnostic Disclaimers (Acceptance Criterion 3) */}
      <section
        aria-label="Important Operational Disclaimers"
        className="rounded-[var(--radius-lg)] p-6 bg-[#FFF9EC] border-2 border-[var(--warning)]/50 shadow-sm"
      >
        <div className="flex items-start gap-3.5">
          <div className="p-2 rounded-full bg-[var(--warning)]/15 text-[var(--warning)] shrink-0 mt-0.5">
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
              <path
                fillRule="evenodd"
                d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
          </div>
          <div className="space-y-2">
            <h2 className="text-base font-bold text-[var(--bark)]">
              Predicted — Not a Diagnosis
            </h2>
            <p className="text-sm text-[var(--bark)]/90 leading-relaxed">
              This application provides <strong>decision support only</strong> and makes{" "}
              <strong>no claims of automated botanical diagnosis or guaranteed accuracy</strong>.
              Foliar chlorosis, necroses, and wilting can result from multiple overlapping factors
              including root asphyxiation, fungal root rots (<em>Pythium</em>), viral pathogens, or
              microclimate stress.
            </p>
            <p className="text-xs text-[var(--stone)]">
              Operators must always verify laboratory water quality metrics, inspect root health,
              and validate chemical dosing prior to adjusting nutrient injection reservoirs.
            </p>
          </div>
        </div>
      </section>

      {/* Interactive 4-Step Pipeline Stepper (Acceptance Criterion 2: Keyboard Accessible) */}
      <section aria-labelledby={stepperHeadingId} className="space-y-6">
        <div>
          <span className="text-xs font-mono font-semibold uppercase tracking-wider text-[var(--clay)]">
            Workflow Overview
          </span>
          <h2 id={stepperHeadingId} className="text-2xl font-serif text-[var(--forest)] mt-0.5">
            The 4-Stage Nutrition Intelligence Pipeline
          </h2>
          <p className="text-sm text-[var(--stone)] mt-1">
            Navigate through each step of the pipeline using buttons or keyboard (Tab & Enter / Space).
          </p>
        </div>

        {/* Stepper Navigation Buttons */}
        <div
          role="tablist"
          aria-label="Pipeline Stages"
          className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 p-1.5 rounded-[var(--radius-lg)] bg-[var(--sand)]/50 border border-[var(--border)]"
        >
          {steps.map((s, idx) => {
            const isCurrent = activeStep === idx;
            return (
              <button
                key={s.id}
                role="tab"
                id={`tab-${s.id}`}
                aria-selected={isCurrent}
                aria-controls={`panel-${s.id}`}
                tabIndex={isCurrent ? 0 : -1}
                onClick={() => setActiveStep(idx)}
                onKeyDown={(e) => {
                  if (e.key === "ArrowRight") {
                    setActiveStep((prev) => (prev + 1) % steps.length);
                  } else if (e.key === "ArrowLeft") {
                    setActiveStep((prev) => (prev - 1 + steps.length) % steps.length);
                  }
                }}
                className={`flex items-center gap-2.5 p-3 rounded-[var(--radius-md)] text-left transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-[var(--moss)] ${
                  isCurrent
                    ? "bg-[var(--surface)] text-[var(--forest)] shadow-[var(--shadow-warm-sm)] font-semibold border border-[var(--border)]"
                    : "text-[var(--stone)] hover:bg-[var(--surface)]/60 hover:text-[var(--bark)]"
                }`}
              >
                <span
                  className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0 transition-colors ${
                    isCurrent
                      ? "bg-[var(--forest)] text-[var(--bg)]"
                      : "bg-[var(--sand)] text-[var(--stone)]"
                  }`}
                >
                  {s.number}
                </span>
                <span className="text-xs truncate">{s.title.split(" ")[0]}</span>
              </button>
            );
          })}
        </div>

        {/* Active Step Content Card */}
        {steps.map((s, idx) => {
          if (activeStep !== idx) return null;
          return (
            <div
              key={s.id}
              role="tabpanel"
              id={`panel-${s.id}`}
              aria-labelledby={`tab-${s.id}`}
              tabIndex={0}
              className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow-warm-sm)] focus:outline-none focus:ring-2 focus:ring-[var(--moss)] space-y-6 animate-in fade-in-50 duration-200"
            >
              <div className="flex items-start justify-between gap-4 border-b border-[var(--border)] pb-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-[var(--clay)]">
                    <span>Stage {s.number} of 4</span>
                    <span>•</span>
                    <span>{s.shortDesc}</span>
                  </div>
                  <h3 className="text-xl font-serif font-bold text-[var(--forest)]">
                    {s.title}
                  </h3>
                </div>
                <div className="p-3 rounded-full bg-[var(--sand)]/70 text-[var(--forest)] shrink-0">
                  {s.icon}
                </div>
              </div>

              <p className="text-sm text-[var(--bark)] leading-relaxed">
                {s.fullDesc}
              </p>

              <div className="space-y-2 bg-[var(--sand)]/20 rounded-[var(--radius-md)] p-4 border border-[var(--border)]">
                <h4 className="text-xs font-bold uppercase tracking-wider text-[var(--bark)]">
                  Key Technical Safeguards & Mechanics
                </h4>
                <ul className="list-disc list-inside space-y-1.5 text-xs text-[var(--stone)]">
                  {s.technicalDetails.map((tech, i) => (
                    <li key={i} className="leading-normal">
                      {tech}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="flex items-center justify-between pt-2">
                <button
                  type="button"
                  disabled={activeStep === 0}
                  onClick={() => setActiveStep((prev) => Math.max(0, prev - 1))}
                  className="px-3.5 py-1.5 rounded-[var(--radius-md)] text-xs font-medium border border-[var(--border)] text-[var(--bark)] hover:bg-[var(--sand)] disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-[var(--moss)] transition-colors"
                >
                  ← Previous Stage
                </button>
                <button
                  type="button"
                  disabled={activeStep === steps.length - 1}
                  onClick={() => setActiveStep((prev) => Math.min(steps.length - 1, prev + 1))}
                  className="px-3.5 py-1.5 rounded-[var(--radius-md)] text-xs font-medium bg-[var(--forest)] text-[#FAF6EC] hover:bg-[var(--moss)] disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-[var(--moss)] transition-colors"
                >
                  Next Stage →
                </button>
              </div>
            </div>
          );
        })}
      </section>

      {/* Model Confidence & Uncertainty Demonstration */}
      <section className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] p-6 space-y-4">
        <div>
          <span className="text-xs font-mono font-semibold uppercase tracking-wider text-[var(--clay)]">
            UI Primitives in Action
          </span>
          <h2 className="text-xl font-serif font-bold text-[var(--forest)] mt-0.5">
            Interpreting Confidence & Uncertainty Tiers
          </h2>
          <p className="text-xs text-[var(--stone)] mt-1">
            Components adhere to WCAG AA: distinct icons, hatch patterns, and lighter contrast ensure no tier relies on color alone.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          <div>
            <h3 className="text-xs font-bold text-[var(--forest)] mb-1.5">Higher Tier (≥ 75%)</h3>
            <ConfidenceMeter
              score={0.88}
              thresholds={{ high: 0.75, moderate: 0.5 }}
              label="Higher demo"
            />
          </div>
          <div>
            <h3 className="text-xs font-bold text-[#8A5A00] mb-1.5">Moderate Tier (50%–75%)</h3>
            <ConfidenceMeter
              score={0.62}
              thresholds={{ high: 0.75, moderate: 0.5 }}
              label="Moderate demo"
            />
          </div>
          <div>
            <h3 className="text-xs font-bold text-[var(--stone)] mb-1.5">Uncertain Tier (&lt; 50%)</h3>
            <ConfidenceMeter
              score={0.34}
              thresholds={{ high: 0.75, moderate: 0.5 }}
              label="Uncertain demo"
            />
          </div>
        </div>
      </section>

      {/* Comprehensive Concept Notes Glossary (Acceptance Criterion 1) */}
      <section aria-labelledby="glossary-heading" className="space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
          <div>
            <span className="text-xs font-mono font-semibold uppercase tracking-wider text-[var(--clay)]">
              Conceptual Foundations
            </span>
            <h2 id="glossary-heading" className="text-2xl font-serif text-[var(--forest)] mt-0.5">
              Technical Glossary & Concept Notes
            </h2>
            <p className="text-sm text-[var(--stone)] mt-1">
              Definitions derived directly from project engineering notes and verified learning curricula.
            </p>
          </div>

          {/* Filter Pills */}
          <div
            role="radiogroup"
            aria-label="Filter glossary by category"
            className="flex flex-wrap gap-1.5 bg-[var(--sand)]/50 p-1 rounded-[var(--radius-md)] border border-[var(--border)]"
          >
            {categories.map((cat) => (
              <button
                key={cat}
                type="button"
                role="radio"
                aria-checked={glossaryFilter === cat}
                onClick={() => setGlossaryFilter(cat)}
                className={`px-2.5 py-1 rounded-[var(--radius-sm)] text-xs font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-[var(--moss)] ${
                  glossaryFilter === cat
                    ? "bg-[var(--surface)] text-[var(--forest)] font-semibold shadow-xs"
                    : "text-[var(--stone)] hover:text-[var(--bark)]"
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Glossary Accordion Cards */}
        <div className="space-y-3">
          {filteredGlossary.map((entry) => {
            const isExpanded = expandedTerm === entry.term;
            return (
              <div
                key={entry.term}
                className="rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] overflow-hidden transition-shadow duration-150 hover:shadow-xs"
              >
                <button
                  type="button"
                  onClick={() => setExpandedTerm(isExpanded ? null : entry.term)}
                  aria-expanded={isExpanded}
                  className="w-full flex items-center justify-between p-4 text-left focus:outline-none focus:ring-2 focus:ring-[var(--moss)] transition-colors hover:bg-[var(--sand)]/20"
                >
                  <div className="flex items-center gap-3">
                    <span className="font-serif font-bold text-base text-[var(--bark)]">
                      {entry.term}
                    </span>
                    <span className="text-[11px] px-2 py-0.5 rounded-[var(--radius-full)] bg-[var(--sand)] text-[var(--stone)] font-mono">
                      {entry.category}
                    </span>
                  </div>

                  <div className="flex items-center gap-3 text-[var(--stone)]">
                    <span className="text-xs hidden md:inline">{entry.summary}</span>
                    <svg
                      className={`w-4 h-4 transition-transform duration-200 ${
                        isExpanded ? "rotate-180 text-[var(--forest)]" : ""
                      }`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 9l-7 7-7-7"
                      />
                    </svg>
                  </div>
                </button>

                {isExpanded && (
                  <div className="p-4 pt-1 border-t border-[var(--border)]/70 bg-[#FAF6EC]/40 space-y-3 text-xs text-[var(--bark)] leading-relaxed">
                    <p className="text-sm font-medium text-[var(--forest)]">
                      {entry.summary}
                    </p>
                    <p className="text-[var(--stone)]">
                      {entry.detailedText}
                    </p>
                    <div className="pt-2 flex items-center justify-between border-t border-[var(--border)]/40 text-[11px] text-[var(--stone)]">
                      <span>Concept reference: <code className="font-mono bg-[var(--sand)] px-1.5 py-0.5 rounded text-[var(--bark)]">{entry.conceptNoteRef}</code></span>
                      <span className="text-[var(--clay)] font-semibold">Strict Section 18.2 Compliant</span>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>

      {/* Navigation CTA */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-6 rounded-[var(--radius-lg)] bg-[var(--forest)] text-[#FAF6EC] shadow-md">
        <div>
          <h3 className="text-lg font-serif font-bold">Ready to analyze hydroponic health?</h3>
          <p className="text-xs text-[#D9CDB1] mt-0.5">
            Choose a crop to begin dual-pipeline inspection or monitor live sensor telemetry.
          </p>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <Link
            href="/"
            className="px-4 py-2 rounded-[var(--radius-md)] text-xs font-semibold bg-[#FAF6EC] text-[var(--forest)] hover:bg-[#FAF6EC]/90 focus:outline-none focus:ring-2 focus:ring-[#FAF6EC] transition-colors"
          >
            Start Inspection
          </Link>
          <Link
            href="/models"
            className="px-4 py-2 rounded-[var(--radius-md)] text-xs font-semibold border border-[#D9CDB1]/60 text-[#FAF6EC] hover:bg-[var(--moss)] focus:outline-none focus:ring-2 focus:ring-[#FAF6EC] transition-colors"
          >
            View Models & Limitations
          </Link>
        </div>
      </div>
    </div>
  );
}
