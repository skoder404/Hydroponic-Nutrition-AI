"use client";

import React from "react";
import { Card } from "../ui/Card";

export const DatasetsView: React.FC = () => {
  const dataTypes = [
    {
      name: "Measured",
      badgeClass: "bg-[#EBF7EE] text-[var(--forest)] border-[var(--forest)]/40",
      description:
        "True physical electrical or optical sensor readings directly acquired from hardware electrodes and probes.",
      examples: "pH electrodes, EC probes, physical air temp sensors",
    },
    {
      name: "Adjusted",
      badgeClass: "bg-[#EDF6F8] text-[var(--water)] border-[var(--water)]/40",
      description:
        "Measured telemetry subjected to hardware calibration curves, offset adjustments, or temperature compensation.",
      examples: "Relative humidity offset compensation, EC normalized to 25°C",
    },
    {
      name: "Derived",
      badgeClass: "bg-[#F3EFF8] text-[#5C3C8A] border-[#5C3C8A]/40",
      description:
        "Calculated mathematical combinations, GLCM texture metrics, or color histogram transformations.",
      examples: "TDS derived from EC (500-factor), GLCM contrast and homogeneity",
    },
    {
      name: "Synthetic",
      badgeClass: "bg-[#FFF4DC] text-[var(--warning)] border-[var(--warning)]/50 font-bold",
      description:
        "Simulated numerical distributions generated to bridge hardware sensor gaps during offline development.",
      examples: "Tomato water temperature model with controlled gaussian noise",
    },
    {
      name: "Rule-generated",
      badgeClass: "bg-[#F7F4EC] text-[var(--clay)] border-[var(--clay)]/40",
      description:
        "Target ground-truth labels assigned via agronomic expert threshold lookup tables rather than lab bio-assays.",
      examples: "Lettuce nutrient stress labels, Spinach water chemistry fault classes",
    },
  ];

  const datasets = [
    {
      crop: "Lettuce (Lactuca sativa)",
      source: "University Hydroponics Repository + Segmented Leaf Imagery",
      records: "726 real water chemistry records (oversampled for train); 2,548 leaf crops",
      splits: "Time-grouped holdout partition; test split strictly untouched",
      classes: "4 foliar classes (Image); 9 nutrient deficiency/excess states (Sensor)",
    },
    {
      crop: "Spinach (Spinacia oleracea)",
      source: "Open Hydroponic Telemetry + Foliar Pathology Dataset",
      records: "1,440 hourly telemetry observations; 18,302 leaf images",
      splits: "Block-stratified time splits (train: 70%, val: 15%, test: 15%)",
      classes: "6 condition classes (Image); 7 water chemistry fault states (Sensor)",
    },
    {
      crop: "Tomato (Solanum lycopersicum)",
      source: "Commercial Greenhouse Telemetry + Multiclass Foliar Survey",
      records: "33,564 total leaf images; 35,760 tabular telemetry points",
      splits: "Calendar-month grouping split (test set preserved holdout)",
      classes: "11 foliar classes (including Powdery Mildew); 10 sensor states",
    },
  ];

  return (
    <div className="space-y-6 text-left" data-testid="datasets-view">
      {/* 1. Data-Type Legend (DESIGN S6 requirement) */}
      <div>
        <div className="mb-3">
          <h3 className="text-base font-bold text-[var(--bark)]">
            Data-Type Provenance Legend
          </h3>
          <p className="text-xs text-[var(--stone)]">
            Every feature and target in our pipeline carries explicit provenance classifications:
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {dataTypes.map((dt) => (
            <Card
              key={dt.name}
              variant="default"
              className="p-4 border border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-warm-sm)]"
            >
              <div className="flex items-center justify-between mb-2">
                <span
                  className={`text-xs px-2.5 py-0.5 rounded-full border font-mono font-semibold ${dt.badgeClass}`}
                >
                  {dt.name}
                </span>
              </div>
              <p className="text-xs text-[var(--bark)] mb-2 leading-relaxed">
                {dt.description}
              </p>
              <div className="text-[11px] text-[var(--stone)] pt-2 border-t border-[var(--border)]/60 font-mono">
                <strong className="text-[var(--bark)]">Examples:</strong> {dt.examples}
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* 2. Crop Dataset Overview Cards */}
      <div>
        <div className="mb-3">
          <h3 className="text-base font-bold text-[var(--bark)]">
            Documented Cultivar Datasets
          </h3>
          <p className="text-xs text-[var(--stone)]">
            Partition counts, feature spaces, and label generation protocols per crop:
          </p>
        </div>

        <div className="space-y-3">
          {datasets.map((ds) => (
            <Card
              key={ds.crop}
              variant="default"
              className="p-4 border border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-warm-sm)]"
            >
              <h4 className="text-sm font-bold text-[var(--bark)] mb-2">
                {ds.crop}
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
                <div>
                  <span className="text-[11px] uppercase tracking-wider text-[var(--stone)] block font-semibold">
                    Source Provenance
                  </span>
                  <span className="text-[var(--bark)]">{ds.source}</span>
                </div>
                <div>
                  <span className="text-[11px] uppercase tracking-wider text-[var(--stone)] block font-semibold">
                    Dataset Volume
                  </span>
                  <span className="text-[var(--bark)] font-mono">{ds.records}</span>
                </div>
                <div>
                  <span className="text-[11px] uppercase tracking-wider text-[var(--stone)] block font-semibold">
                    Split Protocol
                  </span>
                  <span className="text-[var(--bark)]">{ds.splits}</span>
                </div>
                <div>
                  <span className="text-[11px] uppercase tracking-wider text-[var(--stone)] block font-semibold">
                    Target Taxonomy
                  </span>
                  <span className="text-[var(--bark)] font-mono">{ds.classes}</span>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
};
