"use client";

import React, { useState } from "react";
import { Button, Card } from "@/components/ui";
import { ImageDropzone, InputMode, InputModeToggle } from "@/components/analyze";

export default function AnalyzePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [inputMode, setInputMode] = useState<InputMode | null>(null);
  const [selectedCrop, setSelectedCrop] = useState<string>("lettuce");

  return (
    <div className="space-y-8 text-left">
      <div className="border-b border-[var(--border)] pb-4">
        <span className="text-xs font-mono font-semibold uppercase tracking-wider text-[var(--clay)]">
          Diagnostic Workspace
        </span>
        <h1 className="text-2xl sm:text-3xl font-serif text-[var(--forest)] tracking-tight mt-1">
          Crop Nutrition &amp; Anomaly Analysis
        </h1>
        <p className="text-sm text-[var(--stone)] mt-1">
          Provide plant leaf imagery, reservoir sensor measurements, or stream live data from the campus adapter.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Image & Input Collection */}
        <div className="lg:col-span-7 space-y-6">
          <Card className="space-y-5">
            <div className="flex items-center justify-between border-b border-[var(--border)] pb-3">
              <div className="flex items-center gap-2">
                <span className="text-lg">📷</span>
                <h2 className="text-base font-serif font-bold text-[var(--forest)]">
                  1. Plant Imagery Ingestion
                </h2>
              </div>
              <span className="text-xs bg-[#EDF6F8] text-[var(--water)] px-2.5 py-0.5 rounded-[var(--radius-full)] font-medium border border-[var(--water)]/30">
                SAM 2.1 + Vision
              </span>
            </div>

            {/* Input Mode Disambiguation */}
            <InputModeToggle
              value={inputMode}
              onChange={setInputMode}
              error={!inputMode && selectedFile ? "Please explicitly select whether this is a Single Leaf or Whole Plant photo." : undefined}
            />

            {/* Drag & Drop File Picker with Client Pre-Checks */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[var(--bark)] uppercase tracking-wide">
                Leaf / Canopy Photo Upload
              </label>
              <ImageDropzone
                selectedFile={selectedFile}
                onFileSelected={setSelectedFile}
              />
            </div>
          </Card>

          {/* Sensor Input Card */}
          <Card className="space-y-4">
            <div className="flex items-center justify-between border-b border-[var(--border)] pb-3">
              <div className="flex items-center gap-2">
                <span className="text-lg">🧪</span>
                <h2 className="text-base font-serif font-bold text-[var(--forest)]">
                  2. Hydroponic Reservoir Telemetry
                </h2>
              </div>
              <span className="text-xs bg-[var(--sand)] text-[var(--stone)] px-2.5 py-0.5 rounded-[var(--radius-full)] font-medium border border-[var(--border)]">
                Random Forest
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4 text-xs">
              <div className="space-y-1">
                <label className="font-semibold text-[var(--bark)]">pH Level [pH]</label>
                <input
                  type="number"
                  step="0.1"
                  placeholder="e.g. 6.2"
                  className="w-full rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] px-3 py-1.5 text-sm"
                />
              </div>
              <div className="space-y-1">
                <label className="font-semibold text-[var(--bark)]">EC [mS/cm]</label>
                <input
                  type="number"
                  step="0.05"
                  placeholder="e.g. 1.8"
                  className="w-full rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] px-3 py-1.5 text-sm"
                />
              </div>
            </div>
          </Card>
        </div>

        {/* Right Column: Execution Summary */}
        <div className="lg:col-span-5 space-y-6">
          <Card className="space-y-4 sticky top-24">
            <h3 className="text-sm font-serif font-bold text-[var(--forest)] border-b border-[var(--border)] pb-2">
              Analysis Configuration
            </h3>

            <div className="space-y-2 text-xs">
              <div className="flex justify-between py-1 border-b border-[var(--border)]/50">
                <span className="text-[var(--stone)]">Selected Crop:</span>
                <select
                  value={selectedCrop}
                  onChange={(e) => setSelectedCrop(e.target.value)}
                  className="bg-transparent font-semibold text-[var(--bark)] outline-none cursor-pointer"
                >
                  <option value="lettuce">Lettuce (Lactuca sativa)</option>
                  <option value="spinach">Spinach (Spinacia oleracea)</option>
                  <option value="tomato">Tomato (Solanum lycopersicum)</option>
                </select>
              </div>

              <div className="flex justify-between py-1 border-b border-[var(--border)]/50">
                <span className="text-[var(--stone)]">Input Mode:</span>
                <span className="font-semibold text-[var(--bark)]">
                  {inputMode === "single_leaf"
                    ? "Single Leaf"
                    : inputMode === "plant"
                    ? "Whole Plant"
                    : "Not Selected (Required)"}
                </span>
              </div>

              <div className="flex justify-between py-1 border-b border-[var(--border)]/50">
                <span className="text-[var(--stone)]">Image Ready:</span>
                <span className="font-semibold text-[var(--bark)]">
                  {selectedFile ? `${selectedFile.name}` : "None"}
                </span>
              </div>
            </div>

            <Button
              variant="primary"
              size="lg"
              disabled={!selectedFile || !inputMode}
              className="w-full mt-4"
            >
              Execute Prediction
            </Button>

            <p className="text-[11px] text-[var(--stone)] text-center">
              Advisory notice: Sensor and vision models are evaluated independently without score averaging.
            </p>
          </Card>
        </div>
      </div>
    </div>
  );
}
