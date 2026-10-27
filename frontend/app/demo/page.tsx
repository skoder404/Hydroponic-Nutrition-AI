"use client";

import React, { useState } from "react";
import { Badge, Button, Card, Input, Toast, ToastVariant } from "@/components/ui";

export default function DesignSystemDemoPage() {
  const [reducedMotion, setReducedMotion] = useState(false);
  const [activeToasts, setActiveToasts] = useState<Array<{ id: number; variant: ToastVariant; title: string; message: string }>>([
    { id: 1, variant: "success", title: "Calibration Saved", message: "Leaf mask verified with 0.89 confidence." },
    { id: 2, variant: "warning", title: "Sub-optimal EC", message: "Observed EC is 1.1 mS/cm vs 1.6 target." },
  ]);

  const addToast = (variant: ToastVariant) => {
    const id = Date.now();
    const messages: Record<ToastVariant, { title: string; message: string }> = {
      success: { title: "Analysis Complete", message: "Isolated 4 healthy leaves." },
      info: { title: "Campus Telemetry", message: "New reading received from Modbus RTU." },
      warning: { title: "Uncertain Prediction", message: "Confidence 0.48 below threshold (0.65)." },
      error: { title: "Segmentation Failed", message: "Empty mask rejected by validator." },
    };
    setActiveToasts((prev) => [...prev, { id, variant, ...messages[variant] }]);
  };

  const removeToast = (id: number) => {
    setActiveToasts((prev) => prev.filter((t) => t.id !== id));
  };

  return (
    <div className={`min-h-screen bg-[var(--bg)] text-[var(--bark)] p-6 md:p-12 ${reducedMotion ? "reduced-motion-active" : ""}`}>
      <div className="max-w-6xl mx-auto space-y-12">
        {/* Header & Controls */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[var(--border)] pb-6">
          <div>
            <span className="text-xs font-mono font-semibold uppercase tracking-wider text-[var(--clay)]">
              T7-001 Design System Showcase
            </span>
            <h1 className="text-3xl md:text-4xl font-serif text-[var(--forest)] mt-1">
              Greenhouse Journal Design Tokens
            </h1>
            <p className="text-sm text-[var(--stone)] mt-1">
              Live demonstration of tokens, accessibility states, and UI primitives from DESIGN.md.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs font-semibold text-[var(--stone)] flex items-center gap-2 cursor-pointer bg-[var(--surface)] px-3 py-2 rounded-[var(--radius-md)] border border-[var(--border)] shadow-[var(--shadow-warm-sm)]">
              <input
                type="checkbox"
                checked={reducedMotion}
                onChange={(e) => setReducedMotion(e.target.checked)}
                className="rounded accent-[var(--forest)] cursor-pointer"
              />
              <span>Simulate `prefers-reduced-motion`</span>
            </label>
          </div>
        </div>

        {/* Section 1: Color Tokens Palette */}
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-serif text-[var(--forest)]">1. Color Tokens (DESIGN.md)</h2>
            <span className="text-xs text-[var(--stone)]">WCAG 2.1 AA Compliant Ratios</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-3 text-xs">
            {[
              { token: "--bg", name: "Cream (BG)", hex: "#FAF6EC", textDark: true },
              { token: "--surface", name: "Surface", hex: "#FFFFFF", textDark: true },
              { token: "--sand", name: "Sand", hex: "#EFE6D2", textDark: true },
              { token: "--forest", name: "Forest", hex: "#2F4A33", textDark: false },
              { token: "--moss", name: "Moss", hex: "#4C6B3C", textDark: false },
              { token: "--sage", name: "Sage", hex: "#8FA77A", textDark: true },
              { token: "--clay", name: "Clay", hex: "#A4502E", textDark: false },
              { token: "--bark", name: "Bark", hex: "#3E3022", textDark: false },
              { token: "--stone", name: "Stone", hex: "#665C4D", textDark: false },
              { token: "--border", name: "Border", hex: "#D9CDB1", textDark: true },
              { token: "--border-strong", name: "Border Strong", hex: "#857A67", textDark: false },
              { token: "--water", name: "Water", hex: "#2F6F7E", textDark: false },
              { token: "--success", name: "Success", hex: "#2E6B3A", textDark: false },
              { token: "--warning", name: "Warning", hex: "#8A5A00", textDark: false },
              { token: "--error", name: "Error", hex: "#A32A22", textDark: false },
              { token: "--info", name: "Info", hex: "#2A5F8F", textDark: false },
            ].map((col) => (
              <div
                key={col.token}
                className="rounded-[var(--radius-md)] border border-[var(--border)] overflow-hidden shadow-[var(--shadow-warm-sm)]"
              >
                <div
                  className="h-12 flex items-end p-1.5 font-mono text-[10px]"
                  style={{ backgroundColor: col.hex, color: col.textDark ? "#3E3022" : "#FFFFFF" }}
                >
                  {col.hex}
                </div>
                <div className="bg-[var(--surface)] p-2">
                  <div className="font-semibold text-[var(--bark)] truncate">{col.name}</div>
                  <div className="text-[10px] text-[var(--stone)] font-mono">{col.token}</div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Section 2: Button Component States */}
        <section className="space-y-4">
          <h2 className="text-xl font-serif text-[var(--forest)]">2. Button Component States</h2>
          <Card className="space-y-6">
            <div>
              <h3 className="text-sm font-semibold uppercase tracking-wider text-[var(--stone)] mb-3">
                Variants (Primary, Secondary, Ghost, Destructive)
              </h3>
              <div className="flex flex-wrap gap-4 items-center">
                <Button variant="primary">Primary (Forest)</Button>
                <Button variant="secondary">Secondary (Clay)</Button>
                <Button variant="ghost">Ghost (Subtle)</Button>
                <Button variant="destructive">Destructive (Error)</Button>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold uppercase tracking-wider text-[var(--stone)] mb-3">
                All Interactive States (Default, Hover, Focus-Visible, Loading, Disabled)
              </h3>
              <div className="flex flex-wrap gap-4 items-center">
                <Button variant="primary">Default State</Button>
                <Button variant="primary" loading>Loading State</Button>
                <Button variant="primary" disabled>Disabled State</Button>
                <Button variant="secondary" loading>Loading Secondary</Button>
                <Button variant="secondary" disabled>Disabled Secondary</Button>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-semibold uppercase tracking-wider text-[var(--stone)] mb-3">
                Sizes (sm, md, lg)
              </h3>
              <div className="flex flex-wrap gap-4 items-center">
                <Button size="sm">Small Action</Button>
                <Button size="md">Medium (Default)</Button>
                <Button size="lg">Large Hero Button</Button>
              </div>
            </div>
          </Card>
        </section>

        {/* Section 3: Input Component States */}
        <section className="space-y-4">
          <h2 className="text-xl font-serif text-[var(--forest)]">3. Input Component States</h2>
          <Card>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Input
                label="Sensor Reading with Unit"
                unit="mS/cm"
                defaultValue="1.45"
                helperText="Optimal band: 1.2 – 1.8 mS/cm"
              />

              <Input
                label="Default Text Input"
                placeholder="e.g. Bay 3 Lettuce Channel"
                helperText="Operator or location reference"
              />

              <Input
                label="Missing / Required Field"
                missing
                placeholder="Enter reservoir pH"
                helperText="Field is mandatory for nutrition classification"
              />

              <Input
                label="Error State with Validation Message"
                defaultValue="14.2"
                unit="pH"
                error="Exceeds biological limit (0.0 – 14.0 allowed)"
              />

              <Input
                label="Disabled Sensor Input"
                defaultValue="21.5"
                unit="°C"
                disabled
                helperText="Read-only live sensor feed"
              />

              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-semibold text-[var(--bark)] uppercase tracking-wide">
                  Select Dropdown State
                </label>
                <select className="w-full rounded-[var(--radius-md)] bg-[var(--surface)] px-3.5 py-2 text-sm text-[var(--bark)] border border-[var(--border-strong)] shadow-[var(--shadow-warm-sm)] outline-none focus:border-[var(--moss)] focus:ring-2 focus:ring-[var(--moss)]">
                  <option>Lettuce (Lactuca sativa)</option>
                  <option>Spinach (Spinacia oleracea)</option>
                  <option>Tomato (Solanum lycopersicum)</option>
                </select>
                <span className="text-xs text-[var(--stone)]">Crop cultivar selector</span>
              </div>
            </div>
          </Card>
        </section>

        {/* Section 4: Card Component States */}
        <section className="space-y-4">
          <h2 className="text-xl font-serif text-[var(--forest)]">4. Card Component States</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card variant="default">
              <h4 className="font-semibold text-[var(--forest)] mb-1">Default Surface Card</h4>
              <p className="text-xs text-[var(--stone)]">
                Base container with warm shadow and cream border.
              </p>
            </Card>

            <Card variant="default" interactive>
              <h4 className="font-semibold text-[var(--forest)] mb-1">Interactive Card</h4>
              <p className="text-xs text-[var(--stone)]">
                Hover to lift 3px with shadow expansion.
              </p>
            </Card>

            <Card variant="result">
              <h4 className="font-semibold text-[var(--moss)] mb-1">Result Card</h4>
              <p className="text-xs text-[var(--stone)]">
                Double border highlighting validated inference output.
              </p>
            </Card>

            <Card variant="warning">
              <h4 className="font-semibold text-[var(--warning)] mb-1">Warning Card</h4>
              <p className="text-xs text-[var(--stone)]">
                Left accent band for telemetry warnings and drift.
              </p>
            </Card>

            <Card variant="muted-uncertain">
              <h4 className="font-semibold text-[var(--bark)] mb-1">Muted / Uncertain</h4>
              <p className="text-xs text-[var(--stone)]">
                Reduced opacity for unassessed or low-confidence leaves.
              </p>
            </Card>

            <Card skeleton className="col-span-1 sm:col-span-2 lg:col-span-3" />
          </div>
        </section>

        {/* Section 5: Badge / Chip Variants */}
        <section className="space-y-4">
          <h2 className="text-xl font-serif text-[var(--forest)]">5. Badge & Chip Variants</h2>
          <Card className="space-y-4">
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--stone)] mb-2">
                Provenance Sources
              </h3>
              <div className="flex flex-wrap gap-2">
                <Badge variant="live" pulse>Live Monitor (Fresh)</Badge>
                <Badge variant="manual">Manual Entry</Badge>
                <Badge variant="upload">Image Upload</Badge>
                <Badge variant="imported">Imported Batch</Badge>
                <Badge variant="mock">Mock Device</Badge>
              </div>
            </div>

            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--stone)] mb-2">
                Model & Quality Metadata
              </h3>
              <div className="flex flex-wrap gap-2">
                <Badge variant="gemini">Gemini Provider</Badge>
                <Badge variant="rules">Rule Engine Fallback</Badge>
                <Badge variant="experimental">Experimental Model</Badge>
                <Badge variant="historical">Historical Record</Badge>
                <Badge variant="stale">Stale Telemetry (&gt;60s)</Badge>
              </div>
            </div>
          </Card>
        </section>

        {/* Section 6: Toast Notifications */}
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-serif text-[var(--forest)]">6. Toast Component</h2>
            <div className="flex gap-2">
              <Button size="sm" variant="primary" onClick={() => addToast("success")}>Trigger Success</Button>
              <Button size="sm" variant="secondary" onClick={() => addToast("warning")}>Trigger Warning</Button>
              <Button size="sm" variant="destructive" onClick={() => addToast("error")}>Trigger Error</Button>
            </div>
          </div>

          <div className="space-y-3">
            {activeToasts.map((toast) => (
              <Toast
                key={toast.id}
                variant={toast.variant}
                title={toast.title}
                message={toast.message}
                onDismiss={() => removeToast(toast.id)}
              />
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
