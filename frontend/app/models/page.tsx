"use client";

import React, { useCallback, useEffect, useState } from "react";
import { Button } from "../../components/ui/Button";
import { Card } from "../../components/ui/Card";
import {
  DatasetsView,
  LimitationsView,
  MetricsDrawer,
  ModelsTable,
} from "../../components/models";
import { apiClient } from "../../lib/api/client";
import { ModelSummary } from "../../lib/api/types";

type ActiveTab = "models" | "datasets" | "limitations";

export default function ModelsPage() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("models");
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelSummary | null>(null);
  const [drawerOpen, setDrawerOpen] = useState<boolean>(false);

  const fetchModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.getModels();
      setModels(response.models || []);
      setError(null);
    } catch (err: unknown) {
      setModels([]);
      const message =
        err instanceof Error ? err.message : "Failed to load registered models from registry.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const handleSelectModel = (model: ModelSummary) => {
    setSelectedModel(model);
    setDrawerOpen(true);
  };

  const handleCloseDrawer = () => {
    setDrawerOpen(false);
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 text-left">
      {/* Page Header */}
      <div className="border-b border-[var(--border)] pb-4">
        <h1 className="text-2xl sm:text-3xl font-bold text-[var(--bark)] tracking-tight">
          Models, Datasets &amp; Limitations
        </h1>
        <p className="text-sm text-[var(--stone)] mt-1">
          Reviewer-facing transparency portal: promoted models, dataset manifests, and mandatory Section 28 disclosures.
        </p>
      </div>

      {/* Responsive Navigation Tabs (Under 640px becomes a select; >= 640px buttons) */}
      <div className="flex items-center justify-between border-b border-[var(--border)] pb-1">
        {/* Mobile Dropdown Select (< 640px) */}
        <div className="sm:hidden w-full">
          <label htmlFor="models-tab-select" className="sr-only">
            Select portal view
          </label>
          <select
            id="models-tab-select"
            value={activeTab}
            onChange={(e) => setActiveTab(e.target.value as ActiveTab)}
            className="w-full text-xs font-semibold h-10 px-3 rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] text-[var(--bark)]"
          >
            <option value="models">Models Registry</option>
            <option value="datasets">Datasets &amp; Provenance</option>
            <option value="limitations">Known Limitations (Overview §28)</option>
          </select>
        </div>

        {/* Desktop Tab Buttons (>= 640px) */}
        <nav
          className="hidden sm:flex space-x-4"
          aria-label="Models portal navigation"
        >
          <button
            type="button"
            onClick={() => setActiveTab("models")}
            aria-current={activeTab === "models" ? "page" : undefined}
            className={`px-4 py-2 text-sm font-semibold border-b-2 transition-colors ${
              activeTab === "models"
                ? "border-[var(--forest)] text-[var(--forest)]"
                : "border-transparent text-[var(--stone)] hover:text-[var(--bark)] hover:border-[var(--border-strong)]"
            }`}
          >
            Models ({models.length})
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("datasets")}
            aria-current={activeTab === "datasets" ? "page" : undefined}
            className={`px-4 py-2 text-sm font-semibold border-b-2 transition-colors ${
              activeTab === "datasets"
                ? "border-[var(--forest)] text-[var(--forest)]"
                : "border-transparent text-[var(--stone)] hover:text-[var(--bark)] hover:border-[var(--border-strong)]"
            }`}
          >
            Datasets
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("limitations")}
            aria-current={activeTab === "limitations" ? "page" : undefined}
            className={`px-4 py-2 text-sm font-semibold border-b-2 transition-colors ${
              activeTab === "limitations"
                ? "border-[var(--forest)] text-[var(--forest)]"
                : "border-transparent text-[var(--stone)] hover:text-[var(--bark)] hover:border-[var(--border-strong)]"
            }`}
          >
            Limitations (Overview §28)
          </button>
        </nav>
      </div>

      {/* Main Tab Content */}
      {/* TAB 1: Models Registry */}
      {activeTab === "models" && (
        <div className="space-y-4">
          {/* Loading Skeleton */}
          {loading && (
            <div data-testid="models-loading-skeleton" className="space-y-3">
              <div className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow-warm-sm)] animate-pulse">
                <div className="h-5 w-1/4 bg-[var(--sand)] rounded-[var(--radius-sm)] mb-6" />
                <div className="space-y-4">
                  {[1, 2, 3, 4].map((i) => (
                    <div
                      key={i}
                      className="h-10 w-full bg-[var(--sand)]/40 rounded-[var(--radius-sm)]"
                    />
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Error State */}
          {!loading && error && (
            <Card
              variant="warning"
              className="p-6 border-l-4 border-[var(--error)] bg-[#FFFDF7] shadow-[var(--shadow-warm-sm)]"
              data-testid="models-error-card"
              aria-live="polite"
            >
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-full bg-[#FCE8E6] text-[var(--error)] flex items-center justify-center shrink-0">
                  <svg
                    className="w-6 h-6"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    aria-hidden="true"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                </div>

                <div className="flex-1">
                  <h3 className="text-base font-bold text-[var(--bark)]">
                    Model Registry Unreadable
                  </h3>
                  <p className="mt-1 text-sm text-[var(--stone)] leading-relaxed">
                    {error}
                  </p>
                  <div className="mt-4">
                    <Button variant="secondary" size="sm" onClick={fetchModels}>
                      Retry Loading
                    </Button>
                  </div>
                </div>
              </div>
            </Card>
          )}

          {/* Table View */}
          {!loading && !error && (
            <ModelsTable
              models={models}
              onSelectModel={handleSelectModel}
            />
          )}
        </div>
      )}

      {/* TAB 2: Datasets & Provenance */}
      {activeTab === "datasets" && <DatasetsView />}

      {/* TAB 3: Limitations (Overview §28) */}
      {activeTab === "limitations" && <LimitationsView />}

      {/* Metrics Slide-Over Drawer */}
      <MetricsDrawer
        model={selectedModel}
        isOpen={drawerOpen}
        onClose={handleCloseDrawer}
      />
    </div>
  );
}
