"use client";

import React from "react";
import { HistoryQueryParams } from "../../lib/api/types";

export interface HistoryFilterBarProps {
  filters: HistoryQueryParams;
  onFilterChange: (newFilters: HistoryQueryParams) => void;
  onReset: () => void;
  className?: string;
}

export const HistoryFilterBar: React.FC<HistoryFilterBarProps> = ({
  filters,
  onFilterChange,
  onReset,
  className = "",
}) => {
  const isFiltered = Boolean(
    filters.crop ||
    filters.mode ||
    filters.source ||
    filters.start_date ||
    filters.end_date
  );

  const handleSelectChange = (field: keyof HistoryQueryParams, value: string) => {
    onFilterChange({
      ...filters,
      offset: 0, // Reset to first page on filter change
      [field]: value === "all" ? undefined : value,
    });
  };

  const handleDateChange = (field: "start_date" | "end_date", value: string) => {
    onFilterChange({
      ...filters,
      offset: 0,
      [field]: value ? value : undefined,
    });
  };

  return (
    <div
      className={`p-4 bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius-lg)] shadow-[var(--shadow-warm-sm)] ${className}`}
      data-testid="history-filter-bar"
    >
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <svg
            className="w-4 h-4 text-[var(--stone)]"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
            />
          </svg>
          <span className="text-xs font-bold uppercase tracking-wider text-[var(--stone)]">
            Filter History
          </span>
        </div>

        {isFiltered && (
          <button
            type="button"
            onClick={onReset}
            className="text-xs font-semibold text-[var(--clay)] hover:text-[#8B4224] hover:underline transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--moss)] rounded px-1"
          >
            Reset Filters
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        {/* Crop Selector */}
        <div>
          <label
            htmlFor="history-filter-crop"
            className="block text-xs font-medium text-[var(--bark)] mb-1"
          >
            Crop
          </label>
          <select
            id="history-filter-crop"
            value={filters.crop || "all"}
            onChange={(e) => handleSelectChange("crop", e.target.value)}
            className="w-full text-xs h-9 px-2.5 rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] text-[var(--bark)] focus:border-[var(--forest)] focus:outline-none focus:ring-1 focus:ring-[var(--forest)]"
          >
            <option value="all">All Crops</option>
            <option value="lettuce">Lettuce</option>
            <option value="spinach">Spinach</option>
            <option value="tomato">Tomato</option>
          </select>
        </div>

        {/* Modality Selector */}
        <div>
          <label
            htmlFor="history-filter-mode"
            className="block text-xs font-medium text-[var(--bark)] mb-1"
          >
            Modality
          </label>
          <select
            id="history-filter-mode"
            value={filters.mode || "all"}
            onChange={(e) => handleSelectChange("mode", e.target.value)}
            className="w-full text-xs h-9 px-2.5 rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] text-[var(--bark)] focus:border-[var(--forest)] focus:outline-none focus:ring-1 focus:ring-[var(--forest)]"
          >
            <option value="all">All Modalities</option>
            <option value="sensor">Sensor Only</option>
            <option value="image">Image Only</option>
            <option value="both">Both (Multi-modal)</option>
          </select>
        </div>

        {/* Source Selector */}
        <div>
          <label
            htmlFor="history-filter-source"
            className="block text-xs font-medium text-[var(--bark)] mb-1"
          >
            Source
          </label>
          <select
            id="history-filter-source"
            value={filters.source || "all"}
            onChange={(e) => handleSelectChange("source", e.target.value)}
            className="w-full text-xs h-9 px-2.5 rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] text-[var(--bark)] focus:border-[var(--forest)] focus:outline-none focus:ring-1 focus:ring-[var(--forest)]"
          >
            <option value="all">All Sources</option>
            <option value="live">Live Telemetry</option>
            <option value="manual">Manual Entry / Upload</option>
            <option value="imported">Imported Dataset</option>
          </select>
        </div>

        {/* Start Date */}
        <div>
          <label
            htmlFor="history-filter-start-date"
            className="block text-xs font-medium text-[var(--bark)] mb-1"
          >
            Start Date
          </label>
          <input
            id="history-filter-start-date"
            type="date"
            value={filters.start_date || ""}
            onChange={(e) => handleDateChange("start_date", e.target.value)}
            className="w-full text-xs h-9 px-2.5 rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] text-[var(--bark)] focus:border-[var(--forest)] focus:outline-none focus:ring-1 focus:ring-[var(--forest)]"
          />
        </div>

        {/* End Date */}
        <div>
          <label
            htmlFor="history-filter-end-date"
            className="block text-xs font-medium text-[var(--bark)] mb-1"
          >
            End Date
          </label>
          <input
            id="history-filter-end-date"
            type="date"
            value={filters.end_date || ""}
            onChange={(e) => handleDateChange("end_date", e.target.value)}
            className="w-full text-xs h-9 px-2.5 rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] text-[var(--bark)] focus:border-[var(--forest)] focus:outline-none focus:ring-1 focus:ring-[var(--forest)]"
          />
        </div>
      </div>
    </div>
  );
};
