"use client";

import React, { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { Button } from "../../components/ui/Button";
import { Card } from "../../components/ui/Card";
import { HistoryFilterBar, HistoryTable } from "../../components/history";
import { apiClient } from "../../lib/api/client";
import { HistoryQueryParams, RecordListResponse, RecordRead } from "../../lib/api/types";

const PAGE_SIZE = 10;

export default function HistoryPage() {
  const [filters, setFilters] = useState<HistoryQueryParams>({
    limit: PAGE_SIZE,
    offset: 0,
  });

  const [records, setRecords] = useState<RecordRead[]>([]);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchRecords = useCallback(async (currentFilters: HistoryQueryParams) => {
    setLoading(true);
    // Explicitly reset error; never show stale cached rows as current when error occurs
    setError(null);

    try {
      const response: RecordListResponse = await apiClient.getHistory(currentFilters);
      setRecords(response.records || []);
      setTotalCount(response.total ?? (response.records ? response.records.length : 0));
      setError(null);
    } catch (err: unknown) {
      // Upon error, clear records so stale rows are never displayed as current (DESIGN S4)
      setRecords([]);
      setTotalCount(0);
      const message =
        err instanceof Error ? err.message : "Unable to load historical records from server.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRecords(filters);
  }, [fetchRecords, filters]);

  const handleFilterChange = (newFilters: HistoryQueryParams) => {
    setFilters(newFilters);
  };

  const handleResetFilters = () => {
    setFilters({
      limit: PAGE_SIZE,
      offset: 0,
    });
  };

  const handleRetry = () => {
    fetchRecords(filters);
  };

  // Pagination navigation
  const currentOffset = filters.offset || 0;
  const currentLimit = filters.limit || PAGE_SIZE;
  const hasPrevious = currentOffset > 0;
  const hasNext = currentOffset + currentLimit < totalCount;

  const handlePreviousPage = () => {
    setFilters((prev) => ({
      ...prev,
      offset: Math.max(0, currentOffset - currentLimit),
    }));
  };

  const handleNextPage = () => {
    setFilters((prev) => ({
      ...prev,
      offset: currentOffset + currentLimit,
    }));
  };

  const isFiltered = Boolean(
    filters.crop || filters.mode || filters.source || filters.start_date || filters.end_date
  );

  return (
    <div className="space-y-6 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {/* Header */}
      <div className="border-b border-[var(--border)] pb-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-[var(--bark)] tracking-tight">
              Inspection History &amp; Audit Log
            </h1>
            <p className="text-sm text-[var(--stone)] mt-1">
              Browse persisted plant health observations with true capture timestamps and audit provenance.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <a
              href={apiClient.getHistoryExportUrl(filters)}
              download="history_export.csv"
              className="inline-flex"
              aria-label="Export filtered inspection history to CSV file"
            >
              <Button variant="secondary" size="sm" type="button">
                <svg
                  className="w-4 h-4 mr-1.5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                  />
                </svg>
                Export CSV
              </Button>
            </a>

            <Link href="/analyze">
              <Button variant="primary" size="sm">
                <svg
                  className="w-4 h-4 mr-1.5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 4v16m8-8H4"
                  />
                </svg>
                New Inspection
              </Button>
            </Link>
          </div>
        </div>
      </div>

      {/* Filter Bar */}
      <HistoryFilterBar
        filters={filters}
        onFilterChange={handleFilterChange}
        onReset={handleResetFilters}
      />

      {/* Main Content Area */}
      {/* 1. Loading State (Table & Card Skeleton per DESIGN S4) */}
      {loading && (
        <div data-testid="history-loading-skeleton" className="space-y-3">
          {/* Mobile skeletons */}
          <div className="md:hidden space-y-3">
            {[1, 2, 3].map((i) => (
              <Card key={i} skeleton className="h-36" />
            ))}
          </div>

          {/* Desktop table skeleton */}
          <div className="hidden md:block rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow-warm-sm)] animate-pulse">
            <div className="h-5 w-1/4 bg-[var(--sand)] rounded-[var(--radius-sm)] mb-6" />
            <div className="space-y-4">
              {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="h-10 w-full bg-[var(--sand)]/40 rounded-[var(--radius-sm)]" />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* 2. Error State (Error card + Retry; never show stale cached rows per DESIGN S4) */}
      {!loading && error && (
        <Card
          variant="warning"
          className="p-6 border-l-4 border-[var(--error)] bg-[#FFFDF7] shadow-[var(--shadow-warm-sm)]"
          data-testid="history-error-card"
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
                Failed to Load History Records
              </h3>
              <p className="mt-1 text-sm text-[var(--stone)] leading-relaxed">
                {error}
              </p>
              <p className="mt-2 text-xs text-[var(--stone)]/80">
                To guarantee audit integrity, stale cached records are never displayed.
              </p>

              <div className="mt-4">
                <Button variant="secondary" size="sm" onClick={handleRetry}>
                  <svg
                    className="w-4 h-4 mr-1.5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                    />
                  </svg>
                  Retry Loading
                </Button>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* 3. Empty State ("No saved records yet" + CTA to Analyze per DESIGN S4) */}
      {!loading && !error && records.length === 0 && (
        <Card
          className="p-10 text-center bg-[var(--surface)] border border-[var(--border)] shadow-[var(--shadow-warm-sm)]"
          data-testid="history-empty-state"
        >
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-[var(--sand)]/60 text-[var(--stone)] flex items-center justify-center">
            <svg
              className="w-8 h-8"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
              />
            </svg>
          </div>

          <h3 className="text-lg font-bold text-[var(--bark)] mb-1">
            {isFiltered ? "No Matching Records Found" : "No saved records yet"}
          </h3>

          <p className="text-sm text-[var(--stone)] max-w-md mx-auto mb-6">
            {isFiltered
              ? "No historical inspection records match the current filter criteria. Try adjusting or resetting your filters."
              : "No diagnostic or nutrient analysis records have been saved yet. Analyze an image or enter sensor readings to create your first record."}
          </p>

          <div className="flex justify-center gap-3">
            {isFiltered ? (
              <Button variant="secondary" size="sm" onClick={handleResetFilters}>
                Reset Filters
              </Button>
            ) : (
              <Link href="/analyze">
                <Button variant="primary" size="md">
                  Analyze Plant or Sensors
                </Button>
              </Link>
            )}
          </div>
        </Card>
      )}

      {/* 4. Populated State: Table & Cards + Pagination */}
      {!loading && !error && records.length > 0 && (
        <div className="space-y-4">
          <HistoryTable records={records} />

          {/* Pagination Controls */}
          <div className="flex flex-wrap items-center justify-between gap-3 pt-2 text-xs text-[var(--stone)]">
            <div>
              Showing{" "}
              <span className="font-semibold text-[var(--bark)]">
                {currentOffset + 1}
              </span>{" "}
              to{" "}
              <span className="font-semibold text-[var(--bark)]">
                {Math.min(currentOffset + currentLimit, totalCount)}
              </span>{" "}
              of <span className="font-semibold text-[var(--bark)]">{totalCount}</span> records
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                size="sm"
                onClick={handlePreviousPage}
                disabled={!hasPrevious}
                aria-label="Go to previous page of records"
              >
                &larr; Previous
              </Button>

              <span className="px-2 font-mono text-[11px]">
                Page {Math.floor(currentOffset / currentLimit) + 1} of{" "}
                {Math.max(1, Math.ceil(totalCount / currentLimit))}
              </span>

              <Button
                variant="secondary"
                size="sm"
                onClick={handleNextPage}
                disabled={!hasNext}
                aria-label="Go to next page of records"
              >
                Next &rarr;
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
