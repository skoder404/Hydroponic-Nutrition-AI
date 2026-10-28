"use client";

import React, { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";
import { UncertaintyBadge } from "../../../components/ui/UncertaintyBadge";
import { apiClient } from "../../../lib/api/client";
import { RecordRead } from "../../../lib/api/types";

function RecordDetailContent() {
  const params = useParams();
  const recordId = params?.id as string;

  const [record, setRecord] = useState<RecordRead | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [notFound, setNotFound] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  const fetchRecord = useCallback(async () => {
    if (!recordId) return;

    setLoading(true);
    setNotFound(false);
    setError(null);

    try {
      const data: RecordRead = await apiClient.getRecord(recordId);
      setRecord(data);
    } catch (err: unknown) {
      const errStr = err instanceof Error ? err.message : String(err);
      if (errStr.includes("404") || errStr.toLowerCase().includes("not found")) {
        setNotFound(true);
      } else {
        setError(errStr);
      }
    } finally {
      setLoading(false);
    }
  }, [recordId]);

  useEffect(() => {
    fetchRecord();
  }, [fetchRecord]);

  const handleCopyJson = () => {
    if (!record) return;
    navigator.clipboard.writeText(JSON.stringify(record, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Loading State
  if (loading) {
    return (
      <div className="space-y-6 max-w-5xl mx-auto px-4 sm:px-6 py-6 text-left" data-testid="detail-loading-skeleton">
        <div className="h-6 w-32 bg-[var(--sand)] rounded-[var(--radius-sm)] animate-pulse" />
        <Card skeleton className="h-44" />
        <Card skeleton className="h-64" />
      </div>
    );
  }

  // Acceptance Criterion 3: Not-found state works
  if (notFound) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-16 text-center space-y-4" data-testid="record-not-found-state">
        <div className="w-16 h-16 mx-auto rounded-full bg-[var(--sand)]/80 text-[var(--stone)] flex items-center justify-center">
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>

        <h1 className="text-2xl font-bold text-[var(--bark)]">Record Not Found</h1>
        <p className="text-sm text-[var(--stone)] max-w-md mx-auto">
          No historical inspection record matching identifier <code className="font-mono text-xs bg-[var(--sand)] px-1.5 py-0.5 rounded">{recordId}</code> exists in PostgreSQL history.
        </p>

        <div className="pt-2">
          <Link href="/history">
            <Button variant="primary" size="md">
              &larr; Back to History
            </Button>
          </Link>
        </div>
      </div>
    );
  }

  // Server error
  if (error || !record) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-16 text-center space-y-4">
        <Card variant="warning" className="p-6 text-left">
          <h2 className="text-base font-bold text-[var(--bark)] mb-1">Failed to Load Inspection Record</h2>
          <p className="text-sm text-[var(--stone)] mb-4">{error || "Record could not be retrieved."}</p>
          <div className="flex gap-2">
            <Button variant="secondary" size="sm" onClick={fetchRecord}>
              Retry
            </Button>
            <Link href="/history">
              <Button variant="ghost" size="sm">
                Back to History
              </Button>
            </Link>
          </div>
        </Card>
      </div>
    );
  }

  // Extract timestamps with timezone (Acceptance Criterion 2)
  const observedDate = new Date(record.observed_at);
  const observedFormatted = observedDate.toLocaleString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const tzString = record.observed_tz || "UTC";

  // Check predictions & image availability
  const mainPrediction = Array.isArray(record.predictions)
    ? record.predictions[0]
    : record.predictions;

  const imageSource = record.inputs?.image_url as string | undefined;
  const hasImage = Boolean(imageSource);

  return (
    <div className="space-y-6 max-w-5xl mx-auto px-4 sm:px-6 py-6 text-left" data-testid="record-detail-container">
      {/* Back button & Action bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] pb-4">
        <Link
          href="/history"
          className="text-xs font-semibold text-[var(--stone)] hover:text-[var(--bark)] inline-flex items-center gap-1 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--moss)] rounded"
        >
          &larr; Back to Inspection History
        </Link>

        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={handleCopyJson}
            aria-label="Copy full record JSON payload to clipboard"
          >
            {copied ? "JSON Copied!" : "Copy JSON"}
          </Button>

          <Link href="/analyze">
            <Button variant="primary" size="sm">
              New Inspection
            </Button>
          </Link>
        </div>
      </div>

      {/* 1. Provenance Card */}
      <Card className="p-6 bg-[var(--surface)] border border-[var(--border)] shadow-[var(--shadow-warm-sm)] space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-[var(--stone)] block mb-1">
              Inspection Provenance
            </span>
            <h1 className="text-xl sm:text-2xl font-bold text-[var(--bark)] font-mono">
              {record.id}
            </h1>
          </div>

          <div className="flex items-center gap-1.5">
            {record.source === "live" ? (
              <Badge variant="live" pulse>
                Live Capture
              </Badge>
            ) : record.source === "manual" ? (
              <Badge variant="manual">Manual Entry</Badge>
            ) : (
              <Badge variant="imported">Imported Dataset</Badge>
            )}
            <Badge variant="historical">Historical</Badge>
          </div>
        </div>

        {/* Acceptance Criterion 2: Timestamps display with timezone */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-3 border-t border-[var(--border)]/60 text-xs">
          <div>
            <span className="text-[11px] uppercase tracking-wider text-[var(--stone)] font-semibold block">
              Observed Time &amp; TZ
            </span>
            <span className="font-semibold text-[var(--bark)] block mt-0.5">
              {observedFormatted}
            </span>
            <span className="font-mono text-[11px] text-[var(--stone)] block">
              Timezone: {tzString}
            </span>
          </div>

          <div>
            <span className="text-[11px] uppercase tracking-wider text-[var(--stone)] font-semibold block">
              Crop Cultivar
            </span>
            <span className="font-semibold text-[var(--bark)] capitalize block mt-0.5">
              {record.crop}
            </span>
          </div>

          <div>
            <span className="text-[11px] uppercase tracking-wider text-[var(--stone)] font-semibold block">
              Modality Mode
            </span>
            <span className="font-mono text-[11px] uppercase px-2 py-0.5 rounded bg-[var(--sand)] text-[var(--bark)] border border-[var(--border)] inline-block mt-0.5">
              {record.mode}
            </span>
          </div>

          <div>
            <span className="text-[11px] uppercase tracking-wider text-[var(--stone)] font-semibold block">
              Server Persisted UTC
            </span>
            <span className="font-mono text-[11px] text-[var(--stone)] block mt-0.5">
              {new Date(record.created_at).toISOString()}
            </span>
          </div>
        </div>

        {record.notes && (
          <div className="pt-2 text-xs text-[var(--bark)]">
            <span className="font-bold text-[var(--stone)]">Operator Notes:</span>{" "}
            <span>{record.notes}</span>
          </div>
        )}
      </Card>

      {/* 2. Inputs & Imagery / Metadata Only Handling (Acceptance Criterion 1) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
        {/* Foliar Image or Acceptance Criterion 1 (Missing image shows metadata only) */}
        <Card className="p-5 bg-[var(--surface)] border border-[var(--border)] shadow-[var(--shadow-warm-sm)] space-y-3">
          <h3 className="text-sm font-bold text-[var(--bark)]">Foliar Input Asset</h3>

          {hasImage ? (
            <div className="relative w-full aspect-4/3 rounded-[var(--radius-md)] overflow-hidden border border-[var(--border)] bg-[var(--sand)]/30">
              <img
                src={imageSource}
                alt="Original hydroponic leaf input"
                className="w-full h-full object-contain"
              />
            </div>
          ) : (
            /* Acceptance Criterion 1: Missing image shows metadata only */
            <div
              className="p-5 bg-[var(--sand)]/30 border border-dashed border-[var(--border-strong)] rounded-[var(--radius-md)] text-xs text-[var(--stone)] space-y-2"
              data-testid="missing-image-metadata-only"
            >
              <div className="flex items-center gap-2 font-bold text-[var(--bark)]">
                <svg className="w-4 h-4 text-[var(--stone)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <span>Original image no longer available</span>
              </div>
              <p className="leading-relaxed">
                Image binary was not retained or expired per privacy retention policy. Displaying recorded metadata and model provenance only.
              </p>
              <div className="pt-2 border-t border-[var(--border)]/50 font-mono text-[11px] space-y-1">
                <div>Input Type: {record.mode === "image" || record.mode === "both" ? "Foliar RGB capture" : "Sensor telemetry only"}</div>
                <div>Status: Metadata Preserved (No Image Binary)</div>
              </div>
            </div>
          )}
        </Card>

        {/* Sensor Inputs */}
        <Card className="p-5 bg-[var(--surface)] border border-[var(--border)] shadow-[var(--shadow-warm-sm)] space-y-3">
          <h3 className="text-sm font-bold text-[var(--bark)]">Telemetry Input Values</h3>
          {record.inputs && Object.keys(record.inputs).length > 0 ? (
            <div className="space-y-1.5 font-mono text-xs">
              {Object.entries(record.inputs).map(([k, v]) => (
                <div
                  key={k}
                  className="flex items-center justify-between p-2 rounded bg-[var(--sand)]/40 border border-[var(--border)]/60"
                >
                  <span className="font-semibold text-[var(--stone)]">{k}</span>
                  <span className="text-[var(--bark)] font-bold">{String(v)}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-[var(--stone)] italic">No raw sensor telemetry was recorded for this entry.</p>
          )}
        </Card>
      </div>

      {/* 3. Predictions Display */}
      {mainPrediction && (
        <Card className="p-6 bg-[var(--surface)] border border-[var(--border)] shadow-[var(--shadow-warm-sm)] space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] pb-3">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-[var(--stone)] block">
                Model Classification Outcome
              </span>
              <h2 className="text-lg font-bold text-[var(--bark)]">
                {mainPrediction.display_label || mainPrediction.predicted_class}
              </h2>
            </div>

            <UncertaintyBadge
              score={mainPrediction.confidence}
              isUncertain={mainPrediction.is_uncertain}
            />
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="p-2.5 rounded bg-[var(--sand)]/40 border border-[var(--border)]">
              <span className="text-[11px] uppercase tracking-wider text-[var(--stone)] block font-semibold">
                Confidence
              </span>
              <span className="font-mono font-bold text-sm text-[var(--forest)]">
                {(mainPrediction.confidence * 100).toFixed(1)}%
              </span>
            </div>

            <div className="p-2.5 rounded bg-[var(--sand)]/40 border border-[var(--border)]">
              <span className="text-[11px] uppercase tracking-wider text-[var(--stone)] block font-semibold">
                Model Name
              </span>
              <span className="font-mono text-xs text-[var(--bark)] truncate block">
                {mainPrediction.model_name || "Custom CNN"}
              </span>
            </div>

            <div className="p-2.5 rounded bg-[var(--sand)]/40 border border-[var(--border)]">
              <span className="text-[11px] uppercase tracking-wider text-[var(--stone)] block font-semibold">
                Version
              </span>
              <span className="font-mono text-xs text-[var(--bark)] block">
                {mainPrediction.model_version || "v1.0.0"}
              </span>
            </div>

            <div className="p-2.5 rounded bg-[var(--sand)]/40 border border-[var(--border)]">
              <span className="text-[11px] uppercase tracking-wider text-[var(--stone)] block font-semibold">
                Inference Time
              </span>
              <span className="font-mono text-xs text-[var(--bark)] block">
                {mainPrediction.inference_time_ms ? `${mainPrediction.inference_time_ms.toFixed(0)} ms` : "Recorded"}
              </span>
            </div>
          </div>
        </Card>
      )}

      {/* 4. Recommendation & Evidence */}
      {record.recommendations && (
        <Card className="p-6 bg-[var(--surface)] border border-[var(--border)] shadow-[var(--shadow-warm-sm)] space-y-3">
          <div className="flex items-center justify-between border-b border-[var(--border)] pb-2">
            <h3 className="text-base font-bold text-[var(--bark)]">Stored Agronomic Guidance</h3>
            <Badge variant={record.recommendations.source === "gemini" ? "gemini" : "rules"}>
              Source: {record.recommendations.source === "gemini" ? "Gemini" : "Rules"}
            </Badge>
          </div>
          <p className="text-sm text-[var(--bark)] leading-relaxed">
            {record.recommendations.summary}
          </p>
        </Card>
      )}
    </div>
  );
}

export default function RecordDetailPage() {
  return (
    <React.Suspense
      fallback={
        <div className="p-8 text-center text-sm text-[var(--stone)] flex items-center justify-center space-x-2">
          <div className="w-4 h-4 border-2 border-[var(--leaf)] border-t-transparent rounded-full animate-spin" />
          <span>Loading record...</span>
        </div>
      }
    >
      <RecordDetailContent />
    </React.Suspense>
  );
}
