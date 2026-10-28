"use client";

import React, { useState } from "react";
import { Button } from "../ui/Button";
import { Toast } from "../ui/Toast";
import { apiClient } from "../../lib/api/client";
import { RecordCreate, RecordRead } from "../../lib/api/types";

export interface SaveRecordButtonProps {
  recordData: RecordCreate;
  onSaveSuccess?: (savedRecord: RecordRead) => void;
  onSaveError?: (error: Error) => void;
  className?: string;
}

export const SaveRecordButton: React.FC<SaveRecordButtonProps> = ({
  recordData,
  onSaveSuccess,
  onSaveError,
  className = "",
}) => {
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [savedRecord, setSavedRecord] = useState<RecordRead | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [toastMessage, setToastMessage] = useState<{
    variant: "success" | "error";
    title: string;
    message: string;
  } | null>(null);

  const handleSave = async () => {
    // Acceptance Criterion 2: Double-click does not create two records (isSaving guard)
    if (isSaving || savedRecord) {
      return;
    }

    setIsSaving(true);
    setSaveError(null);

    try {
      const response: RecordRead = await apiClient.createRecord(recordData);

      // Successfully persisted
      setSavedRecord(response);
      setSaveError(null);
      setToastMessage({
        variant: "success",
        title: "Inspection Record Saved",
        message: `Saved with ID: ${response.id}. Audit provenance recorded.`,
      });

      onSaveSuccess?.(response);
    } catch (err: unknown) {
      // Acceptance Criterion 1: Failed save never shows success
      setSavedRecord(null);
      const errorMessage =
        err instanceof Error ? err.message : "Database error: Not saved.";
      setSaveError(errorMessage);

      setToastMessage({
        variant: "error",
        title: "Not saved",
        message: errorMessage,
      });

      onSaveError?.(err instanceof Error ? err : new Error(errorMessage));
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className={`inline-flex flex-col gap-2 ${className}`} data-testid="save-record-container">
      <div className="flex items-center gap-3">
        <Button
          variant={saveError ? "destructive" : "primary"}
          size="md"
          onClick={handleSave}
          disabled={isSaving || Boolean(savedRecord)}
          loading={isSaving}
          aria-label={
            savedRecord
              ? "Record already saved"
              : isSaving
              ? "Saving inspection record..."
              : "Save inspection record to history"
          }
          data-testid="save-record-button"
        >
          {savedRecord ? (
            <span className="flex items-center gap-1.5">
              <svg className="w-4 h-4 text-[#FFFFFF]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
              </svg>
              Record Saved
            </span>
          ) : isSaving ? (
            "Saving Record..."
          ) : (
            <span className="flex items-center gap-1.5">
              <svg className="w-4 h-4 mr-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
              </svg>
              Save Record
            </span>
          )}
        </Button>

        {/* Clear 'Not saved' error state display */}
        {saveError && !isSaving && (
          <span
            className="text-xs text-[var(--error)] font-bold flex items-center gap-1"
            role="alert"
            data-testid="not-saved-indicator"
          >
            <span aria-hidden="true">&#x2715;</span> Not saved
          </span>
        )}
      </div>

      {/* Acceptance Criterion 3: Toast announces via aria-live */}
      {toastMessage && (
        <div className="fixed bottom-4 right-4 z-50">
          <Toast
            variant={toastMessage.variant}
            title={toastMessage.title}
            message={toastMessage.message}
            onDismiss={() => setToastMessage(null)}
          />
        </div>
      )}
    </div>
  );
};
