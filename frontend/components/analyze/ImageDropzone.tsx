"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui";

export const MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024; // 15 MB matching backend MAX_UPLOAD_BYTES
export const ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"];
export const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"];

export interface ImageDropzoneProps {
  onFileSelected: (file: File | null) => void;
  selectedFile?: File | null;
  disabled?: boolean;
  className?: string;
}

export const ImageDropzone: React.FC<ImageDropzoneProps> = ({
  onFileSelected,
  selectedFile = null,
  disabled = false,
  className = "",
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  // Sync preview URL with selectedFile
  useEffect(() => {
    if (selectedFile) {
      const url = URL.createObjectURL(selectedFile);
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setPreviewUrl(null);
    }
  }, [selectedFile]);

  // Client-side validation function returning specific error message
  const validateFile = useCallback((file: File): string | null => {
    // 1. Check MIME type and file extension
    const extension = "." + file.name.split(".").pop()?.toLowerCase();
    const isMimeValid = ALLOWED_MIME_TYPES.includes(file.type.toLowerCase());
    const isExtValid = ALLOWED_EXTENSIONS.includes(extension);

    if (!isMimeValid && !isExtValid) {
      const reportedType = file.type || extension || "unknown";
      return `Invalid file format: "${reportedType}". Only JPEG, PNG, and WebP images are supported.`;
    }

    // 2. Check maximum file size
    if (file.size > MAX_FILE_SIZE_BYTES) {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
      return `File size (${sizeMb} MB) exceeds maximum allowed limit of 15 MB.`;
    }

    return null;
  }, []);

  const handleIncomingFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0 || disabled) return;

      const file = files[0];
      const validationError = validateFile(file);

      if (validationError) {
        setErrorMessage(validationError);
        onFileSelected(null);
      } else {
        setErrorMessage(null);
        onFileSelected(file);
      }
    },
    [disabled, onFileSelected, validateFile]
  );

  // Drag-and-drop event handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    if (!disabled && e.dataTransfer.files) {
      handleIncomingFiles(e.dataTransfer.files);
    }
  };

  // Keyboard accessibility: Enter or Space triggers file input dialog
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (disabled) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInputRef.current?.click();
    }
  };

  const handleRemove = () => {
    setErrorMessage(null);
    onFileSelected(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (cameraInputRef.current) cameraInputRef.current.value = "";
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className={`w-full space-y-3 text-left ${className}`}>
      {/* Hidden standard file picker */}
      <input
        ref={fileInputRef}
        type="file"
        accept={ALLOWED_MIME_TYPES.join(",")}
        onChange={(e) => handleIncomingFiles(e.target.files)}
        disabled={disabled}
        className="sr-only"
        aria-hidden="true"
        tabIndex={-1}
        id="image-dropzone-file-input"
      />

      {/* Hidden mobile camera capture input with capture="environment" */}
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        capture="environment"
        onChange={(e) => handleIncomingFiles(e.target.files)}
        disabled={disabled}
        className="sr-only"
        aria-hidden="true"
        tabIndex={-1}
        id="image-dropzone-camera-input"
      />

      {/* Dropzone Container */}
      {!selectedFile ? (
        <div
          role="button"
          tabIndex={disabled ? -1 : 0}
          aria-label="Upload leaf image. Drag and drop file or press Enter to browse"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onKeyDown={handleKeyDown}
          onClick={() => !disabled && fileInputRef.current?.click()}
          className={`relative border-2 border-dashed rounded-[var(--radius-lg)] p-8 text-center transition-all duration-150 cursor-pointer outline-none select-none ${
            isDragOver
              ? "border-[var(--forest)] bg-[#F2F7F3] scale-[1.01]"
              : errorMessage
              ? "border-[var(--error)] bg-[#FDF5F5]"
              : "border-[var(--border-strong)] bg-[var(--surface)] hover:bg-[var(--sand)]/30 hover:border-[var(--moss)]"
          } focus-visible:ring-2 focus-visible:ring-[var(--moss)] focus-visible:ring-offset-2 ${
            disabled ? "opacity-50 cursor-not-allowed pointer-events-none" : ""
          }`}
        >
          <div className="flex flex-col items-center justify-center space-y-3">
            <span className="text-4xl block" aria-hidden="true">
              📷
            </span>
            <div>
              <p className="text-sm font-semibold text-[var(--forest)]">
                Drag &amp; drop plant image here, or{" "}
                <span className="text-[var(--moss)] underline">browse files</span>
              </p>
              <p className="text-xs text-[var(--stone)] mt-1">
                Supports JPEG, PNG, and WebP (up to 15 MB)
              </p>
            </div>

            {/* Mobile / Keyboard actions row */}
            <div
              className="flex flex-wrap items-center justify-center gap-2 pt-2"
              onClick={(e) => e.stopPropagation()}
            >
              <Button
                type="button"
                variant="secondary"
                size="sm"
                disabled={disabled}
                onClick={() => fileInputRef.current?.click()}
              >
                Choose File
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                disabled={disabled}
                onClick={() => cameraInputRef.current?.click()}
                className="gap-1.5"
              >
                <span>📱</span>
                <span>Take Photo</span>
              </Button>
            </div>
          </div>
        </div>
      ) : (
        /* Image Preview Box */
        <div className="relative border border-[var(--border)] rounded-[var(--radius-lg)] bg-[var(--surface)] p-4 shadow-[var(--shadow-warm-sm)]">
          <div className="flex flex-col sm:flex-row items-center gap-4">
            {previewUrl && (
              <div className="relative h-28 w-28 shrink-0 overflow-hidden rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--sand)]/40">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={previewUrl}
                  alt="Selected plant leaf preview"
                  className="h-full w-full object-cover"
                />
              </div>
            )}

            <div className="flex-1 min-w-0 space-y-1 text-center sm:text-left">
              <div className="flex items-center justify-center sm:justify-start gap-2">
                <span className="text-xs bg-[#EBF7EE] text-[var(--success)] px-2 py-0.5 rounded-[var(--radius-full)] font-semibold font-mono border border-[var(--success)]/30">
                  Ready
                </span>
                <span className="text-xs font-mono text-[var(--stone)]">
                  {formatFileSize(selectedFile.size)}
                </span>
              </div>
              <h4 className="text-sm font-semibold text-[var(--bark)] truncate" title={selectedFile.name}>
                {selectedFile.name}
              </h4>
              <p className="text-xs text-[var(--stone)]">
                Client pre-check passed. Autorotation and per-leaf isolation will execute on server.
              </p>
            </div>

            <div className="shrink-0 flex items-center gap-2">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={handleRemove}
                disabled={disabled}
                className="text-[var(--error)] hover:bg-[#FDF3F2]"
              >
                Remove
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Validation Error Message */}
      {errorMessage && (
        <div
          role="alert"
          aria-live="polite"
          className="rounded-[var(--radius-md)] bg-[#FDF3F2] border border-[var(--error)]/40 p-3 flex items-start gap-2.5 text-xs text-[var(--error)] font-medium"
        >
          <svg className="h-4 w-4 shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
              clipRule="evenodd"
            />
          </svg>
          <div className="flex-1">
            <p>{errorMessage}</p>
            <p className="text-[11px] text-[var(--stone)] mt-0.5 font-normal">
              Ensure you upload a clear JPEG, PNG, or WebP photo under 15 MB.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
