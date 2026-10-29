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

  // Live Camera Capture State
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [capturedBlob, setCapturedBlob] = useState<Blob | null>(null);
  const [capturedPreview, setCapturedPreview] = useState<string | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Sync preview URL with selectedFile
  useEffect(() => {
    if (selectedFile) {
      const url = URL.createObjectURL(selectedFile);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setPreviewUrl(null);
    }
  }, [selectedFile]);

  // Clean up camera stream tracks when unmounting or stopping camera
  const stopCameraStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setCameraStream(null);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  // Cleanup exclusively on component unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // Ensure video element receives camera stream and starts playing as soon as both are available
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !cameraStream) return;

    if (video.srcObject !== cameraStream) {
      video.srcObject = cameraStream;
    }
    video.muted = true;

    const playVideo = () => {
      video.play().catch((err) => {
        console.warn("Video play error:", err);
      });
    };

    video.addEventListener("loadedmetadata", playVideo);
    playVideo();

    return () => {
      video.removeEventListener("loadedmetadata", playVideo);
    };
  }, [cameraStream, isCameraActive]);

  // Callback ref for immediate assignment when <video> mounts
  const attachVideoRef = useCallback(
    (node: HTMLVideoElement | null) => {
      videoRef.current = node;
      if (node && streamRef.current) {
        if (node.srcObject !== streamRef.current) {
          node.srcObject = streamRef.current;
        }
        node.muted = true;
        node.play().catch(() => {});
      }
    },
    []
  );

  // Start live device camera stream
  const startCamera = async () => {
    setCameraError(null);
    setCapturedBlob(null);
    if (capturedPreview) {
      URL.revokeObjectURL(capturedPreview);
      setCapturedPreview(null);
    }
    setIsCameraActive(true);

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Camera API is not supported in this browser environment.");
      }

      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
      } catch {
        // Fallback for laptop / desktop webcams that don't support environment facingMode
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
      }

      streamRef.current = stream;
      setCameraStream(stream);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        videoRef.current.play().catch(() => {});
      }
    } catch (err: unknown) {
      console.warn("Unable to open device camera:", err);
      const msg =
        err instanceof Error
          ? err.message
          : "Unable to access camera. Please check camera permissions or upload an image file.";
      setCameraError(msg);
    }
  };

  // Close camera mode and reset
  const closeCamera = () => {
    stopCameraStream();
    setIsCameraActive(false);
    setCapturedBlob(null);
    if (capturedPreview) {
      URL.revokeObjectURL(capturedPreview);
      setCapturedPreview(null);
    }
    setCameraError(null);
  };

  // Freeze snapshot from active video stream
  const handleCapturePhoto = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, width, height);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        setCapturedBlob(blob);
        const url = URL.createObjectURL(blob);
        setCapturedPreview(url);
        // Pause video while reviewing captured photo
        video.pause();
      },
      "image/jpeg",
      0.95
    );
  };

  // Retake photo: discard snapshot and resume live video stream
  const handleRetake = () => {
    setCapturedBlob(null);
    if (capturedPreview) {
      URL.revokeObjectURL(capturedPreview);
      setCapturedPreview(null);
    }
    if (videoRef.current) {
      if (streamRef.current && videoRef.current.srcObject !== streamRef.current) {
        videoRef.current.srcObject = streamRef.current;
        videoRef.current.muted = true;
      }
      videoRef.current.play().catch(() => {});
    }
  };

  // OK: Accept photo, convert to File, pass to onFileSelected, and close camera
  const handleConfirmPhoto = () => {
    if (!capturedBlob) return;
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const capturedFile = new File([capturedBlob], `leaf_camera_${timestamp}.jpg`, {
      type: "image/jpeg",
      lastModified: Date.now(),
    });

    onFileSelected(capturedFile);
    closeCamera();
  };

  // Client-side validation function returning specific error message
  const validateFile = useCallback((file: File): string | null => {
    const extension = "." + file.name.split(".").pop()?.toLowerCase();
    const isMimeValid = ALLOWED_MIME_TYPES.includes(file.type.toLowerCase());
    const isExtValid = ALLOWED_EXTENSIONS.includes(extension);

    if (!isMimeValid && !isExtValid) {
      const reportedType = file.type || extension || "unknown";
      return `Invalid file format: "${reportedType}". Only JPEG, PNG, and WebP images are supported.`;
    }

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
    if (disabled || isCameraActive) return;
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

      {/* Hidden mobile camera capture fallback input */}
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

      {/* 1. Live Interactive Camera Viewport */}
      {isCameraActive ? (
        <div className="relative border-2 border-[var(--forest)] rounded-[var(--radius-lg)] bg-[#111111] overflow-hidden shadow-[var(--shadow-warm-md)]">
          {/* Camera Header */}
          <div className="flex items-center justify-between px-4 py-2.5 bg-[#000000]/80 backdrop-blur-xs border-b border-white/10 text-white z-10">
            <div className="flex items-center gap-2 text-xs font-semibold">
              <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
              <span>{capturedPreview ? "Photo Captured (Review)" : "Live Foliar Camera"}</span>
            </div>
            <button
              type="button"
              onClick={closeCamera}
              className="text-white/70 hover:text-white text-xs px-2 py-1 rounded bg-white/10 hover:bg-white/20 transition-colors"
            >
              ✕ Cancel
            </button>
          </div>

          {/* Viewport Area */}
          <div className="relative aspect-[4/3] sm:aspect-[16/9] w-full bg-black flex items-center justify-center overflow-hidden">
            {/* Live Stream Video */}
            <video
              ref={attachVideoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-contain transition-opacity duration-200 ${
                capturedPreview ? "hidden" : "block"
              }`}
            />

            {/* Captured Preview Image */}
            {capturedPreview && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={capturedPreview}
                alt="Captured leaf preview"
                className="w-full h-full object-contain"
              />
            )}

            {/* Framing Guide Reticle (only during live streaming) */}
            {!capturedPreview && !cameraError && (
              <div className="pointer-events-none absolute inset-6 sm:inset-10 border-2 border-dashed border-white/40 rounded-[var(--radius-lg)] flex items-center justify-center">
                <span className="text-[11px] font-medium text-white/75 bg-black/60 px-3 py-1 rounded-full backdrop-blur-xs">
                  Center plant leaf in frame
                </span>
              </div>
            )}

            {/* Camera Access Error Message */}
            {cameraError && (
              <div className="p-6 text-center text-white space-y-3 max-w-sm">
                <p className="text-sm font-semibold text-red-400">Camera Access Issue</p>
                <p className="text-xs text-white/80 leading-relaxed">{cameraError}</p>
                <div className="flex justify-center gap-2 pt-2">
                  <Button type="button" variant="secondary" size="sm" onClick={startCamera}>
                    Retry Camera
                  </Button>
                  <Button
                    type="button"
                    variant="primary"
                    size="sm"
                    onClick={() => {
                      closeCamera();
                      fileInputRef.current?.click();
                    }}
                  >
                    Choose File Instead
                  </Button>
                </div>
              </div>
            )}
          </div>

          {/* Camera Bottom Action Controls */}
          {!cameraError && (
            <div className="p-4 bg-[#000000]/90 border-t border-white/10 flex items-center justify-center gap-3">
              {!capturedPreview ? (
                /* Shutter Button */
                <Button
                  type="button"
                  variant="primary"
                  size="md"
                  onClick={handleCapturePhoto}
                  className="gap-2 bg-emerald-600 hover:bg-emerald-500 text-white font-bold px-6 shadow-lg"
                >
                  <span className="text-base">📸</span>
                  <span>Capture Photo</span>
                </Button>
              ) : (
                /* Retake vs OK Controls */
                <div className="flex items-center gap-3 w-full max-w-xs justify-center">
                  <Button
                    type="button"
                    variant="secondary"
                    size="md"
                    onClick={handleRetake}
                    className="flex-1 gap-1.5 font-bold"
                  >
                    <span>🔄</span>
                    <span>Retake</span>
                  </Button>
                  <Button
                    type="button"
                    variant="primary"
                    size="md"
                    onClick={handleConfirmPhoto}
                    className="flex-1 gap-1.5 bg-emerald-600 hover:bg-emerald-500 text-white font-bold shadow-md"
                  >
                    <span>✓</span>
                    <span>OK</span>
                  </Button>
                </div>
              )}
            </div>
          )}
        </div>
      ) : !selectedFile ? (
        /* 2. Standard Dropzone Container */
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

            {/* Actions row: Choose File vs Take Photo */}
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
                variant="primary"
                size="sm"
                disabled={disabled}
                onClick={startCamera}
                className="gap-1.5 bg-[var(--forest)] text-white hover:bg-[var(--moss)]"
              >
                <span>📷</span>
                <span>Take Photo</span>
              </Button>
            </div>
          </div>
        </div>
      ) : (
        /* 3. Image Preview Box when File is Selected */
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

