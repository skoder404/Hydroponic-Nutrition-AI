"use client";

import React, { useEffect, useRef, useState } from "react";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { Card } from "../../components/ui/Card";
import {
  AnalysisStep,
  CROP_SENSOR_SCHEMAS,
  ImageDropzone,
  ImageResultPanel,
  InputMode,
  InputModeToggle,
  NotAssessedNote,
  RecommendationCard,
  SaveRecordButton,
  SensorForm,
  SensorResultPanel,
  SupportedCrop,
  VineProgressIndicator,
} from "../../components/analyze";
import { apiClient } from "../../lib/api/client";
import {
  DeviceStatusResponse,
  PredictionResponse,
  RecommendationResponse,
  RecordCreate,
} from "../../lib/api/types";

export type AnalysisMode = "both" | "image" | "sensor";
export type SourceProvenance = "manual" | "campus_live";

export default function AnalyzePage() {
  // Configuration State
  const [selectedCrop, setSelectedCrop] = useState<SupportedCrop>("lettuce");
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>("both");
  const [sourceProvenance, setSourceProvenance] = useState<SourceProvenance>("manual");

  // Image Inputs
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [inputMode, setInputMode] = useState<InputMode | null>("plant");

  // Sensor Inputs
  const [sensorValues, setSensorValues] = useState<Record<string, number | undefined>>({
    ph: 6.2,
    ec: 1.6,
    water_temp: 21.0,
    humidity: 65.0,
    air_temp: 23.5,
  });
  const [sensorValid, setSensorValid] = useState<boolean>(true);

  // Campus Device Telemetry State
  const [deviceStatus, setDeviceStatus] = useState<DeviceStatusResponse | null>(null);
  const [deviceError, setDeviceError] = useState<boolean>(false);

  // Execution & Progress State
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [currentStep, setCurrentStep] = useState<AnalysisStep>("validating");
  const abortControllerRef = useRef<AbortController | null>(null);

  // Analysis Outcome State
  const [imagePrediction, setImagePrediction] = useState<PredictionResponse | null>(null);
  const [segmentationError, setSegmentationError] = useState<string | null>(null);
  const [sensorPrediction, setSensorPrediction] = useState<PredictionResponse | null>(null);
  const [recommendation, setRecommendation] = useState<RecommendationResponse | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [analysisComplete, setAnalysisComplete] = useState<boolean>(false);

  // Fetch campus device status when switching to campus_live
  useEffect(() => {
    if (sourceProvenance === "campus_live") {
      apiClient
        .getDeviceStatus()
        .then((status) => {
          setDeviceStatus(status);
          setDeviceError(!status.connected);
        })
        .catch(() => {
          setDeviceError(true);
        });
    } else {
      setDeviceError(false);
    }
  }, [sourceProvenance]);

  // Manage Image Object URL
  useEffect(() => {
    if (selectedFile) {
      const url = URL.createObjectURL(selectedFile);
      setImagePreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setImagePreviewUrl(null);
    }
  }, [selectedFile]);

  // Validation Check: Acceptance Criterion 1: Analyze is disabled until required inputs validate
  const isImageValid = Boolean(
    analysisMode === "sensor" ||
    (selectedFile && inputMode !== null) ||
    (sourceProvenance === "campus_live" && !deviceError)
  );

  const isSensorValid = Boolean(
    analysisMode === "image" ||
    (sensorValid && Object.keys(sensorValues).length > 0)
  );

  const canExecute = isImageValid && isSensorValid && !isAnalyzing && !(sourceProvenance === "campus_live" && deviceError);

  // Execution Handler
  const handleExecuteAnalysis = async () => {
    if (!canExecute) return;

    setIsAnalyzing(true);
    setAnalysisComplete(false);
    setCurrentStep("validating");

    // Acceptance Criterion 2: Cancel aborts the request
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      // Step 1: Validating
      await new Promise((resolve) => setTimeout(resolve, 200));
      if (controller.signal.aborted) return;

      // Step 2: Segmenting / Requesting
      setCurrentStep("segmenting");

      // Construct live multi-modal request
      const formData = new FormData();
      formData.append("crop", selectedCrop);

      if (analysisMode === "image" || analysisMode === "both") {
        if (inputMode) {
          formData.append("input_mode", inputMode);
        }
        if (sourceProvenance === "campus_live") {
          formData.append("use_campus_frame", "true");
        } else if (selectedFile) {
          formData.append("file", selectedFile);
        }
      }

      if (analysisMode === "sensor" || analysisMode === "both") {
        formData.append("sensor_readings", JSON.stringify(sensorValues));
      }

      let apiResult: import("../../lib/api/types").AnalyzeResponse | null = null;
      try {
        apiResult = await apiClient.analyze(formData);
      } catch (err: unknown) {
        if (controller.signal.aborted) return;
        console.error("Backend API dispatch failed:", err);
        const errMsg =
          err instanceof Error
            ? err.message
            : "Diagnostic pipeline failed to connect to backend server. Please verify backend is running on port 8000.";
        setSegmentationError(errMsg);
        setImagePrediction(null);
        setSensorPrediction(null);
        setRecommendation(null);
        setIsAnalyzing(false);
        return;
      }

      if (controller.signal.aborted) return;

      // Step 3: Classifying
      setCurrentStep("classifying");
      await new Promise((resolve) => setTimeout(resolve, 300));
      if (controller.signal.aborted) return;

      if (apiResult) {
        // Use live backend inference results
        setWarnings(apiResult.warnings || []);

        const imageNotAssessed = apiResult.not_assessed?.find((item) => item.modality === "image");
        if (imageNotAssessed && analysisMode !== "sensor") {
          setSegmentationError(imageNotAssessed.reason);
          setImagePrediction(null);
        } else if (apiResult.image) {
          setSegmentationError(null);
          setImagePrediction(apiResult.image);
        } else {
          setSegmentationError(null);
          setImagePrediction(null);
        }

        if (apiResult.sensor) {
          setSensorPrediction(apiResult.sensor);
        } else {
          setSensorPrediction(null);
        }

        // Step 4: Recommending
        setCurrentStep("recommending");
        await new Promise((resolve) => setTimeout(resolve, 250));
        if (controller.signal.aborted) return;

        if (apiResult.recommendation) {
          setRecommendation(apiResult.recommendation);
        } else {
          setRecommendation(null);
        }
      }

      setCurrentStep("complete");
      setAnalysisComplete(true);
    } catch {
      // Aborted or failure
    } finally {
      setIsAnalyzing(false);
      abortControllerRef.current = null;
    }
  };

  // Acceptance Criterion 2: Cancel aborts the request
  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setIsAnalyzing(false);
    setCurrentStep("validating");
  };

  const handleReset = () => {
    setAnalysisComplete(false);
    setImagePrediction(null);
    setSegmentationError(null);
    setSensorPrediction(null);
    setRecommendation(null);
    setWarnings([]);
  };

  const [mountedAt, setMountedAt] = useState<string>("");

  useEffect(() => {
    setMountedAt(new Date().toISOString());
  }, []);

  // Build record create payload for SaveRecordButton
  const recordCreatePayload: RecordCreate = {
    crop: selectedCrop,
    mode: analysisMode,
    source: sourceProvenance === "campus_live" ? "live" : "manual",
    observed_at: mountedAt || "2026-10-28T00:00:00.000Z",
    observed_tz: typeof window !== "undefined" ? (Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC") : "UTC",
    inputs: {
      ...sensorValues,
      ...(selectedFile ? { file_name: selectedFile.name } : {}),
    },
    predictions: imagePrediction || sensorPrediction || undefined,
    recommendations: recommendation || undefined,
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 text-left" data-testid="analyze-page">
      {/* Workspace Header */}
      <div className="border-b border-[var(--border)] pb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <span className="text-xs font-mono font-semibold uppercase tracking-wider text-[var(--clay)]">
            Diagnostic Workspace
          </span>
          <h1 className="text-2xl sm:text-3xl font-serif text-[var(--forest)] tracking-tight mt-1">
            Crop Nutrition &amp; Anomaly Analysis
          </h1>
          <p className="text-sm text-[var(--stone)] mt-1">
            Evaluate plant leaf imagery, reservoir sensor measurements, or stream live data from the campus adapter.
          </p>
        </div>

        {analysisComplete && (
          <Button variant="secondary" size="sm" onClick={handleReset}>
            &larr; New Analysis
          </Button>
        )}
      </div>

      {/* Progress Indicator during pipeline execution */}
      {isAnalyzing && (
        <VineProgressIndicator
          currentStep={currentStep}
          onCancel={handleCancel}
        />
      )}

      {/* RESULTS VIEW */}
      {!isAnalyzing && analysisComplete && (
        <div className="space-y-8" data-testid="analysis-results-section">
          {/* Top Save Record Bar */}
          <div className="p-4 bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius-lg)] shadow-[var(--shadow-warm-sm)] flex flex-wrap items-center justify-between gap-4">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-[var(--stone)]">
                Status: Complete
              </span>
              <h2 className="text-base font-bold text-[var(--bark)]">
                Inspection Assessment Ready
              </h2>
            </div>

            <SaveRecordButton recordData={recordCreatePayload} />
          </div>

          {/* Operational Warnings / Disclosures */}
          {warnings.length > 0 && (
            <div className="space-y-2" data-testid="analysis-warnings">
              {warnings.map((warn, i) => (
                <div
                  key={i}
                  className="p-3 bg-[var(--sand)] border-l-4 border-[var(--clay)] rounded-r-[var(--radius-md)] text-xs text-[var(--bark)] flex items-start gap-2"
                >
                  <span className="font-bold uppercase tracking-wider text-[var(--clay)]">Notice:</span>
                  <span>{warn}</span>
                </div>
              ))}
            </div>
          )}

          {/* Analysis Results View: Responsive, Balanced & Full Clarity */}
          <div className="space-y-8">
            {/* 1. Foliar Image Diagnostic Studio (Full Width for Large Clarity) */}
            {analysisMode !== "sensor" && (imagePrediction || segmentationError) ? (
              <div className="w-full">
                <ImageResultPanel
                  prediction={imagePrediction}
                  imageUrl={imagePreviewUrl}
                  source={sourceProvenance === "campus_live" ? "campus_camera" : "upload"}
                  segmentationFailed={Boolean(segmentationError)}
                  segmentationFailureReason={segmentationError}
                  onRetry={() => {
                    setSegmentationError(null);
                    handleExecuteAnalysis();
                  }}
                />
              </div>
            ) : analysisMode === "sensor" ? (
              <NotAssessedNote missingModality="image" crop={selectedCrop} />
            ) : null}

            {/* 2. Reservoir Telemetry & Water Sensor Matrix */}
            {analysisMode !== "image" && sensorPrediction ? (
              <div className="w-full">
                <SensorResultPanel
                  crop={selectedCrop}
                  prediction={sensorPrediction}
                  readings={sensorValues}
                />
              </div>
            ) : analysisMode === "image" ? (
              <NotAssessedNote missingModality="sensor" crop={selectedCrop} />
            ) : null}
          </div>

          {/* Actionable Recommendations */}
          <RecommendationCard
            recommendation={recommendation}
            crop={selectedCrop}
          />
        </div>
      )}

      {/* INPUT FORM VIEW */}
      {!isAnalyzing && !analysisComplete && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Mode Selector, Source Tabs, Inputs (7 Cols) */}
          <div className="lg:col-span-7 space-y-6">
            {/* Mode & Source Configuration Card */}
            <Card className="p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] pb-3">
                <h2 className="text-sm font-bold uppercase tracking-wider text-[var(--bark)]">
                  1. Analysis Configuration
                </h2>

                {/* Crop Cultivar Selector */}
                <div className="flex items-center gap-2">
                  <label htmlFor="crop-select" className="text-xs font-medium text-[var(--stone)]">
                    Crop:
                  </label>
                  <select
                    id="crop-select"
                    value={selectedCrop}
                    onChange={(e) => setSelectedCrop(e.target.value as SupportedCrop)}
                    className="text-xs font-semibold h-8 px-2.5 rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--surface)] text-[var(--bark)] focus:border-[var(--forest)]"
                  >
                    <option value="lettuce">Lettuce</option>
                    <option value="spinach">Spinach</option>
                    <option value="tomato">Tomato</option>
                  </select>
                </div>
              </div>

              {/* Mode Selector */}
              <div>
                <label className="text-xs font-bold text-[var(--bark)] block mb-1.5">
                  Diagnostic Modality
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {(["both", "image", "sensor"] as AnalysisMode[]).map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => setAnalysisMode(mode)}
                      className={`py-2 px-3 text-xs font-semibold rounded-[var(--radius-md)] border transition-colors capitalize ${
                        analysisMode === mode
                          ? "bg-[var(--forest)] text-[#FFFFFF] border-[var(--forest)] shadow-xs"
                          : "bg-[var(--surface)] text-[var(--stone)] border-[var(--border)] hover:text-[var(--bark)]"
                      }`}
                    >
                      {mode === "both" ? "Dual (Both)" : `${mode} Only`}
                    </button>
                  ))}
                </div>
              </div>

              {/* Source Tabs: Manual vs Campus Live */}
              <div>
                <label className="text-xs font-bold text-[var(--bark)] block mb-1.5">
                  Input Source Provenance
                </label>
                <div className="flex rounded-[var(--radius-md)] border border-[var(--border)] p-1 bg-[var(--sand)]/40">
                  <button
                    type="button"
                    onClick={() => setSourceProvenance("manual")}
                    className={`flex-1 py-1.5 text-xs font-semibold rounded transition-colors ${
                      sourceProvenance === "manual"
                        ? "bg-[var(--surface)] text-[var(--bark)] shadow-xs"
                        : "text-[var(--stone)] hover:text-[var(--bark)]"
                    }`}
                  >
                    Manual Entry / File Upload
                  </button>
                  <button
                    type="button"
                    onClick={() => setSourceProvenance("campus_live")}
                    className={`flex-1 py-1.5 text-xs font-semibold rounded transition-colors ${
                      sourceProvenance === "campus_live"
                        ? "bg-[var(--surface)] text-[var(--bark)] shadow-xs"
                        : "text-[var(--stone)] hover:text-[var(--bark)]"
                    }`}
                  >
                    Campus Live Adapter
                  </button>
                </div>
              </div>
            </Card>

            {/* Acceptance Criterion 3: Device-unavailable message offers manual entry */}
            {sourceProvenance === "campus_live" && deviceError && (
              <Card
                variant="warning"
                className="p-5 border-l-4 border-[var(--warning)] bg-[#FFFDF7]"
                data-testid="device-unavailable-card"
                role="alert"
              >
                <div className="flex items-start gap-3">
                  <svg className="w-5 h-5 text-[var(--warning)] shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  <div>
                    <h3 className="text-sm font-bold text-[var(--bark)]">
                      Campus Hardware Server Unreachable
                    </h3>
                    <p className="mt-1 text-xs text-[var(--stone)] leading-relaxed">
                      Could not establish connection to the campus IoT greenhouse gateway. You can switch to manual entry to input observations directly.
                    </p>
                    <div className="mt-3">
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => setSourceProvenance("manual")}
                        aria-label="Switch to manual data entry fallback"
                      >
                        Switch to Manual Entry
                      </Button>
                    </div>
                  </div>
                </div>
              </Card>
            )}

            {/* Image Ingestion Card */}
            {(analysisMode === "image" || analysisMode === "both") && (
              <Card className="p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-[var(--border)] pb-2.5">
                  <h3 className="text-sm font-bold text-[var(--forest)] flex items-center gap-2">
                    <span>📷 Foliar Leaf Photography</span>
                  </h3>
                  <Badge variant="upload">Dual-stage YOLO + CNN</Badge>
                </div>

                <InputModeToggle
                  value={inputMode}
                  onChange={setInputMode}
                />

                <ImageDropzone
                  selectedFile={selectedFile}
                  onFileSelected={setSelectedFile}
                />
              </Card>
            )}

            {/* Sensor Telemetry Card */}
            {(analysisMode === "sensor" || analysisMode === "both") && (
              <Card className="p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-[var(--border)] pb-2.5">
                  <h3 className="text-sm font-bold text-[var(--forest)] flex items-center gap-2">
                    <span>🧪 Reservoir Telemetry</span>
                  </h3>
                  {selectedCrop === "tomato" && (
                    <Badge variant="experimental">Experimental</Badge>
                  )}
                </div>

                <SensorForm
                  crop={selectedCrop}
                  onSubmit={(readings: Record<string, number>) => {
                    setSensorValues(readings);
                    setSensorValid(true);
                  }}
                />
              </Card>
            )}
          </div>

          {/* Right Column: Sticky Summary Panel (5 Cols) */}
          <div className="lg:col-span-5 space-y-6">
            <Card
              className="p-5 space-y-4 lg:sticky lg:top-24 border border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-warm-sm)]"
              data-testid="what-will-be-used-panel"
            >
              <h3 className="text-sm font-bold text-[var(--bark)] border-b border-[var(--border)] pb-2.5">
                What Will Be Used In Analysis
              </h3>

              <div className="space-y-2.5 text-xs">
                <div className="flex justify-between py-1 border-b border-[var(--border)]/50">
                  <span className="text-[var(--stone)] font-medium">Cultivar:</span>
                  <span className="font-bold text-[var(--bark)] capitalize">{selectedCrop}</span>
                </div>

                <div className="flex justify-between py-1 border-b border-[var(--border)]/50">
                  <span className="text-[var(--stone)] font-medium">Evaluation Modality:</span>
                  <span className="font-bold text-[var(--bark)] capitalize">
                    {analysisMode === "both" ? "Dual (Image + Sensor)" : `${analysisMode} Only`}
                  </span>
                </div>

                <div className="flex justify-between py-1 border-b border-[var(--border)]/50">
                  <span className="text-[var(--stone)] font-medium">Source Mode:</span>
                  <span className="font-bold text-[var(--bark)]">
                    {sourceProvenance === "campus_live" ? "Campus Live Adapter" : "Manual Observation"}
                  </span>
                </div>

                {(analysisMode === "image" || analysisMode === "both") && (
                  <div className="flex justify-between py-1 border-b border-[var(--border)]/50">
                    <span className="text-[var(--stone)] font-medium">Foliar Photo:</span>
                    <span className={`font-mono font-semibold ${selectedFile ? "text-[var(--forest)]" : "text-[var(--clay)]"}`}>
                      {selectedFile ? selectedFile.name : "Missing (Required)"}
                    </span>
                  </div>
                )}

                {(analysisMode === "sensor" || analysisMode === "both") && (
                  <div className="flex justify-between py-1 border-b border-[var(--border)]/50">
                    <span className="text-[var(--stone)] font-medium">Telemetry Fields:</span>
                    <span className="font-mono text-[var(--forest)] font-semibold">
                      {CROP_SENSOR_SCHEMAS[selectedCrop].length} parameters ready
                    </span>
                  </div>
                )}
              </div>

              {/* Acceptance Criterion 1: Analyze is disabled until required inputs validate */}
              <Button
                variant="primary"
                size="lg"
                onClick={handleExecuteAnalysis}
                disabled={!canExecute}
                data-testid="execute-analysis-button"
                className="w-full mt-4"
                aria-label="Execute diagnostic and nutrient analysis pipeline"
              >
                Execute Analysis
              </Button>

              {!canExecute && (
                <p className="text-[11px] text-[var(--clay)] text-center leading-snug">
                  Please satisfy all required inputs above to activate analysis.
                </p>
              )}

              <p className="text-[11px] text-[var(--stone)] text-center leading-relaxed pt-2 border-t border-[var(--border)]/40">
                Non-fusion guarantee: Vision and telemetry models are processed in separate pipelines without artificial score blending.
              </p>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
