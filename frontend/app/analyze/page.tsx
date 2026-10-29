"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
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

const DEFAULT_CROP_SENSOR_VALUES: Record<SupportedCrop, Record<string, number>> = {
  lettuce: {
    ph: 6.0,
    ec: 1.6,
    water_temp: 21.0,
    humidity: 65.0,
    air_temp: 22.5,
  },
  spinach: {
    air_temperature_C: 20.0,
    relative_humidity_percent: 60.0,
    water_temperature_C: 19.5,
    ph: 6.2,
    ec_mS_cm: 2.0,
  },
  tomato: {
    air_temperature_C: 24.0,
    relative_humidity_percent: 65.0,
    water_temperature_C: 22.0,
    ph: 6.2,
    ec_mS_cm: 2.8,
    nitrogen_mg_L: 180.0,
    phosphorus_mg_L: 50.0,
    potassium_mg_L: 280.0,
  },
};

export default function AnalyzePage() {
  // Configuration State
  const [selectedCrop, setSelectedCrop] = useState<SupportedCrop>("lettuce");
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>("both");
  const [sourceProvenance, setSourceProvenance] = useState<SourceProvenance>("manual");

  // Image Inputs: Default to single leaf for typical user leaf captures
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [inputMode, setInputMode] = useState<InputMode | null>("single_leaf");

  // Sensor Inputs synchronized to crop schema
  const [initialSensorValues, setInitialSensorValues] = useState<Record<string, number | undefined>>(
    DEFAULT_CROP_SENSOR_VALUES.lettuce
  );
  const [sensorValues, setSensorValues] = useState<Record<string, number | undefined>>(
    DEFAULT_CROP_SENSOR_VALUES.lettuce
  );
  const [sensorValid, setSensorValid] = useState<boolean>(true);

  // Campus Device Telemetry & Live Camera State
  const [deviceStatus, setDeviceStatus] = useState<DeviceStatusResponse | null>(null);
  const [deviceError, setDeviceError] = useState<boolean>(false);
  const [cameraRefreshKey, setCameraRefreshKey] = useState<number>(0);

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

  const fetchCampusTelemetry = useCallback((cropToFetch?: SupportedCrop) => {
    const targetCrop = cropToFetch || selectedCrop;
    apiClient
      .getDeviceStatus()
      .then((status) => {
        setDeviceStatus(status);
        const isConnected =
          status.status === "ok" ||
          status.connected === true ||
          Boolean(status.readings && Object.keys(status.readings).length > 0);
        setDeviceError(!isConnected);

        if (isConnected) {
          // Fetch accurately mapped sensor schema for the target crop
          apiClient
            .getLatestSensors(targetCrop)
            .then((sensorData) => {
              if (sensorData && sensorData.readings && Object.keys(sensorData.readings).length > 0) {
                const liveMap: Record<string, number> = {};
                for (const [k, v] of Object.entries(sensorData.readings)) {
                  if (typeof v === "number") liveMap[k] = v;
                }
                setInitialSensorValues(liveMap);
                setSensorValues(liveMap);
                setSensorValid(true);
              }
            })
            .catch(() => {
              // Fallback to unpacking device status readings if latest sensors route fails
              if (status.readings) {
                const liveVals: Record<string, number> = {};
                for (const [k, v] of Object.entries(status.readings)) {
                  if (v && typeof v.value === "number") {
                    liveVals[k] = v.value;
                  }
                }
                if (Object.keys(liveVals).length > 0) {
                  setInitialSensorValues((prev) => ({ ...prev, ...liveVals }));
                  setSensorValues((prev) => ({ ...prev, ...liveVals }));
                  setSensorValid(true);
                }
              }
            });
        }
      })
      .catch(() => {
        setDeviceError(true);
      });
  }, [selectedCrop]);

  // Fetch campus device status and live camera preview when switching to campus_live
  useEffect(() => {
    if (sourceProvenance === "campus_live") {
      fetchCampusTelemetry(selectedCrop);
      setImagePreviewUrl(`/api/v1/device/camera/frame?t=${Date.now()}`);
    } else {
      setDeviceError(false);
      if (selectedFile) {
        const url = URL.createObjectURL(selectedFile);
        setImagePreviewUrl(url);
      } else {
        setImagePreviewUrl(null);
      }
    }
  }, [sourceProvenance, selectedCrop, fetchCampusTelemetry]);

  // Manage Image Object URL in manual mode
  useEffect(() => {
    if (sourceProvenance === "manual") {
      if (selectedFile) {
        const url = URL.createObjectURL(selectedFile);
        setImagePreviewUrl(url);
        return () => URL.revokeObjectURL(url);
      } else {
        setImagePreviewUrl(null);
      }
    }
  }, [selectedFile, sourceProvenance]);

  // Synchronize default sensor values when selectedCrop changes in manual mode
  useEffect(() => {
    if (sourceProvenance === "manual") {
      setInitialSensorValues(DEFAULT_CROP_SENSOR_VALUES[selectedCrop]);
      setSensorValues(DEFAULT_CROP_SENSOR_VALUES[selectedCrop]);
      setSensorValid(true);
    }
  }, [selectedCrop, sourceProvenance]);

  const handleSensorChange = useCallback((readings: Record<string, number>, isValid: boolean) => {
    setSensorValues((prev) => {
      const pKeys = Object.keys(prev);
      const rKeys = Object.keys(readings);
      if (pKeys.length === rKeys.length && pKeys.every((k) => prev[k] === readings[k])) {
        return prev;
      }
      return readings;
    });
    setSensorValid((prev) => (prev === isValid ? prev : isValid));
  }, []);

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
  const handleExecuteAnalysis = async (overrideSensors?: Record<string, number>) => {
    const effectiveSensors = overrideSensors || sensorValues;
    const hasSensors = Object.keys(effectiveSensors).length > 0;
    const isDirectSensorSubmit = Boolean(overrideSensors);
    const activeExecMode =
      isDirectSensorSubmit && !selectedFile && sourceProvenance !== "campus_live"
        ? "sensor"
        : analysisMode;

    const effectiveSensorValid = Boolean(activeExecMode === "image" || hasSensors);
    const effectiveImageValid = Boolean(
      activeExecMode === "sensor" ||
      (selectedFile && inputMode !== null) ||
      (sourceProvenance === "campus_live" && !deviceError)
    );

    if (!effectiveImageValid || !effectiveSensorValid || isAnalyzing || (sourceProvenance === "campus_live" && deviceError)) {
      return;
    }

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

      if (activeExecMode === "image" || activeExecMode === "both") {
        if (inputMode) {
          const apiMode = inputMode === "single_leaf" ? "leaf" : "plant";
          formData.append("input_mode", apiMode);
        }
        if (sourceProvenance === "campus_live") {
          formData.append("use_campus_frame", "true");
        } else if (selectedFile) {
          formData.append("file", selectedFile);
        }
      }

      if (activeExecMode === "sensor" || activeExecMode === "both") {
        formData.append("sensor_readings", JSON.stringify(effectiveSensors));
      }

      // Progress animation during deep inference pipeline
      const stepTimer1 = setTimeout(() => {
        if (!controller.signal.aborted) setCurrentStep("classifying");
      }, 5000);
      const stepTimer2 = setTimeout(() => {
        if (!controller.signal.aborted) setCurrentStep("recommending");
      }, 12000);

      let apiResult: import("../../lib/api/types").AnalyzeResponse | null = null;
      try {
        apiResult = await apiClient.analyze(formData);
      } catch (err: unknown) {
        if (controller.signal.aborted) return;
        const errMsg =
          err instanceof Error
            ? err.message
            : "Diagnostic pipeline failed to connect to backend server. Please verify backend is running on port 8000.";
        console.warn("Backend API dispatch failed:", errMsg);
        setSegmentationError(errMsg);
        setImagePrediction(null);
        setSensorPrediction(null);
        setRecommendation(null);
        setIsAnalyzing(false);
        return;
      } finally {
        clearTimeout(stepTimer1);
        clearTimeout(stepTimer2);
      }

      if (controller.signal.aborted) return;

      // Ensure live preview reflects newly evaluated frame
      if (sourceProvenance === "campus_live") {
        setImagePreviewUrl(`/api/v1/device/camera/frame?t=${Date.now()}`);
      }

      // Step 3: Classifying
      setCurrentStep("classifying");
      await new Promise((resolve) => setTimeout(resolve, 300));
      if (controller.signal.aborted) return;

      if (apiResult) {
        // Filter out redundant sensor optimal range notices (shown in detail within Sensor Matrix below)
        const filteredWarnings = (apiResult.warnings || []).filter(
          (warn) => !warn.toLowerCase().includes("outside optimal agronomic range") && !warn.toLowerCase().includes("optimal range")
        );
        setWarnings(filteredWarnings);

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

  // Collect all valid predictions from current analysis
  const activePredictions: PredictionResponse[] = [];
  if (imagePrediction) activePredictions.push(imagePrediction);
  if (sensorPrediction) activePredictions.push(sensorPrediction);

  // Build record create payload for SaveRecordButton
  const recordCreatePayload: RecordCreate = {
    crop: selectedCrop,
    mode: analysisMode,
    source: sourceProvenance === "campus_live" ? "live" : "manual",
    observed_at: mountedAt || new Date().toISOString(),
    observed_tz:
      typeof window !== "undefined"
        ? Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"
        : "UTC",
    inputs: {
      ...sensorValues,
      ...(selectedFile ? { file_name: selectedFile.name } : {}),
    },
    predictions: activePredictions.length > 0 ? activePredictions : undefined,
    recommendation: recommendation
      ? {
          source: recommendation.source || "rules",
          rules_version: "v1.0.0",
          content: {
            summary: recommendation.summary,
            actions: recommendation.actions,
          },
          evidence: {
            citations: recommendation.evidence_citations,
          },
          record_id: "00000000-0000-0000-0000-000000000000",
        }
      : undefined,
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

                {sourceProvenance === "campus_live" ? (
                  <div className="space-y-3">
                    <div className="relative rounded-[var(--radius-md)] overflow-hidden border border-[var(--border)] bg-[#111111] aspect-[16/9] flex items-center justify-center">
                      <img
                        key={cameraRefreshKey}
                        src={`/api/v1/device/camera/frame?t=${cameraRefreshKey}`}
                        alt="Campus Greenhouse Live Camera Preview"
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          (e.target as HTMLElement).style.display = "none";
                        }}
                      />
                      <div className="absolute top-3 left-3 flex items-center gap-2 bg-[#000000]/75 backdrop-blur-xs px-2.5 py-1 rounded text-white text-xs font-semibold">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                        LIVE CANOPY FEED
                      </div>
                      <button
                        type="button"
                        onClick={() => setCameraRefreshKey((k) => k + 1)}
                        className="absolute bottom-3 right-3 bg-[var(--surface)]/90 hover:bg-[var(--surface)] text-[var(--bark)] text-xs font-semibold px-3 py-1.5 rounded shadow-sm border border-[var(--border)] transition-colors flex items-center gap-1.5 cursor-pointer"
                      >
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                        Refresh Snapshot
                      </button>
                    </div>
                    <p className="text-xs text-[var(--stone)] leading-relaxed">
                      Streaming from polyhouse CPPLUS IP camera. When you click <strong>Execute Analysis</strong> below, this active frame is captured in real-time and evaluated by the multi-leaf segmentation model.
                    </p>
                  </div>
                ) : (
                  <>
                    <InputModeToggle
                      value={inputMode}
                      onChange={setInputMode}
                    />

                    <ImageDropzone
                      selectedFile={selectedFile}
                      onFileSelected={setSelectedFile}
                    />
                  </>
                )}
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

                {sourceProvenance === "campus_live" && !deviceError && (
                  <div className="p-3 bg-emerald-50/70 border border-emerald-200/80 rounded-[var(--radius-md)] text-xs text-emerald-900 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                      <span className="font-semibold">Populated from Live Campus IoT Gateway</span>
                      <span className="text-emerald-700 font-mono text-[11px]">(Fresh)</span>
                    </div>
                    <button
                      type="button"
                      onClick={() => fetchCampusTelemetry()}
                      className="text-emerald-800 font-medium underline hover:text-emerald-950 text-xs cursor-pointer"
                    >
                      Refresh Telemetry
                    </button>
                  </div>
                )}

                <SensorForm
                  crop={selectedCrop}
                  initialValues={initialSensorValues}
                  onChange={handleSensorChange}
                  onSubmit={(readings: Record<string, number>) => {
                    setSensorValues(readings);
                    setSensorValid(true);
                    handleExecuteAnalysis(readings);
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
                    <span
                      className={`font-mono font-semibold ${
                        sourceProvenance === "campus_live" && !deviceError
                          ? "text-[var(--forest)]"
                          : selectedFile
                          ? "text-[var(--forest)]"
                          : "text-[var(--clay)]"
                      }`}
                    >
                      {sourceProvenance === "campus_live" && !deviceError
                        ? "Live Polyhouse Camera Stream"
                        : selectedFile
                        ? selectedFile.name
                        : "Missing (Required)"}
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
                onClick={() => handleExecuteAnalysis()}
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
