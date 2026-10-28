/**
 * TypeScript API contracts matching docs/api/openapi.json and FastAPI backend schemas (T7-005).
 */

// =============================================================================
// Error Envelopes (Acceptance Criterion 1)
// =============================================================================

export interface ErrorDetail {
  code: string;
  message: string;
  details: Record<string, unknown>;
  retryable: boolean;
}

export interface ErrorEnvelope {
  error: ErrorDetail;
}

// =============================================================================
// Health & Telemetry
// =============================================================================

export interface HealthResponse {
  status: string;
  device: string;
  db: string;
  registry: string;
}

// =============================================================================
// Model Registry Discovery (GET /models)
// =============================================================================

export interface ModelSummary {
  crop: "lettuce" | "spinach" | "tomato" | "tomato_live" | string;
  task: "classification" | "segmentation" | "feature_extraction" | string;
  modality: "sensor" | "image" | "feature" | string;
  family: string;
  version: string;
  status: "promoted" | "candidate" | "MODEL_UNAVAILABLE" | "SCHEMA_MISMATCH" | string;
  validation_macro_f1?: number | null;
  experimental: boolean;
  error_detail?: string | null;
  classes?: string[] | null;
  ordered_features?: string[] | null;
  display_mapping?: Record<string, string> | null;
  split_protocol?: string | null;
  metrics?: Record<string, unknown> | null;
}

// =============================================================================
// Sensor Telemetry & Prediction
// =============================================================================

export interface SensorReadings {
  ph?: number;
  ec_ms_cm?: number;
  water_temp_c?: number;
  tds_ppm?: number;
  dissolved_oxygen_mg_l?: number;
  air_temp_c?: number;
  humidity_pct?: number;
  co2_ppm?: number;
  [key: string]: number | undefined;
}

export interface SensorPredictRequest {
  crop: "lettuce" | "spinach" | "tomato";
  readings: SensorReadings;
  source?: "manual" | "live" | "imported";
}

export interface LeafPrediction {
  leaf_id: string;
  contour_area: number;
  predicted_class: string;
  display_label: string;
  confidence: number;
  is_uncertain: boolean;
  mask_metrics?: Record<string, unknown>;
}

export interface PredictionResponse {
  crop: string;
  modality: "sensor" | "image" | "both";
  model_name: string;
  model_version: string;
  predicted_class: string;
  display_label: string;
  confidence: number;
  is_uncertain: boolean;
  inference_time_ms: number;
  evidence: Record<string, unknown>;
  leaves?: LeafPrediction[] | null;
  leaf_count?: number | null;
  rejected_leaves?: Array<{ leaf_id: string; reason: string }> | null;
}

// =============================================================================
// Advisory Recommendations
// =============================================================================

export interface RecommendationAction {
  parameter: string;
  current_reading: number;
  target_range: [number, number];
  action_type: "increase" | "decrease" | "maintain" | "flush" | string;
  adjustment_amount: number;
  unit: string;
  urgency: "immediate" | "scheduled" | "advisory";
  rationale: string;
}

export interface RecommendationResponse {
  recommendation_id: string;
  crop: string;
  summary: string;
  actions: RecommendationAction[];
  evidence_citations: string[];
  source: "rules" | "gemini";
  generated_at: string;
}

// =============================================================================
// Device Status & Telemetry Adapters (GET /device/status)
// =============================================================================

export interface DeviceStatusResponse {
  adapter: "mock" | "manual" | "campus_server" | string;
  connected: boolean;
  available_sensors: string[];
  camera_available: boolean;
  last_reading_utc?: string | null;
}

// =============================================================================
// Historical Records (GET / POST /api/v1/history)
// =============================================================================

export interface RecordCreate {
  crop: "lettuce" | "spinach" | "tomato";
  mode: "sensor" | "image" | "both";
  source: "manual" | "live" | "imported";
  observed_at: string;
  observed_tz: string;
  notes?: string | null;
  inputs?: Record<string, unknown> | null;
  predictions?: PredictionResponse | null;
  recommendations?: RecommendationResponse | null;
}

export interface RecordRead extends RecordCreate {
  id: string;
  created_at: string;
}

export interface HistoryQueryParams {
  crop?: string;
  mode?: string;
  source?: string;
  limit?: number;
  offset?: number;
  start_date?: string;
  end_date?: string;
}

export interface RecordListResponse {
  records: RecordRead[];
  total: number;
  limit: number;
  offset: number;
}
