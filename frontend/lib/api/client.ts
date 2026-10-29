import { ApiError, NetworkError, TimeoutError } from "./errors";
import {
  AnalyzeResponse,
  DeviceStatusResponse,
  ErrorEnvelope,
  HealthResponse,
  HistoryQueryParams,
  ModelSummary,
  PredictionResponse,
  RecordCreate,
  RecordListResponse,
  RecordRead,
  SensorPredictRequest,
} from "./types";

export interface ApiClientConfig {
  baseUrl?: string;
  defaultTimeoutMs?: number;
}

export class ApiClient {
  private baseUrl: string;
  private defaultTimeoutMs: number;

  constructor(config: ApiClientConfig = {}) {
    this.baseUrl =
      config.baseUrl ||
      (typeof window !== "undefined"
        ? "/api/v1"
        : typeof process !== "undefined" && process.env.NEXT_PUBLIC_API_URL
        ? process.env.NEXT_PUBLIC_API_URL
        : "http://127.0.0.1:8000/api/v1");
    // Strip trailing slash if present
    this.baseUrl = this.baseUrl.replace(/\/+$/, "");
    this.defaultTimeoutMs = config.defaultTimeoutMs ?? 60000;
  }

  /**
   * Core typed HTTP request wrapper enforcing timeouts and structured error envelope parsing.
   */
  async request<T>(
    endpoint: string,
    options: RequestInit & { timeoutMs?: number } = {}
  ): Promise<T> {
    const { timeoutMs = this.defaultTimeoutMs, ...fetchOptions } = options;

    const normalizedEndpoint = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
    const url = `${this.baseUrl}${normalizedEndpoint}`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(url, {
        ...fetchOptions,
        signal: controller.signal,
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
          ...fetchOptions.headers,
        },
      });

      // Clear the timeout timer upon receiving HTTP headers
      clearTimeout(timeoutId);

      // Handle successful HTTP status codes
      if (response.ok) {
        if (response.status === 204) {
          return undefined as unknown as T;
        }
        return (await response.json()) as T;
      }

      // Handle failure HTTP status codes: Parse structured error envelope
      let envelope: ErrorEnvelope;
      try {
        const json = await response.json();
        if (json && typeof json === "object" && "error" in json) {
          envelope = json as ErrorEnvelope;
        } else {
          // Fallback if backend returned non-envelope JSON
          envelope = {
            error: {
              code: `HTTP_${response.status}`,
              message: json?.detail || response.statusText || "Request failed",
              details: json || {},
              retryable: response.status >= 500,
            },
          };
        }
      } catch {
        // Fallback if response body is non-JSON (e.g. gateway timeout 504 HTML)
        envelope = {
          error: {
            code: `HTTP_${response.status}`,
            message: response.statusText || `HTTP ${response.status} Error`,
            details: {},
            retryable: response.status >= 500,
          },
        };
      }

      throw new ApiError(response.status, envelope);
    } catch (err: unknown) {
      clearTimeout(timeoutId);

      // Distinguish already-created ApiError
      if (err instanceof ApiError) {
        throw err;
      }

      // Distinguish Timeout / Abort errors
      if (
        (err instanceof DOMException && err.name === "AbortError") ||
        (err instanceof Error && err.name === "AbortError")
      ) {
        throw new TimeoutError(timeoutMs);
      }

      // Distinguish network connection failures
      throw new NetworkError(err);
    }
  }

  // ===========================================================================
  // Domain Endpoints
  // ===========================================================================

  /**
   * Health check and detected compute device.
   */
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>("/health");
  }

  /**
   * Model registry discovery: list promoted and registered models.
   */
  async getModels(params?: {
    crop?: string;
    task?: string;
    modality?: string;
  }): Promise<{ models: ModelSummary[]; count: number }> {
    const query = new URLSearchParams();
    if (params?.crop) query.set("crop", params.crop);
    if (params?.task) query.set("task", params.task);
    if (params?.modality) query.set("modality", params.modality);

    const qs = query.toString();
    const raw = await this.request<ModelSummary[] | { models: ModelSummary[]; count: number }>(
      `/models${qs ? `?${qs}` : ""}`
    );
    if (Array.isArray(raw)) {
      return { models: raw, count: raw.length };
    }
    return raw;
  }

  /**
   * Classical machine learning prediction on water telemetry features.
   */
  async predictSensor(request: SensorPredictRequest): Promise<PredictionResponse> {
    return this.request<PredictionResponse>("/predict/sensor", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  /**
   * Foliar plant image prediction via multipart upload or live campus camera frame.
   */
  async predictImage(formData: FormData): Promise<PredictionResponse> {
    const url = `${this.baseUrl}/predict/image`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.defaultTimeoutMs);

    try {
      const response = await fetch(url, {
        method: "POST",
        body: formData,
        signal: controller.signal,
        headers: {
          Accept: "application/json",
        },
      });
      clearTimeout(timeoutId);
      if (response.ok) {
        return (await response.json()) as PredictionResponse;
      }
      let envelope: ErrorEnvelope;
      try {
        envelope = (await response.json()) as ErrorEnvelope;
      } catch {
        envelope = {
          error: {
            code: `HTTP_${response.status}`,
            message: response.statusText,
            details: {},
            retryable: response.status >= 500,
          },
        };
      }
      throw new ApiError(response.status, envelope);
    } catch (err: unknown) {
      clearTimeout(timeoutId);
      if (err instanceof ApiError) throw err;
      if (
        (err instanceof DOMException && err.name === "AbortError") ||
        (err instanceof Error && err.name === "AbortError")
      ) {
        throw new TimeoutError(this.defaultTimeoutMs);
      }
      throw new NetworkError(err);
    }
  }

  /**
   * Multi-modal analysis across sensor telemetry and/or foliar imagery with partial failure tolerance.
   */
  async analyze(formData: FormData): Promise<AnalyzeResponse> {
    const url = `${this.baseUrl}/analyze`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.defaultTimeoutMs);

    try {
      const response = await fetch(url, {
        method: "POST",
        body: formData,
        signal: controller.signal,
        headers: {
          Accept: "application/json",
        },
      });
      clearTimeout(timeoutId);
      if (response.ok) {
        return (await response.json()) as AnalyzeResponse;
      }
      let envelope: ErrorEnvelope;
      try {
        envelope = (await response.json()) as ErrorEnvelope;
      } catch {
        envelope = {
          error: {
            code: `HTTP_${response.status}`,
            message: response.statusText,
            details: {},
            retryable: response.status >= 500,
          },
        };
      }
      throw new ApiError(response.status, envelope);
    } catch (err: unknown) {
      clearTimeout(timeoutId);
      if (err instanceof ApiError) throw err;
      if (
        (err instanceof DOMException && err.name === "AbortError") ||
        (err instanceof Error && err.name === "AbortError")
      ) {
        throw new TimeoutError(this.defaultTimeoutMs);
      }
      throw new NetworkError(err);
    }
  }

  /**
   * List historical inspection records with pagination and filters.
   */
  async getHistory(params: HistoryQueryParams = {}): Promise<RecordListResponse> {
    const query = new URLSearchParams();
    if (params.crop) query.set("crop", params.crop);
    if (params.mode) query.set("mode", params.mode);
    if (params.source) query.set("source", params.source);
    if (params.limit !== undefined) query.set("limit", String(params.limit));
    if (params.offset !== undefined) query.set("offset", String(params.offset));
    if (params.start_date) query.set("start_date", params.start_date);
    if (params.end_date) query.set("end_date", params.end_date);

    const qs = query.toString();
    return this.request<RecordListResponse>(`/history${qs ? `?${qs}` : ""}`);
  }

  /**
   * Build complete URL for downloading history CSV with current filters applied.
   */
  getHistoryExportUrl(params: HistoryQueryParams = {}): string {
    const query = new URLSearchParams();
    if (params.crop) query.set("crop", params.crop);
    if (params.mode) query.set("mode", params.mode);
    if (params.source) query.set("source", params.source);
    if (params.start_date) query.set("start_date", params.start_date);
    if (params.end_date) query.set("end_date", params.end_date);

    const qs = query.toString();
    return `${this.baseUrl}/history/export${qs ? `?${qs}` : ""}`;
  }

  /**
   * Fetch a single historical record by ID.
   */
  async getRecord(recordId: string): Promise<RecordRead> {
    return this.request<RecordRead>(`/history/${recordId}`);
  }

  /**
   * Persist a completed inspection record.
   */
  async createRecord(record: RecordCreate): Promise<RecordRead> {
    return this.request<RecordRead>("/history", {
      method: "POST",
      body: JSON.stringify(record),
    });
  }

  /**
   * Query status of campus hardware / simulated devices.
   */
  async getDeviceStatus(): Promise<DeviceStatusResponse> {
    return this.request<DeviceStatusResponse>("/device/status");
  }

  /**
   * Fetch latest mapped sensor readings strictly conforming to crop schema.
   */
  async getLatestSensors(crop: string = "lettuce"): Promise<{
    crop: string;
    source: string;
    readings: Record<string, number>;
    is_mock: boolean;
    experimental: boolean;
    warnings: string[];
  }> {
    return this.request<{
      crop: string;
      source: string;
      readings: Record<string, number>;
      is_mock: boolean;
      experimental: boolean;
      warnings: string[];
    }>(`/device/sensors/latest?crop=${encodeURIComponent(crop)}`);
  }

  /**
   * Return direct URL for streaming/fetching the latest live camera frame.
   */
  getCameraFrameUrl(): string {
    return `${this.baseUrl}/device/camera/frame`;
  }
}

/**
 * Singleton API client instance for standard application usage.
 */
export const apiClient = new ApiClient();

