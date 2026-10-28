import { ErrorEnvelope } from "./types";

/**
 * Base custom error for Hydroponic Nutrition API interactions.
 */
export abstract class BaseApiError extends Error {
  abstract readonly kind: "api" | "network" | "timeout";
}

/**
 * Thrown when the HTTP server responds with an error code and ErrorEnvelope payload.
 */
export class ApiError extends BaseApiError {
  readonly kind = "api" as const;

  constructor(
    public readonly status: number,
    public readonly envelope: ErrorEnvelope,
    message?: string
  ) {
    const errorMsg = message || envelope?.error?.message || `API Error: HTTP ${status}`;
    super(errorMsg);
    this.name = "ApiError";
    Object.setPrototypeOf(this, ApiError.prototype);
  }

  /**
   * Machine-readable error code from the error envelope.
   */
  get code(): string {
    return this.envelope?.error?.code || "UNKNOWN_ERROR";
  }

  /**
   * Whether the backend tagged this error as safe to retry.
   */
  get retryable(): boolean {
    return Boolean(this.envelope?.error?.retryable);
  }

  /**
   * Diagnostic details attached to the envelope.
   */
  get details(): Record<string, unknown> {
    return this.envelope?.error?.details || {};
  }
}

/**
 * Thrown when low-level network connectivity fails (DNS, offline, connection refused).
 * Strictly distinguishable from HTTP 4xx/5xx ApiErrors.
 */
export class NetworkError extends BaseApiError {
  readonly kind = "network" as const;

  constructor(
    public readonly originalError: unknown,
    message = "Network connection failed. Please check internet or local server connectivity."
  ) {
    super(message);
    this.name = "NetworkError";
    Object.setPrototypeOf(this, NetworkError.prototype);
  }
}

/**
 * Thrown when a request exceeds the configured client timeout.
 */
export class TimeoutError extends BaseApiError {
  readonly kind = "timeout" as const;

  constructor(
    public readonly timeoutMs: number,
    message = `Request timed out after ${timeoutMs}ms`
  ) {
    super(message);
    this.name = "TimeoutError";
    Object.setPrototypeOf(this, TimeoutError.prototype);
  }
}

/**
 * Type guard for ApiError.
 */
export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError;
}

/**
 * Type guard for NetworkError.
 */
export function isNetworkError(error: unknown): error is NetworkError {
  return error instanceof NetworkError;
}

/**
 * Type guard for TimeoutError.
 */
export function isTimeoutError(error: unknown): error is TimeoutError {
  return error instanceof TimeoutError;
}

