"""Operational telemetry and fallback event tracking (T9-004).

Tracks subsystem fallback events across:
- gemini_to_rules: LLM recommendation failures falling back to deterministic rules
- gpu_to_cpu: CUDA OOM or device errors falling back to CPU execution
- segmentation_failure: Leaf extraction / segmentation pipeline failures

Provides thread-safe atomic counters, structured logging, and Prometheus-formatted export.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class FallbackTracker:
    """Thread-safe telemetry tracker for resilience and fallback events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {
            "gemini_to_rules": 0,
            "gpu_to_cpu": 0,
            "segmentation_failure": 0,
        }

    def increment_gemini_to_rules(self, reason: str | None = None) -> int:
        """Increment gemini_to_rules counter and emit a structured log event."""
        with self._lock:
            self._counters["gemini_to_rules"] += 1
            count = self._counters["gemini_to_rules"]

        logger.warning(
            "Fallback event recorded: gemini_to_rules (total=%d, reason=%s)",
            count,
            reason or "unspecified",
            extra={
                "fallback_event": "gemini_to_rules",
                "counter_value": count,
                "reason": reason or "unspecified",
            },
        )
        return count

    def increment_gpu_to_cpu(self, reason: str | None = None) -> int:
        """Increment gpu_to_cpu counter and emit a structured log event."""
        with self._lock:
            self._counters["gpu_to_cpu"] += 1
            count = self._counters["gpu_to_cpu"]

        logger.warning(
            "Fallback event recorded: gpu_to_cpu (total=%d, reason=%s)",
            count,
            reason or "unspecified",
            extra={
                "fallback_event": "gpu_to_cpu",
                "counter_value": count,
                "reason": reason or "unspecified",
            },
        )
        return count

    def increment_segmentation_failure(self, reason: str | None = None) -> int:
        """Increment segmentation_failure counter and emit a structured log event."""
        with self._lock:
            self._counters["segmentation_failure"] += 1
            count = self._counters["segmentation_failure"]

        logger.warning(
            "Fallback event recorded: segmentation_failure (total=%d, reason=%s)",
            count,
            reason or "unspecified",
            extra={
                "fallback_event": "segmentation_failure",
                "counter_value": count,
                "reason": reason or "unspecified",
            },
        )
        return count

    def get_counters(self) -> dict[str, int]:
        """Return a snapshot of current counter values."""
        with self._lock:
            return dict(self._counters)

    def reset(self) -> None:
        """Reset all counters to zero (useful for testing)."""
        with self._lock:
            for key in self._counters:
                self._counters[key] = 0

    def format_prometheus_metrics(self) -> str:
        """Format counters into standard Prometheus text format."""
        counters = self.get_counters()
        lines = [
            "# HELP gemini_to_rules_total Total count of Gemini fallback events to rule-based recommendations",
            "# TYPE gemini_to_rules_total counter",
            f"gemini_to_rules_total {counters.get('gemini_to_rules', 0)}",
            "",
            "# HELP gpu_to_cpu_total Total count of GPU OOM or compute fallback events to CPU",
            "# TYPE gpu_to_cpu_total counter",
            f"gpu_to_cpu_total {counters.get('gpu_to_cpu', 0)}",
            "",
            "# HELP segmentation_failure_total Total count of leaf segmentation failures",
            "# TYPE segmentation_failure_total counter",
            f"segmentation_failure_total {counters.get('segmentation_failure', 0)}",
            "",
        ]
        return "\n".join(lines)


# Singleton instance
_fallback_tracker = FallbackTracker()


def get_fallback_tracker() -> FallbackTracker:
    """Return the global FallbackTracker instance."""
    return _fallback_tracker
