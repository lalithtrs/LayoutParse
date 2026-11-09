"""Production metrics and data drift monitoring for Document AI API."""

import logging
import threading
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Thread-safe telemetry tracker for API latency, confidence, and drift detection."""

    def __init__(self, low_confidence_threshold: float = 0.65):
        self.threshold = low_confidence_threshold
        self._lock = threading.Lock()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.low_confidence_drift_count = 0
        self.latency_records_ms: List[float] = []
        self.confidence_records: List[float] = []

    def record_request(
        self,
        latency_ms: float,
        overall_confidence: float,
        success: bool = True,
    ) -> List[str]:
        """Record an inference event and return any telemetry warnings."""
        warnings: List[str] = []

        with self._lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
                self.latency_records_ms.append(latency_ms)
                self.confidence_records.append(overall_confidence)

                # Keep rolling window of last 1000 items
                if len(self.latency_records_ms) > 1000:
                    self.latency_records_ms.pop(0)
                    self.confidence_records.pop(0)

                # Drift / low confidence detection
                if overall_confidence < self.threshold:
                    self.low_confidence_drift_count += 1
                    msg = (
                        f"Low confidence warning: Document confidence {overall_confidence:.2f} "
                        f"is below drift threshold {self.threshold:.2f}."
                    )
                    logger.warning(msg)
                    warnings.append(msg)
            else:
                self.failed_requests += 1

        return warnings

    def get_summary(self) -> Dict[str, Any]:
        """Return aggregated operational telemetry."""
        with self._lock:
            if not self.latency_records_ms:
                avg_latency = 0.0
                p95_latency = 0.0
            else:
                sorted_lat = sorted(self.latency_records_ms)
                avg_latency = sum(sorted_lat) / len(sorted_lat)
                p95_index = int(0.95 * len(sorted_lat))
                p95_latency = sorted_lat[min(p95_index, len(sorted_lat) - 1)]

            avg_conf = (
                sum(self.confidence_records) / len(self.confidence_records)
                if self.confidence_records
                else 0.0
            )

            drift_rate = (
                self.low_confidence_drift_count / max(1, self.successful_requests)
            )

            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "avg_latency_ms": round(avg_latency, 2),
                "p95_latency_ms": round(p95_latency, 2),
                "avg_confidence": round(avg_conf, 4),
                "low_confidence_drift_count": self.low_confidence_drift_count,
                "drift_rate": round(drift_rate, 4),
            }


# Singleton tracker for the running application
telemetry = MetricsTracker()
