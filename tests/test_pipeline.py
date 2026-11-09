"""Unit tests for inference pipeline and telemetry tracker."""

import unittest
from src.api.monitoring import MetricsTracker
from src.pipeline.extractor import DocumentExtractor


class TestPipeline(unittest.TestCase):
    def test_mock_document_extractor(self):
        extractor = DocumentExtractor(model_path=None, ocr_engine_type="mock")
        res = extractor.extract("dummy_path.jpg")

        self.assertEqual(res["status"], "success")
        self.assertIn("data", res)
        self.assertIn("company", res["data"])
        self.assertIn("total", res["data"])
        self.assertIn("date", res["data"])
        self.assertGreater(res["overall_confidence"], 0.0)

    def test_telemetry_metrics_and_drift(self):
        tracker = MetricsTracker(low_confidence_threshold=0.60)

        # Record high-confidence request
        w1 = tracker.record_request(latency_ms=120.5, overall_confidence=0.92, success=True)
        self.assertEqual(len(w1), 0)

        # Record low-confidence request (should trigger drift warning)
        w2 = tracker.record_request(latency_ms=150.0, overall_confidence=0.45, success=True)
        self.assertEqual(len(w2), 1)
        self.assertIn("Low confidence warning", w2[0])

        # Record failure
        tracker.record_request(latency_ms=0, overall_confidence=0.0, success=False)

        summary = tracker.get_summary()
        self.assertEqual(summary["total_requests"], 3)
        self.assertEqual(summary["successful_requests"], 2)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["low_confidence_drift_count"], 1)
        self.assertAlmostEqual(summary["drift_rate"], 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
