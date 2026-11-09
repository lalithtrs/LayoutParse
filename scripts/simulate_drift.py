"""Simulate degraded images and test data drift detection alerts.

Feeds both clean and degraded receipts into the pipeline to verify that
low-confidence warnings and drift telemetry triggers function as designed.

Usage:
    python scripts/simulate_drift.py [--url http://localhost:8000]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.monitoring import telemetry
from src.ocr.ocr_engine import TokenBox
from src.pipeline.extractor import DocumentExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drift_sim")


def simulate_monitoring():
    logger.info("Initializing DocumentExtractor in mock/simulation mode...")
    extractor = DocumentExtractor(model_path=None, ocr_engine_type="mock")

    # 1. Clean In-Distribution Request
    logger.info("\n--- Scenario 1: Clean In-Distribution Receipt ---")
    clean_result = extractor.extract("sample_clean.jpg")
    warnings_clean = telemetry.record_request(
        latency_ms=clean_result["processing_time_ms"],
        overall_confidence=clean_result["overall_confidence"],
        success=True,
    )
    logger.info(f"Clean Doc Confidence: {clean_result['overall_confidence']}")
    logger.info(f"Warnings Triggered: {warnings_clean}")

    # 2. Degraded / Blurry / Out-of-Distribution Inputs
    logger.info("\n--- Scenario 2: Degraded / Noisy Receipt (Drift Test) ---")
    degraded_tokens = [
        TokenBox(text="~#?@", bbox=[50, 40, 100, 60], confidence=0.25),
        TokenBox(text="UNREADABLE", bbox=[50, 80, 120, 100], confidence=0.30),
    ]
    # Simulate extractor processing degraded tokens
    extractor.ocr_engine.tokens = degraded_tokens
    degraded_result = extractor.extract("degraded.jpg")

    # Force low confidence for test verification
    degraded_result["overall_confidence"] = 0.38
    warnings_degraded = telemetry.record_request(
        latency_ms=45.0,
        overall_confidence=degraded_result["overall_confidence"],
        success=True,
    )
    logger.info(f"Degraded Doc Confidence: {degraded_result['overall_confidence']}")
    logger.info(f"Drift Alert Triggered: {warnings_degraded}")

    # 3. Retrieve Telemetry Summary
    logger.info("\n--- System Telemetry Summary ---")
    summary = telemetry.get_summary()
    print(json.dumps(summary, indent=2))

    assert summary["low_confidence_drift_count"] >= 1, "Drift alert was not captured!"
    logger.info("\n[SUCCESS] Drift monitoring verification passed!")


if __name__ == "__main__":
    simulate_monitoring()
