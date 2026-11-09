"""Unit tests for field validation and anomaly detection."""

import unittest
from src.models.genai_validator import DocumentFieldValidator


class TestFieldValidator(unittest.TestCase):
    def setUp(self):
        self.validator = DocumentFieldValidator()

    def test_valid_document_fields(self):
        data = {
            "company": {"value": "BOOK XPRESS SDN BHD"},
            "date": {"value": "2018-12-25"},
            "address": {"value": "NO 12 JALAN SULTAN"},
            "total": {"value": "45.50"},
        }
        res = self.validator.validate_fields(data)
        self.assertTrue(res["is_valid"])
        self.assertEqual(len(res["anomalies"]), 0)
        self.assertEqual(res["validation_score"], 1.0)

    def test_anomalies_detected(self):
        data = {
            "company": {"value": ""},
            "date": {"value": "invalid-date"},
            "address": {"value": "SOME ADDRESS"},
            "total": {"value": "-10.00"},
        }
        res = self.validator.validate_fields(data)
        self.assertFalse(res["is_valid"])
        self.assertGreaterEqual(len(res["anomalies"]), 3)
        self.assertLess(res["validation_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
