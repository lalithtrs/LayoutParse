"""Unit tests for entity postprocessing, confidence aggregation and value normalization."""

import unittest
from src.models.postprocessor import DocumentPostProcessor, parse_date, parse_total


class TestPostProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentPostProcessor()

    def test_parse_date_formats(self):
        self.assertEqual(parse_date("25/12/2018"), "2018-12-25")
        self.assertEqual(parse_date("2019-03-15"), "2019-03-15")
        self.assertEqual(parse_date("15-04-19"), "2019-04-15")

    def test_parse_total_formats(self):
        self.assertEqual(parse_total("RM 45.50"), "45.50")
        self.assertEqual(parse_total("$1,250.00"), "1250.00")
        self.assertEqual(parse_total("TOTAL: 8.90"), "8.90")

    def test_aggregate_entities(self):
        tokens = [
            "BOOK", "XPRESS", "SDN", "BHD",
            "DATE:", "25/12/2018",
            "TOTAL", "AMOUNT", "RM", "99.50",
            "THANK", "YOU",
        ]
        tags = [
            "B-COMPANY", "I-COMPANY", "I-COMPANY", "I-COMPANY",
            "O", "B-DATE",
            "O", "O", "O", "B-TOTAL",
            "O", "O",
        ]
        probs = [0.95, 0.96, 0.94, 0.95, 0.99, 0.98, 0.99, 0.99, 0.99, 0.92, 0.99, 0.99]

        res = self.processor.aggregate_entities(tokens, tags, probs)

        self.assertEqual(res["company"].value, "BOOK XPRESS SDN BHD")
        self.assertAlmostEqual(res["company"].confidence, 0.95, places=2)

        self.assertEqual(res["date"].normalized_value, "2018-12-25")
        self.assertAlmostEqual(res["date"].confidence, 0.98, places=2)

        self.assertEqual(res["total"].normalized_value, "99.50")
        self.assertAlmostEqual(res["total"].confidence, 0.92, places=2)

        self.assertIsNone(res["address"].value)
        self.assertEqual(res["address"].confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
