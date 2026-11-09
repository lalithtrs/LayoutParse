"""Unit tests for BIO sequence alignment and tag conversion."""

import unittest
from src.data.bio_tagger import BIOTagger


class TestBIOTagger(unittest.TestCase):
    def setUp(self):
        self.tagger = BIOTagger()

    def test_bio_tagger_exact_match(self):
        tokens = ["BOOK", "XPRESS", "SDN", "BHD", "DATE:", "12/05/2018", "TOTAL", "RM", "45.00"]
        ground_truth = {
            "company": "BOOK XPRESS SDN BHD",
            "date": "12/05/2018",
            "total": "45.00",
            "address": "NO 12 JALAN SULTAN",
        }

        tags = self.tagger.tag_tokens(tokens, ground_truth)

        expected = [
            "B-COMPANY",
            "I-COMPANY",
            "I-COMPANY",
            "I-COMPANY",
            "O",
            "B-DATE",
            "O",
            "O",
            "B-TOTAL",
        ]
        self.assertEqual(tags, expected)

    def test_bio_tagger_id_conversions(self):
        tags = ["O", "B-COMPANY", "I-COMPANY", "B-TOTAL"]
        ids = self.tagger.tags_to_ids(tags)
        recovered_tags = self.tagger.ids_to_tags(ids)
        self.assertEqual(recovered_tags, tags)

    def test_bio_tagger_no_matches(self):
        tokens = ["RANDOM", "RECEIPT", "HEADER"]
        ground_truth = {
            "company": "XYZ STORE",
            "total": "99.99",
        }
        tags = self.tagger.tag_tokens(tokens, ground_truth)
        self.assertEqual(tags, ["O", "O", "O"])


if __name__ == "__main__":
    unittest.main()
