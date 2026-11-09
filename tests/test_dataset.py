"""Unit tests for SROIE dataset parsing and document creation."""

import unittest
from pathlib import Path
from src.data.dataset import SROIEDatasetLoader


class TestSROIEDataset(unittest.TestCase):
    def test_load_sample_directory(self):
        loader = SROIEDatasetLoader()
        sample_dir = Path("data/samples")
        docs = loader.load_directory(sample_dir)

        self.assertEqual(len(docs), 2)
        doc1 = next(d for d in docs if d.doc_id == "sample_001")

        self.assertIn("BOOK", doc1.tokens)
        self.assertEqual(len(doc1.tokens), len(doc1.normalized_boxes))
        self.assertEqual(len(doc1.tokens), len(doc1.bio_tags))

        # Check that COMPANY tag is assigned
        self.assertIn("B-COMPANY", doc1.bio_tags)
        self.assertIn("B-TOTAL", doc1.bio_tags)


if __name__ == "__main__":
    unittest.main()
