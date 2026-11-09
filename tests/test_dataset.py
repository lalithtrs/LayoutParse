"""Unit tests for SROIE dataset parsing and document creation."""

import unittest
from pathlib import Path

from src.data.dataset import SROIEDatasetLoader


class TestSROIEDataset(unittest.TestCase):
    SAMPLE_DIR = Path("data/samples")

    def test_load_sample_directory(self):
        loader = SROIEDatasetLoader()
        docs = loader.load_directory(self.SAMPLE_DIR)

        # Count should match the committed sample pairs, whatever that number is.
        n_keys = len(list((self.SAMPLE_DIR / "key").glob("*.json")))
        self.assertGreater(len(docs), 0, "no sample documents were loaded")
        self.assertEqual(len(docs), n_keys)

        # Every document must have aligned tokens / boxes / tags.
        for doc in docs:
            self.assertEqual(len(doc.tokens), len(doc.normalized_boxes))
            self.assertEqual(len(doc.tokens), len(doc.bio_tags))
            self.assertTrue(doc.tokens, f"{doc.doc_id} produced no tokens")

    def test_bio_tags_cover_all_fields(self):
        """The robust tagger should label all four fields on a known real receipt."""
        loader = SROIEDatasetLoader()
        docs = {d.doc_id: d for d in loader.load_directory(self.SAMPLE_DIR)}

        target = "X00016469670"
        if target not in docs:
            self.skipTest(f"sample {target} not present")

        doc = docs[target]
        self.assertIn("OJC", doc.tokens)
        tagged = {t.split("-")[-1] for t in doc.bio_tags if t != "O"}
        for field in ("COMPANY", "DATE", "ADDRESS", "TOTAL"):
            self.assertIn(field, tagged, f"{field} not labeled on {target}")


if __name__ == "__main__":
    unittest.main()
