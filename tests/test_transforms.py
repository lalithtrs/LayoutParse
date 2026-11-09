"""Unit tests for coordinate normalization and polygon transformations."""

import unittest
from src.data.transforms import denormalize_bbox, normalize_bbox, quad_to_bbox


class TestTransforms(unittest.TestCase):
    def test_normalize_bbox_standard(self):
        bbox = [100, 200, 300, 400]
        width, height = 1000, 2000
        norm = normalize_bbox(bbox, width, height)
        self.assertEqual(norm, [100, 100, 300, 200])

    def test_normalize_bbox_clamping(self):
        # Boundary exceeding coordinates are clamped to [0, 1000]
        bbox = [-50, -10, 1200, 2500]
        width, height = 1000, 2000
        norm = normalize_bbox(bbox, width, height)
        self.assertEqual(norm, [0, 0, 1000, 1000])

    def test_normalize_bbox_inverted_coords(self):
        # Inverted x0 > x1 and y0 > y1 should be corrected
        bbox = [300, 400, 100, 200]
        width, height = 1000, 1000
        norm = normalize_bbox(bbox, width, height)
        self.assertEqual(norm, [100, 200, 300, 400])

    def test_normalize_bbox_invalid_inputs(self):
        with self.assertRaises(ValueError):
            normalize_bbox([10, 20, 30], 100, 100)

        with self.assertRaises(ValueError):
            normalize_bbox([10, 20, 30, 40], 0, 100)

    def test_denormalize_bbox(self):
        norm_bbox = [100, 100, 500, 500]
        width, height = 800, 1000
        denorm = denormalize_bbox(norm_bbox, width, height)
        self.assertEqual(denorm, [80, 100, 400, 500])

    def test_quad_to_bbox(self):
        # 8-point polygon [x1, y1, x2, y2, x3, y3, x4, y4]
        quad = [10, 15, 100, 12, 105, 50, 8, 48]
        bbox = quad_to_bbox(quad)
        self.assertEqual(bbox, [8, 12, 105, 50])


if __name__ == "__main__":
    unittest.main()
