"""Bounding box coordinate transformation and normalization utilities.

LayoutLM models require bounding boxes normalized to an integer scale of 0 to 1000
relative to the original image dimensions [width, height].
"""

from typing import List, Sequence, Tuple, Union


def normalize_bbox(
    bbox: Sequence[Union[int, float]],
    width: int,
    height: int,
    scale: int = 1000,
) -> List[int]:
    """Normalize a bounding box [x0, y0, x1, y1] to the [0, scale] coordinate system.

    Args:
        bbox: 4-element sequence [x0, y0, x1, y1] in pixel space.
        width: Image width in pixels.
        height: Image height in pixels.
        scale: Target coordinate scale (default: 1000 for LayoutLM).

    Returns:
        List of 4 integers [norm_x0, norm_y0, norm_x1, norm_y1] clamped between 0 and scale.
    """
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox of length 4 [x0, y0, x1, y1], got {len(bbox)}")

    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got width={width}, height={height}")

    x0, y0, x1, y1 = bbox

    # Correct inverted coordinates if detected
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0

    norm_x0 = int(round(1000 * (float(x0) / width)))
    norm_y0 = int(round(1000 * (float(y0) / height)))
    norm_x1 = int(round(1000 * (float(x1) / width)))
    norm_y1 = int(round(1000 * (float(y1) / height)))

    # Strictly clamp values inside [0, scale]
    norm_x0 = max(0, min(scale, norm_x0))
    norm_y0 = max(0, min(scale, norm_y0))
    norm_x1 = max(norm_x0, min(scale, norm_x1))
    norm_y1 = max(norm_y0, min(scale, norm_y1))

    return [norm_x0, norm_y0, norm_x1, norm_y1]


def denormalize_bbox(
    norm_bbox: Sequence[Union[int, float]],
    width: int,
    height: int,
    scale: int = 1000,
) -> List[int]:
    """Convert normalized [0, scale] bounding box back to original image pixel coordinates.

    Args:
        norm_bbox: 4-element sequence [x0, y0, x1, y1] scaled to [0, scale].
        width: Original image width in pixels.
        height: Original image height in pixels.
        scale: Coordinate scale used during normalization (default: 1000).

    Returns:
        List of 4 integers [x0, y0, x1, y1] in pixel space.
    """
    if len(norm_bbox) != 4:
        raise ValueError(f"Expected norm_bbox of length 4, got {len(norm_bbox)}")

    x0 = int(round((norm_bbox[0] / scale) * width))
    y0 = int(round((norm_bbox[1] / scale) * height))
    x1 = int(round((norm_bbox[2] / scale) * width))
    y1 = int(round((norm_bbox[3] / scale) * height))

    return [x0, y0, x1, y1]


def quad_to_bbox(quad: Sequence[Union[int, float]]) -> List[int]:
    """Convert an 8-point polygon [x1, y1, x2, y2, x3, y3, x4, y4] to [x0, y0, x1, y1] bbox.

    Many OCR tools (like ICDAR SROIE raw annotations) provide 4 corner vertices.
    """
    if len(quad) != 8:
        raise ValueError(f"Expected 8 coordinates for 4-point polygon, got {len(quad)}")

    xs = [quad[0], quad[2], quad[4], quad[6]]
    ys = [quad[1], quad[3], quad[5], quad[7]]

    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
