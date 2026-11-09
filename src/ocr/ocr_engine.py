"""Abstract OCR Engine interface and data containers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Union


@dataclass
class TokenBox:
    """Represents a single OCR token with its bounding box and confidence."""

    text: str
    bbox: List[int]  # [x0, y0, x1, y1] in absolute pixel coordinates
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
        }


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def extract_tokens(self, image_input: Union[str, Path, Any]) -> List[TokenBox]:
        """Extract text tokens and bounding boxes from an image.

        Args:
            image_input: File path, numpy array, or PIL Image.

        Returns:
            List of TokenBox objects sorted in reading order.
        """
        pass
