"""PaddleOCR and Precomputed SROIE OCR implementations."""

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

from src.data.transforms import quad_to_bbox
from src.ocr.ocr_engine import BaseOCREngine, TokenBox

logger = logging.getLogger(__name__)


def split_line_into_words(line_text: str, line_bbox: List[int], confidence: float) -> List[TokenBox]:
    """Split a multi-word OCR line into individual word tokens with interpolated bounding boxes.

    LayoutLM performs best with word-level bounding boxes rather than entire sentence lines.
    """
    words = line_text.strip().split()
    if not words:
        return []

    if len(words) == 1:
        return [TokenBox(text=words[0], bbox=line_bbox, confidence=confidence)]

    x0, y0, x1, y1 = line_bbox
    total_chars = sum(len(w) for w in words)
    line_width = max(1, x1 - x0)

    word_boxes: List[TokenBox] = []
    current_x = x0

    for idx, word in enumerate(words):
        word_len = len(word)
        char_ratio = word_len / max(1, total_chars)
        estimated_width = int(round(line_width * char_ratio))

        w_x0 = current_x
        w_x1 = min(x1, current_x + estimated_width) if idx < len(words) - 1 else x1
        w_y0 = y0
        w_y1 = y1

        word_boxes.append(
            TokenBox(
                text=word,
                bbox=[w_x0, w_y0, max(w_x0 + 1, w_x1), max(w_y0 + 1, w_y1)],
                confidence=confidence,
            )
        )
        # Advance with spacing
        current_x = w_x1

    return word_boxes


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR integration with word-level interpolation."""

    def __init__(
        self,
        use_angle_cls: bool = True,
        lang: str = "en",
        show_log: bool = False,
    ):
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.show_log = show_log
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR

                try:
                    self._ocr = PaddleOCR(
                        use_angle_cls=self.use_angle_cls,
                        lang=self.lang,
                    )
                except Exception as ex:
                    logger.warning(f"PaddleOCR standard init failed ({ex}), trying basic initialization...")
                    self._ocr = PaddleOCR(lang=self.lang)
            except ImportError as err:
                raise ImportError(
                    "paddleocr is not installed. Run 'pip install paddleocr paddlepaddle' "
                    "or use MockOCREngine / SROIEPrecomputedOCREngine."
                ) from err
        return self._ocr

    def extract_tokens(self, image_input: Union[str, Path, Any]) -> List[TokenBox]:
        ocr = self._get_ocr()

        # Prepare image for PaddleOCR: supports file path, PIL Image, BytesIO, or numpy array
        img_for_ocr = image_input
        try:
            from PIL import Image
            import numpy as np

            if hasattr(image_input, "seek"):
                image_input.seek(0)
                pil_img = Image.open(image_input).convert("RGB")
                img_for_ocr = np.array(pil_img)[:, :, ::-1]  # RGB to BGR
                image_input.seek(0)
            elif isinstance(image_input, Image.Image):
                img_for_ocr = np.array(image_input.convert("RGB"))[:, :, ::-1]
            elif isinstance(image_input, bytes):
                import io
                pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
                img_for_ocr = np.array(pil_img)[:, :, ::-1]
            elif isinstance(image_input, (str, Path)):
                img_for_ocr = str(image_input)
        except Exception as prep_err:
            logger.warning(f"PaddleOCR input preprocessing fallback: {prep_err}")
            img_for_ocr = image_input

        # PaddleOCR returns: [ [ [ [x,y] * 4 ], (text, confidence) ], ... ]
        try:
            raw_results = ocr.ocr(img_for_ocr)
        except TypeError:
            raw_results = ocr.ocr(img_for_ocr, cls=self.use_angle_cls)
        except Exception as ocr_err:
            logger.error(f"PaddleOCR execution error: {ocr_err}")
            return []

        token_boxes: List[TokenBox] = []
        if not raw_results or not raw_results[0]:
            return token_boxes

        for line in raw_results[0]:
            if not line or len(line) < 2:
                continue
            try:
                quad = [coord for pt in line[0] for coord in pt]
                line_bbox = quad_to_bbox(quad)
                text, conf = str(line[1][0]), float(line[1][1])

                # Split line into words for token-level precision
                words = split_line_into_words(text, line_bbox, conf)
                token_boxes.extend(words)
            except Exception as parse_line_err:
                logger.warning(f"Skipping unparseable OCR line {line}: {parse_line_err}")
                continue

        # Sort reading order: top-to-bottom, left-to-right
        token_boxes.sort(key=lambda tb: (tb.bbox[1] // 15, tb.bbox[0]))
        return token_boxes


class SROIEPrecomputedOCREngine(BaseOCREngine):
    """Parses precomputed SROIE OCR text annotations."""

    def extract_from_file(self, txt_path: Union[str, Path]) -> List[TokenBox]:
        token_boxes: List[TokenBox] = []
        path = Path(txt_path)
        if not path.exists():
            return token_boxes

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 8)
                if len(parts) >= 9:
                    try:
                        quad = [int(p.strip()) for p in parts[:8]]
                        text = parts[8].strip()
                        line_bbox = quad_to_bbox(quad)
                        words = split_line_into_words(text, line_bbox, confidence=1.0)
                        token_boxes.extend(words)
                    except ValueError:
                        continue

        token_boxes.sort(key=lambda tb: (tb.bbox[1] // 15, tb.bbox[0]))
        return token_boxes

    def extract_tokens(self, image_input: Union[str, Path, Any]) -> List[TokenBox]:
        """Expects corresponding .txt file with matching stem, or falls back gracefully."""
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                return self.extract_from_file(txt_path)
            # Check data/samples/box if available
            sample_box = Path("data/samples/box") / f"{img_path.stem}.txt"
            if sample_box.exists():
                return self.extract_from_file(sample_box)

        logger.info("Precomputed OCR received in-memory upload or unmapped path. Using sample receipt OCR.")
        sample_txt = Path("data/samples/box/sample_001.txt")
        if sample_txt.exists():
            return self.extract_from_file(sample_txt)
        return MockOCREngine().extract_tokens(image_input)


class MockOCREngine(BaseOCREngine):
    """Mock OCR engine for deterministic unit testing and dry runs."""

    def __init__(self, predefined_tokens: Optional[List[TokenBox]] = None):
        self.tokens = predefined_tokens or [
            TokenBox(text="SUPERMARKET", bbox=[50, 40, 200, 60], confidence=0.99),
            TokenBox(text="12/03/2019", bbox=[50, 80, 150, 95], confidence=0.98),
            TokenBox(text="TOTAL", bbox=[50, 200, 100, 215], confidence=0.95),
            TokenBox(text="15.80", bbox=[120, 200, 180, 215], confidence=0.97),
        ]

    def extract_tokens(self, image_input: Union[str, Path, Any]) -> List[TokenBox]:
        return self.tokens
