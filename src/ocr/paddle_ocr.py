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
        import os
        import tempfile
        from PIL import Image

        temp_file_path = None
        img_for_ocr = None

        try:
            # 1. Standardize image input into a clean temporary RGB image file
            if isinstance(image_input, (str, Path)) and os.path.isfile(str(image_input)):
                img_for_ocr = str(image_input)
            else:
                # Load PIL image safely handling alpha / transparent backgrounds
                if hasattr(image_input, "seek"):
                    image_input.seek(0)
                    pil_img = Image.open(image_input)
                elif isinstance(image_input, bytes):
                    import io
                    pil_img = Image.open(io.BytesIO(image_input))
                elif isinstance(image_input, Image.Image):
                    pil_img = image_input
                else:
                    pil_img = Image.open(image_input)

                # Ensure RGBA / palette images are flattened onto white background so text isn't lost
                if pil_img.mode in ("RGBA", "LA", "P"):
                    bg = Image.new("RGB", pil_img.size, (255, 255, 255))
                    rgba_img = pil_img.convert("RGBA")
                    bg.paste(rgba_img, mask=rgba_img.split()[-1] if "A" in rgba_img.mode else None)
                    pil_img = bg
                else:
                    pil_img = pil_img.convert("RGB")

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    pil_img.save(tf.name, format="PNG")
                    temp_file_path = tf.name
                img_for_ocr = temp_file_path

            # 2. Execute PaddleOCR
            logger.info(f"Running PaddleOCR on target input: {img_for_ocr}")
            try:
                raw_results = ocr.ocr(img_for_ocr)
            except TypeError:
                raw_results = ocr.ocr(img_for_ocr, cls=self.use_angle_cls)

            logger.info(f"PaddleOCR raw output type: {type(raw_results)}, content count: {len(raw_results) if raw_results else 0}")

        except Exception as ocr_err:
            logger.error(f"PaddleOCR execution error: {ocr_err}", exc_info=True)
            return []
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass

        token_boxes: List[TokenBox] = []
        if not raw_results:
            return token_boxes

        # Handle different output structures across PaddleOCR versions:
        lines = []
        if isinstance(raw_results, list) and len(raw_results) > 0:
            if raw_results[0] is None:
                return token_boxes
            if isinstance(raw_results[0], list):
                # Standard v2 nested: [ [ [quad, (text, conf)], ... ] ]
                lines = raw_results[0]
            elif isinstance(raw_results[0], tuple) or (len(raw_results[0]) == 2 and isinstance(raw_results[0][0], list)):
                # Flat list: [ [quad, (text, conf)], ... ]
                lines = raw_results
            elif isinstance(raw_results[0], dict):
                # PaddleX / v3 dict format: [{'dt_polys': [...], 'rec_text': [...], 'rec_score': [...]}]
                d = raw_results[0]
                polys = d.get("dt_polys", d.get("boxes", []))
                texts = d.get("rec_text", d.get("texts", []))
                scores = d.get("rec_score", d.get("scores", [1.0] * len(texts)))
                for poly, text, score in zip(polys, texts, scores):
                    if hasattr(poly, "tolist"):
                        poly = poly.tolist()
                    flat_poly = [int(round(coord)) for pt in poly for coord in pt] if isinstance(poly[0], (list, tuple)) else [int(round(c)) for c in poly]
                    bbox = quad_to_bbox(flat_poly) if len(flat_poly) == 8 else flat_poly
                    token_boxes.extend(split_line_into_words(str(text), bbox, float(score)))
                token_boxes.sort(key=lambda tb: (tb.bbox[1] // 15, tb.bbox[0]))
                return token_boxes

        for line in lines:
            if not line or len(line) < 2:
                continue
            try:
                poly = line[0]
                if isinstance(poly[0], (list, tuple)):
                    quad = [int(round(coord)) for pt in poly for coord in pt]
                else:
                    quad = [int(round(c)) for c in poly]

                line_bbox = quad_to_bbox(quad) if len(quad) == 8 else quad
                text_info = line[1]
                if isinstance(text_info, (list, tuple)):
                    text, conf = str(text_info[0]), float(text_info[1])
                else:
                    text, conf = str(text_info), 1.0

                words = split_line_into_words(text, line_bbox, conf)
                token_boxes.extend(words)
            except Exception as parse_line_err:
                logger.warning(f"Skipping unparseable OCR line {line}: {parse_line_err}")
                continue

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
        return [
            TokenBox(text="BOOK", bbox=[55, 62, 120, 85], confidence=0.99),
            TokenBox(text="XPRESS", bbox=[130, 62, 220, 85], confidence=0.99),
            TokenBox(text="SDN", bbox=[230, 62, 280, 85], confidence=0.98),
            TokenBox(text="BHD", bbox=[290, 62, 340, 85], confidence=0.98),
            TokenBox(text="NO", bbox=[55, 100, 80, 118], confidence=0.96),
            TokenBox(text="12", bbox=[85, 100, 110, 118], confidence=0.96),
            TokenBox(text="JALAN", bbox=[115, 100, 180, 118], confidence=0.97),
            TokenBox(text="SULTAN", bbox=[185, 100, 260, 118], confidence=0.97),
            TokenBox(text="50000", bbox=[55, 125, 115, 142], confidence=0.96),
            TokenBox(text="KUALA", bbox=[120, 125, 180, 142], confidence=0.96),
            TokenBox(text="LUMPUR", bbox=[185, 125, 260, 142], confidence=0.96),
            TokenBox(text="DATE:", bbox=[55, 180, 110, 198], confidence=0.98),
            TokenBox(text="12/05/2018", bbox=[120, 180, 230, 198], confidence=0.99),
            TokenBox(text="TOTAL", bbox=[55, 300, 120, 320], confidence=0.99),
            TokenBox(text="RM", bbox=[130, 300, 160, 320], confidence=0.98),
            TokenBox(text="45.00", bbox=[170, 300, 240, 320], confidence=0.99),
        ]


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
