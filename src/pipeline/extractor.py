"""End-to-end Invoice/Receipt Key Information Extraction Pipeline."""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.data.bio_tagger import DEFAULT_LABELS, ID2LABEL
from src.data.transforms import normalize_bbox
from src.models.postprocessor import DocumentPostProcessor, ExtractedField
from src.ocr.ocr_engine import BaseOCREngine, TokenBox
from src.ocr.paddle_ocr import MockOCREngine, PaddleOCREngine, SROIEPrecomputedOCREngine

logger = logging.getLogger(__name__)


class DocumentExtractor:
    """End-to-end orchestrator: Image -> OCR -> LayoutLMv3 -> Postprocessing -> JSON."""

    def __init__(
        self,
        model_path: Optional[str] = "models/best_layoutlmv3_sroie",
        ocr_engine_type: str = "mock",  # "paddle", "sroie_precomputed", "mock"
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.ocr_engine_type = ocr_engine_type
        self.postprocessor = DocumentPostProcessor()
        self.ocr_engine = self._initialize_ocr(ocr_engine_type)
        self.model = None
        self.processor = None
        self.device = device or "cpu"

        self._load_model_if_available()

    def _initialize_ocr(self, engine_type: str) -> BaseOCREngine:
        if engine_type == "paddle":
            return PaddleOCREngine()
        elif engine_type == "sroie_precomputed":
            return SROIEPrecomputedOCREngine()
        else:
            return MockOCREngine()

    def _load_model_if_available(self):
        """Load LayoutLMv3 weights if present on disk."""
        if not self.model_path or not os.path.exists(self.model_path):
            logger.info(
                f"Model weights not found at '{self.model_path}'. "
                "Pipeline running in fallback/heuristic mode until weights are placed."
            )
            return

        try:
            import torch
            from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = AutoProcessor.from_pretrained(self.model_path, apply_ocr=False)
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded LayoutLMv3 model from '{self.model_path}' onto {self.device}")
        except Exception as e:
            logger.warning(f"Could not load LayoutLMv3 model from {self.model_path}: {e}")

    def _heuristic_tagging(self, token_boxes: List[TokenBox]) -> Tuple[List[str], List[float]]:
        """Fallback rule-based tagger when neural weights are not yet present."""
        import re

        tags: List[str] = ["O"] * len(token_boxes)
        probs: List[float] = [0.85] * len(token_boxes)

        for idx, tb in enumerate(token_boxes):
            text = tb.text.upper()

            # Date pattern
            if re.search(r"\d{2}[\/\-\.]\d{2}[\/\-\.]\d{2,4}", text):
                tags[idx] = "B-DATE"
                probs[idx] = 0.95
            # Currency / Total pattern
            elif re.search(r"^\$?\d+\.\d{2}$", text) or (idx > 0 and "TOTAL" in token_boxes[idx - 1].text.upper()):
                tags[idx] = "B-TOTAL"
                probs[idx] = 0.92
            # First line often company
            elif idx == 0:
                tags[idx] = "B-COMPANY"
                probs[idx] = 0.90
            elif idx == 1 and tags[0] == "B-COMPANY":
                tags[idx] = "I-COMPANY"
                probs[idx] = 0.90
            # Address keyword
            elif any(k in text for k in ["JALAN", "STREET", "ROAD", "AVENUE", "LANE", "SUITE"]):
                tags[idx] = "B-ADDRESS"
                probs[idx] = 0.88

        return tags, probs

    def extract(
        self,
        image_input: Union[str, Path, Any],
        width: int = 1000,
        height: int = 1000,
    ) -> Dict[str, Any]:
        """Execute end-to-end extraction from input image."""
        start_time = time.perf_counter()

        # Step 1: Run OCR
        token_boxes = self.ocr_engine.extract_tokens(image_input)
        if not token_boxes:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return {
                "status": "success",
                "processing_time_ms": round(elapsed_ms, 2),
                "data": {
                    "company": {"value": None, "confidence": 0.0, "raw_value": None},
                    "date": {"value": None, "confidence": 0.0, "raw_value": None},
                    "address": {"value": None, "confidence": 0.0, "raw_value": None},
                    "total": {"value": None, "confidence": 0.0, "raw_value": None},
                },
                "overall_confidence": 0.0,
                "warnings": ["No text detected by OCR engine."],
            }

        words = [tb.text for tb in token_boxes]
        raw_boxes = [tb.bbox for tb in token_boxes]
        norm_boxes = [normalize_bbox(b, width, height) for b in raw_boxes]

        # Step 2: Token Classification (LayoutLMv3 or Heuristic)
        if self.model is not None and self.processor is not None:
            import torch

            # Prepare encoding for LayoutLMv3
            encoding = self.processor(
                text=words,
                boxes=norm_boxes,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512,
            )
            for k, v in encoding.items():
                encoding[k] = v.to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_ids = torch.argmax(probabilities, dim=-1).squeeze().tolist()
                max_probs = torch.max(probabilities, dim=-1).values.squeeze().tolist()

            # Align back to words (first len(words) tokens excluding special tokens)
            pred_tags = [ID2LABEL.get(pid, "O") for pid in predicted_ids[1 : 1 + len(words)]]
            conf_scores = max_probs[1 : 1 + len(words)]
        else:
            pred_tags, conf_scores = self._heuristic_tagging(token_boxes)

        # Step 3: Post-processing & Aggregation
        extracted_fields = self.postprocessor.aggregate_entities(
            tokens=words,
            predicted_tags=pred_tags,
            probabilities=conf_scores,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        data_dict = {k: v.to_dict() for k, v in extracted_fields.items()}
        confs = [v.confidence for v in extracted_fields.values() if v.value is not None]
        overall_conf = sum(confs) / len(confs) if confs else 0.0

        return {
            "status": "success",
            "processing_time_ms": round(elapsed_ms, 2),
            "data": data_dict,
            "overall_confidence": round(overall_conf, 4),
            "warnings": [],
        }
