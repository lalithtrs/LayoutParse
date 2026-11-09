"""Dataset loader and processor for ICDAR 2019 SROIE format."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.data.bio_tagger import BIOTagger
from src.data.transforms import normalize_bbox, quad_to_bbox

logger = logging.getLogger(__name__)


@dataclass
class SROIEDocument:
    """Represents a fully parsed SROIE receipt document."""

    doc_id: str
    tokens: List[str]
    normalized_boxes: List[List[int]]
    bio_tags: List[str]
    label_ids: List[int]
    ground_truth: Dict[str, str]
    image_path: Optional[str] = None
    width: int = 1000
    height: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.doc_id,
            "tokens": self.tokens,
            "bboxes": self.normalized_boxes,
            "ner_tags": self.label_ids,
            "bio_tags": self.bio_tags,
            "image_path": self.image_path,
        }


class SROIEDatasetLoader:
    """Loads raw SROIE directory structure (box/, key/, images/) into model-ready features."""

    def __init__(
        self,
        tagger: Optional[BIOTagger] = None,
        default_width: int = 1000,
        default_height: int = 1000,
    ):
        self.tagger = tagger or BIOTagger()
        self.default_width = default_width
        self.default_height = default_height

    def load_single_doc(
        self,
        box_file: Path,
        key_file: Path,
        image_file: Optional[Path] = None,
        image_dims: Optional[Tuple[int, int]] = None,
    ) -> Optional[SROIEDocument]:
        """Parse box annotations, ground truth keys, and perform BIO alignment."""
        if not box_file.exists() or not key_file.exists():
            return None

        # 1. Read ground truth key values
        with open(key_file, "r", encoding="utf-8", errors="ignore") as f:
            try:
                ground_truth = json.load(f)
            except json.JSONDecodeError:
                return None

        # 2. Determine image dimensions
        width, height = self.default_width, self.default_height
        if image_dims:
            width, height = image_dims
        elif image_file and image_file.exists():
            try:
                from PIL import Image

                with Image.open(image_file) as img:
                    width, height = img.size
            except Exception:
                pass

        # 3. Read box annotations
        tokens: List[str] = []
        raw_boxes: List[List[int]] = []

        with open(box_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 8)
                if len(parts) >= 9:
                    try:
                        quad = [int(p.strip()) for p in parts[:8]]
                        text = parts[8].strip()
                        if not text:
                            continue
                        bbox = quad_to_bbox(quad)
                        # Split line into words
                        words = text.split()
                        if not words:
                            continue
                        for word in words:
                            tokens.append(word)
                            raw_boxes.append(bbox)
                    except ValueError:
                        continue

        if not tokens:
            return None

        # 4. Normalize bounding boxes to [0, 1000]
        norm_boxes = [normalize_bbox(box, width, height) for box in raw_boxes]

        # 5. BIO Tagging
        bio_tags = self.tagger.tag_tokens(tokens, ground_truth)
        label_ids = self.tagger.tags_to_ids(bio_tags)

        return SROIEDocument(
            doc_id=box_file.stem,
            tokens=tokens,
            normalized_boxes=norm_boxes,
            bio_tags=bio_tags,
            label_ids=label_ids,
            ground_truth=ground_truth,
            image_path=str(image_file) if image_file else None,
            width=width,
            height=height,
        )

    def load_directory(self, base_dir: Union[str, Path]) -> List[SROIEDocument]:
        """Load all matching document pairs from base_dir."""
        path = Path(base_dir)
        box_dir = path / "box"
        key_dir = path / "key"
        img_dir = path / "images"

        documents: List[SROIEDocument] = []
        if not box_dir.exists():
            return documents

        for box_file in sorted(box_dir.glob("*.txt")):
            doc_id = box_file.stem
            key_file = key_dir / f"{doc_id}.json"
            img_file = img_dir / f"{doc_id}.jpg"
            if not img_file.exists():
                img_file = img_dir / f"{doc_id}.png"

            doc = self.load_single_doc(
                box_file=box_file,
                key_file=key_file,
                image_file=img_file if img_file.exists() else None,
            )
            if doc:
                documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from {base_dir}")
        return documents
