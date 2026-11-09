"""Convert raw SROIE (Hugging Face jsdnrs/ICDAR2019-SROIE) rows into
LayoutLMv3 token-classification features.

This is the single source of truth for turning a receipt into
(word tokens, normalized boxes, BIO label ids) so that TRAINING and the
evaluation pipeline (src.data.dataset.SROIEDatasetLoader) tokenize receipts
the same way — word-level, not line-level. The line-vs-word mismatch is what
previously left COMPANY and ADDRESS unlabeled during training.
"""

from typing import Any, Dict, List, Tuple

from src.data.bio_tagger import BIOTagger
from src.data.transforms import normalize_bbox

FIELDS = ["company", "date", "address", "total"]


def _dims(row: Dict[str, Any]) -> Tuple[int, int]:
    size = row.get("image_size") or {}
    width, height = size.get("width"), size.get("height")
    if not width or not height:
        width, height = row["image"].size
    return int(width), int(height)


def row_to_words_boxes(row: Dict[str, Any]) -> Tuple[List[str], List[List[int]]]:
    """Split each OCR line into word tokens, each inheriting its line's box
    (normalized to the LayoutLM [0, 1000] scale)."""
    width, height = _dims(row)
    tokens: List[str] = []
    norm_boxes: List[List[int]] = []
    for line, box in zip(row["words"], row["bboxes"]):
        text = str(line).replace("\n", " ").strip()
        if not text or not box or len(box) < 4:
            continue
        nb = normalize_bbox([int(box[0]), int(box[1]), int(box[2]), int(box[3])], width, height)
        for word in text.split():
            tokens.append(word)
            norm_boxes.append(nb)
    return tokens, norm_boxes


def row_to_example(row: Dict[str, Any], tagger: BIOTagger) -> Tuple[List[str], List[List[int]], List[int]]:
    tokens, norm_boxes = row_to_words_boxes(row)
    entities = row.get("entities") or {}
    gt = {k: (entities.get(k) or "") for k in FIELDS}
    bio_tags = tagger.tag_tokens(tokens, gt)
    label_ids = tagger.tags_to_ids(bio_tags)
    return tokens, norm_boxes, label_ids


def build_prepare_fn(processor, tagger: BIOTagger = None, max_length: int = 512):
    """Return a `datasets.map(batched=True)` function producing LayoutLMv3 features
    (input_ids, attention_mask, bbox, pixel_values, labels)."""
    tagger = tagger or BIOTagger()

    def prepare_examples(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        n = len(examples["image"])
        images, words, boxes, labels = [], [], [], []
        for i in range(n):
            row = {
                "words": examples["words"][i],
                "bboxes": examples["bboxes"][i],
                "entities": examples["entities"][i],
                "image_size": examples["image_size"][i] if "image_size" in examples else None,
                "image": examples["image"][i],
            }
            toks, nb, ids = row_to_example(row, tagger)
            images.append(examples["image"][i].convert("RGB"))
            words.append(toks)
            boxes.append(nb)
            labels.append(ids)

        encoding = processor(
            images=images,
            text=words,
            boxes=boxes,
            word_labels=labels,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        return encoding

    return prepare_examples
