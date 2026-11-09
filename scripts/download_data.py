"""Acquire and structure the SROIE ICDAR 2019 dataset into the repo's
box / key / images layout (word-level OCR boxes + key fields + receipt image).

Usage:
    # Full real test benchmark (347 receipts) from Hugging Face:
    python scripts/download_data.py --source hf --split test --output-dir data/test

    # A few developer samples committed to the repo:
    python scripts/download_data.py --source hf --split test --limit 5 --output-dir data/samples

    # Offline synthetic placeholders (no network, no real receipts):
    python scripts/download_data.py --source mock --output-dir data/samples

Output layout (consumed by src.data.dataset.SROIEDatasetLoader):
    <output-dir>/images/<id>.jpg   receipt image
    <output-dir>/box/<id>.txt      SROIE 8-point quad + text, one word per line
    <output-dir>/key/<id>.json     {company, date, address, total}
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FIELDS = ("company", "date", "address", "total")

SAMPLE_RECEIPTS = [
    {
        "id": "sample_001",
        "company": "BOOK XPRESS SDN BHD",
        "date": "25/12/2018",
        "address": "NO 12 JALAN SULTAN 50000 KUALA LUMPUR",
        "total": "45.50",
        "lines": [
            ([50, 40, 200, 60], "BOOK XPRESS SDN BHD"),
            ([50, 70, 300, 90], "NO 12 JALAN SULTAN 50000 KUALA LUMPUR"),
            ([50, 110, 150, 125], "DATE: 25/12/2018"),
            ([50, 210, 120, 230], "TOTAL:"),
            ([220, 210, 280, 230], "RM 45.50"),
        ],
    },
]


def _write_box_file(box_file: Path, lines: List) -> None:
    """lines: list of (bbox[x0,y0,x1,y1], text). Emits one SROIE quad line per word."""
    out: List[str] = []
    for bbox, text in lines:
        text = str(text).replace("\n", " ").strip()
        if not text:
            continue
        x0, y0, x1, y1 = [int(round(c)) for c in bbox]
        out.append(f"{x0},{y0},{x1},{y0},{x1},{y1},{x0},{y1},{text}")
    box_file.write_text("\n".join(out), encoding="utf-8")


def _write_doc(root: Path, doc_id: str, lines: List, gt: Dict[str, str], image=None) -> None:
    (root / "box").mkdir(parents=True, exist_ok=True)
    (root / "key").mkdir(parents=True, exist_ok=True)
    _write_box_file(root / "box" / f"{doc_id}.txt", lines)
    clean_gt = {k: (gt.get(k) or "") for k in FIELDS}
    (root / "key" / f"{doc_id}.json").write_text(json.dumps(clean_gt, indent=2), encoding="utf-8")
    if image is not None:
        (root / "images").mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(root / "images" / f"{doc_id}.jpg", format="JPEG", quality=92)


def generate_sample_dataset(output_dir: Path) -> None:
    """Synthetic placeholders for offline development / unit tests (not real receipts)."""
    for item in SAMPLE_RECEIPTS:
        _write_doc(output_dir, item["id"], item["lines"], {k: item[k] for k in FIELDS})
    logger.info(f"Synthetic sample dataset generated in {output_dir}")


def download_hf_dataset(hf_name: str, output_dir: Path, split: str, limit: Optional[int]) -> None:
    """Download real SROIE from Hugging Face and convert to box/key/images layout."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("Hugging Face 'datasets' not installed. Falling back to synthetic samples.")
        generate_sample_dataset(output_dir)
        return

    logger.info(f"Downloading '{hf_name}' split='{split}' from Hugging Face...")
    ds = load_dataset(hf_name, split=split)
    n = 0
    for i, row in enumerate(ds):
        if limit is not None and n >= limit:
            break
        doc_id = str(row.get("key") or f"doc_{i:04d}").strip() or f"doc_{i:04d}"
        words = row.get("words") or []
        bboxes = row.get("bboxes") or []
        entities = row.get("entities") or {}
        if not words or not bboxes:
            continue
        lines = list(zip(bboxes, words))
        _write_doc(output_dir, doc_id, lines, entities, image=row.get("image"))
        n += 1
    logger.info(f"Converted {n} real receipts into {output_dir} (images/, box/, key/).")


def main():
    parser = argparse.ArgumentParser(description="Download / prepare the SROIE dataset")
    parser.add_argument("--source", choices=["hf", "mock"], default="hf")
    parser.add_argument("--hf-name", default="jsdnrs/ICDAR2019-SROIE")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Max documents to convert")
    parser.add_argument("--output-dir", default="data/test")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    if args.source == "hf":
        download_hf_dataset(args.hf_name, out_path, args.split, args.limit)
    else:
        generate_sample_dataset(out_path)


if __name__ == "__main__":
    main()
