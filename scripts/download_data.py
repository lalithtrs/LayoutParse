"""Script to acquire and structure the SROIE ICDAR 2019 Dataset.

Usage:
    python scripts/download_data.py [--source hf|mock] [--output-dir data/raw]
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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
            ([50, 140, 180, 155], "ITEM 1: BOOK 1"),
            ([220, 140, 280, 155], "25.00"),
            ([50, 170, 180, 185], "ITEM 2: PEN SET"),
            ([220, 170, 280, 185], "20.50"),
            ([50, 210, 120, 230], "TOTAL:"),
            ([220, 210, 280, 230], "RM 45.50"),
            ([50, 260, 250, 280], "THANK YOU PLEASE COME AGAIN"),
        ],
    },
    {
        "id": "sample_002",
        "company": "SUNSHINE GROCERY MART",
        "date": "14/01/2019",
        "address": "LOT 45 TAMAN MELAWATI SELANGOR",
        "total": "128.90",
        "lines": [
            ([40, 30, 250, 55], "SUNSHINE GROCERY MART"),
            ([40, 65, 320, 85], "LOT 45 TAMAN MELAWATI SELANGOR"),
            ([40, 100, 140, 115], "TAX INVOICE"),
            ([40, 130, 160, 145], "14/01/2019 14:32"),
            ([40, 170, 120, 185], "FRESH MILK"),
            ([240, 170, 290, 185], "12.50"),
            ([40, 200, 130, 215], "ORGANIC EGGS"),
            ([240, 200, 290, 215], "8.40"),
            ([40, 230, 160, 245], "OLIVE OIL 1L"),
            ([240, 230, 290, 245], "108.00"),
            ([40, 280, 150, 305], "SUBTOTAL: 128.90"),
            ([40, 320, 120, 345], "TOTAL:"),
            ([230, 320, 290, 345], "128.90"),
        ],
    },
]


def generate_sample_dataset(output_dir: Path):
    """Generate sample SROIE format files for development and tests."""
    images_dir = output_dir / "images"
    box_dir = output_dir / "box"
    key_dir = output_dir / "key"

    images_dir.mkdir(parents=True, exist_ok=True)
    box_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)

    for item in SAMPLE_RECEIPTS:
        item_id = item["id"]

        # 1. Write SROIE OCR text annotation: x1,y1,x2,y2,x3,y3,x4,y4,text
        box_file = box_dir / f"{item_id}.txt"
        with open(box_file, "w", encoding="utf-8") as f:
            for bbox, text in item["lines"]:
                x0, y0, x1, y1 = bbox
                # 4-point polygon: top-left, top-right, bottom-right, bottom-left
                line_str = f"{x0},{y0},{x1},{y0},{x1},{y1},{x0},{y1},{text}\n"
                f.write(line_str)

        # 2. Write SROIE Key fields JSON
        key_file = key_dir / f"{item_id}.json"
        ground_truth = {
            "company": item["company"],
            "date": item["date"],
            "address": item["address"],
            "total": item["total"],
        }
        with open(key_file, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=2)

    logger.info(f"Sample dataset generated successfully in {output_dir}")


def download_hf_dataset(hf_name: str, output_dir: Path):
    """Download SROIE from Hugging Face datasets hub."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("Hugging Face 'datasets' not installed. Falling back to sample generation.")
        generate_sample_dataset(output_dir)
        return

    logger.info(f"Downloading dataset '{hf_name}' from Hugging Face...")
    try:
        dataset = load_dataset(hf_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_dir / "sroie_hf"))
        logger.info(f"Dataset successfully saved to {output_dir / 'sroie_hf'}")
    except Exception as e:
        logger.error(f"Failed to download from Hugging Face: {e}")
        logger.info("Falling back to local sample generation.")
        generate_sample_dataset(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Download / Prepare SROIE Dataset")
    parser.add_argument("--source", choices=["hf", "mock"], default="mock", help="Download source")
    parser.add_argument("--hf-name", default="jsdnrs/ICDAR2019-SROIE", help="HF Dataset repository")
    parser.add_argument("--output-dir", default="data/raw", help="Target output directory")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    if args.source == "hf":
        download_hf_dataset(args.hf_name, out_path)
    else:
        generate_sample_dataset(out_path)


if __name__ == "__main__":
    main()
