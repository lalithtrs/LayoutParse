"""CLI entrypoint for Document AI Invoice/Receipt Extraction.

Usage:
    python extract.py <path_to_image> [--ocr-engine mock|paddle|sroie_precomputed] [--model-path models/best_layoutlmv3_sroie]
"""

import argparse
import json
import sys
from pathlib import Path

from src.pipeline.extractor import DocumentExtractor


def main():
    parser = argparse.ArgumentParser(
        description="DocuExtract: Extract structured fields from a receipt/invoice image"
    )
    parser.add_argument("image_path", help="Path to input invoice/receipt image")
    parser.add_argument(
        "--ocr-engine",
        default="mock",
        choices=["mock", "paddle", "sroie_precomputed"],
        help="OCR engine to use (default: mock)",
    )
    parser.add_argument(
        "--model-path",
        default="models/best_layoutlmv3_sroie",
        help="Path to fine-tuned LayoutLMv3 checkpoint",
    )
    args = parser.parse_args()

    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"Error: Image file '{img_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading pipeline (OCR: {args.ocr_engine})...")
    extractor = DocumentExtractor(
        model_path=args.model_path,
        ocr_engine_type=args.ocr_engine,
    )

    print(f"Extracting from '{img_path}'...")
    result = extractor.extract(image_input=str(img_path))

    print("\n" + "=" * 50)
    print("EXTRACTION RESULTS:")
    print("=" * 50)
    print(json.dumps(result, indent=2))
    print("=" * 50)


if __name__ == "__main__":
    main()
