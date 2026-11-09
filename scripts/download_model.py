"""Download fine-tuned LayoutLMv3 SROIE model checkpoint from Hugging Face.

Usage:
    python scripts/download_model.py [--model-id darentang/layoutlmv3-base-finetuned-sroie] [--output-dir models/best_layoutlmv3_sroie]
"""

import argparse
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_model")

# Essential files needed for LayoutLMv3 inference
MODEL_FILES = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "model.safetensors",
]

FALLBACK_WEIGHTS = "pytorch_model.bin"


def download_file(url: str, dest_path: Path):
    """Download single file with progress reporting."""
    logger.info(f"Downloading: {url} -> {dest_path}")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        logger.info(f"Saved: {dest_path} ({len(data)} bytes)")
        return True
    except Exception as e:
        logger.warning(f"Could not download {url}: {e}")
        return False


def download_checkpoint(model_id: str, output_dir: Path):
    """Download model files from Hugging Face."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"https://huggingface.co/{model_id}/resolve/main"

    success_count = 0
    weights_downloaded = False

    for filename in MODEL_FILES:
        dest_file = output_dir / filename
        if dest_file.exists():
            logger.info(f"File already exists: {dest_file}")
            success_count += 1
            if filename in ("model.safetensors", "pytorch_model.bin"):
                weights_downloaded = True
            continue

        file_url = f"{base_url}/{filename}"
        if download_file(file_url, dest_file):
            success_count += 1
            if filename == "model.safetensors":
                weights_downloaded = True

    # If model.safetensors failed, try pytorch_model.bin
    if not weights_downloaded:
        dest_file = output_dir / FALLBACK_WEIGHTS
        if not dest_file.exists():
            logger.info("Attempting fallback to pytorch_model.bin...")
            file_url = f"{base_url}/{FALLBACK_WEIGHTS}"
            if download_file(file_url, dest_file):
                weights_downloaded = True

    if weights_downloaded:
        logger.info(f"Model checkpoint successfully downloaded to '{output_dir}'.")
    else:
        logger.warning(
            "Weights file could not be downloaded automatically (network restricted or missing file). "
            "Please check network connection or copy weights manually from Colab."
        )


def main():
    parser = argparse.ArgumentParser(description="Download LayoutLMv3 SROIE Checkpoint")
    parser.add_argument(
        "--model-id",
        default="darentang/layoutlmv3-base-finetuned-sroie",
        help="Hugging Face Model ID",
    )
    parser.add_argument(
        "--output-dir",
        default="models/best_layoutlmv3_sroie",
        help="Destination directory for model weights",
    )
    args = parser.parse_args()

    download_checkpoint(model_id=args.model_id, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
