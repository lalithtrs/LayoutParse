"""Unpack and verify LayoutLMv3 weights exported from Google Colab.

Usage:
    python scripts/setup_weights.py [path_to_zip_or_dir]
"""

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("setup_weights")

TARGET_DIR = Path("models/best_layoutlmv3_sroie")
REQUIRED_FILES = [
    "config.json",
    "preprocessor_config.json",
]


def setup_weights(source_path: str):
    source = Path(source_path)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        # Check standard default locations
        download_candidate = Path.home() / "Downloads" / "best_layoutlmv3_sroie.zip"
        local_candidate = Path("best_layoutlmv3_sroie.zip")
        if local_candidate.exists():
            source = local_candidate
        elif download_candidate.exists():
            source = download_candidate
        else:
            logger.error(
                f"Source weights file '{source_path}' not found.\n"
                f"Please provide the path to your downloaded Colab zip, e.g.:\n"
                f"  python scripts/setup_weights.py ~/Downloads/best_layoutlmv3_sroie.zip"
            )
            sys.exit(1)

    logger.info(f"Extracting weights from: {source}")

    if source.resolve() == TARGET_DIR.resolve():
        logger.info("Source directory is already the target directory. Skipping copy.")
    elif source.suffix == ".zip":
        with zipfile.ZipFile(source, "r") as zf:
            # Check if contents are nested inside a top-level directory
            members = zf.namelist()
            top_level = {m.split("/")[0] for m in members if "/" in m}
            if len(top_level) == 1 and all(m.startswith(list(top_level)[0]) for m in members):
                prefix = list(top_level)[0] + "/"
                for m in members:
                    if m == prefix:
                        continue
                    extracted_rel = m[len(prefix) :]
                    if not extracted_rel:
                        continue
                    dest = TARGET_DIR / extracted_rel
                    if m.endswith("/"):
                        dest.mkdir(parents=True, exist_ok=True)
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(m) as src, open(dest, "wb") as dst:
                            dst.write(src.read())
            else:
                zf.extractall(TARGET_DIR)
    elif source.is_dir():
        for item in source.iterdir():
            dest = TARGET_DIR / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    # Verification
    present_files = os.listdir(TARGET_DIR)
    logger.info(f"Files unpacked into {TARGET_DIR}:\n  " + "\n  ".join(present_files))

    has_weights = any(
        f.endswith((".safetensors", ".bin", ".pt")) for f in present_files
    )
    has_config = (TARGET_DIR / "config.json").exists()

    if has_weights and has_config:
        logger.info("\n" + "=" * 60)
        logger.info("[SUCCESS] Model weights are verified and ready for inference!")
        logger.info("=" * 60)
        logger.info("You can now run:")
        logger.info("  python extract.py data/samples/box/sample_001.txt")
        logger.info("  uvicorn src.api.app:app --host 0.0.0.0 --port 8000")
    else:
        logger.warning(
            "Some expected model files (e.g. config.json, model.safetensors) may be missing in "
            f"{TARGET_DIR}. Check unpacked files above."
        )


def main():
    parser = argparse.ArgumentParser(description="Unpack and verify Colab weights")
    parser.add_argument(
        "source",
        nargs="?",
        default="best_layoutlmv3_sroie.zip",
        help="Path to downloaded best_layoutlmv3_sroie.zip or extracted folder",
    )
    args = parser.parse_args()
    setup_weights(args.source)


if __name__ == "__main__":
    main()
