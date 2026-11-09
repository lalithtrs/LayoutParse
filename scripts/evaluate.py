"""Evaluation script computing entity F1, exact-match, and full-document accuracy.

Usage:
    python scripts/evaluate.py [--data-dir data/samples] [--ocr-engine mock|sroie_precomputed|paddle]
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import SROIEDatasetLoader
from src.pipeline.extractor import DocumentExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def clean_for_comparison(val: Any) -> str:
    """Normalize text for exact match comparison."""
    if val is None:
        return ""
    return re.sub(r"[^\w]", "", str(val)).lower()


def evaluate_pipeline(data_dir: str, ocr_engine: str = "mock", model_path: str = "models/best_layoutlmv3_sroie"):
    """Evaluate pipeline on dataset and print comprehensive metrics table."""
    data_path = Path(data_dir)
    loader = SROIEDatasetLoader()
    docs = loader.load_directory(data_path)

    if not docs:
        logger.warning(f"No valid SROIE document pairs found in {data_dir}.")
        return

    logger.info(f"Loaded {len(docs)} documents for evaluation.")

    extractor = DocumentExtractor(
        model_path=model_path,
        ocr_engine_type=ocr_engine,
    )

    field_names = ["company", "date", "address", "total"]
    field_stats = {
        f: {"tp": 0, "fp": 0, "fn": 0, "exact_matches": 0, "total": 0}
        for f in field_names
    }
    full_doc_matches = 0
    total_docs = len(docs)
    total_latency_ms = 0.0

    for doc in docs:
        box_file = data_path / "box" / f"{doc.doc_id}.txt"
        target_input = str(box_file) if ocr_engine == "sroie_precomputed" else (doc.image_path or str(box_file))

        res = extractor.extract(image_input=target_input)
        total_latency_ms += res["processing_time_ms"]
        pred_data = res["data"]
        gt = doc.ground_truth

        all_fields_matched = True

        for f in field_names:
            field_stats[f]["total"] += 1
            pred_item = pred_data.get(f, {})
            pred_val = pred_item.get("value")
            gt_val = gt.get(f)

            pred_norm = clean_for_comparison(pred_val)
            gt_norm = clean_for_comparison(gt_val)

            if pred_norm and gt_norm:
                if pred_norm == gt_norm:
                    field_stats[f]["tp"] += 1
                    field_stats[f]["exact_matches"] += 1
                elif pred_norm in gt_norm or gt_norm in pred_norm:
                    field_stats[f]["tp"] += 0.8
                    field_stats[f]["fp"] += 0.2
                    field_stats[f]["exact_matches"] += 0
                    all_fields_matched = False
                else:
                    field_stats[f]["fp"] += 1
                    all_fields_matched = False
            elif pred_norm and not gt_norm:
                field_stats[f]["fp"] += 1
                all_fields_matched = False
            elif not pred_norm and gt_norm:
                field_stats[f]["fn"] += 1
                all_fields_matched = False
            else:
                field_stats[f]["tp"] += 1
                field_stats[f]["exact_matches"] += 1

        if all_fields_matched:
            full_doc_matches += 1

    report: Dict[str, Any] = {
        "dataset_size": total_docs,
        "avg_latency_ms": round(total_latency_ms / max(1, total_docs), 2),
        "fields": {},
        "full_document_accuracy": round((full_doc_matches / max(1, total_docs)) * 100, 2),
    }

    print("\n" + "=" * 70)
    print("EVALUATION BENCHMARK REPORT:")
    print("=" * 70)
    print(f"{'Field':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Exact Match %':<14}")
    print("-" * 70)

    f1_scores = []
    for f in field_names:
        stats = field_stats[f]
        precision = stats["tp"] / max(1e-6, stats["tp"] + stats["fp"])
        recall = stats["tp"] / max(1e-6, stats["tp"] + stats["fn"])
        f1 = 2 * (precision * recall) / max(1e-6, precision + recall)
        exact_pct = (stats["exact_matches"] / max(1, stats["total"])) * 100
        f1_scores.append(f1)

        report["fields"][f] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "exact_match_pct": round(exact_pct, 2),
        }

        print(
            f"{f.capitalize():<15} | {precision * 100:>8.1f}% | {recall * 100:>8.1f}% | "
            f"{f1 * 100:>8.1f}% | {exact_pct:>12.1f}%"
        )

    macro_f1 = sum(f1_scores) / len(f1_scores)
    report["overall_macro_f1"] = round(macro_f1, 4)

    print("-" * 70)
    print(f"Overall Macro F1: {macro_f1 * 100:.1f}%")
    print(f"Full-Document Accuracy (All fields correct): {report['full_document_accuracy']}%")
    print(f"Average Inference Latency: {report['avg_latency_ms']} ms/doc")
    print("=" * 70 + "\n")

    out_file = Path("evaluation_report.json")
    with open(out_file, "w", encoding="utf-8") as out:
        json.dump(report, out, indent=2)
    logger.info(f"Report saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Document AI Pipeline")
    parser.add_argument("--data-dir", default="data/samples", help="Path to evaluation data directory")
    parser.add_argument("--ocr-engine", default="sroie_precomputed", choices=["mock", "sroie_precomputed", "paddle"])
    parser.add_argument("--model-path", default="models/best_layoutlmv3_sroie")
    args = parser.parse_args()

    evaluate_pipeline(
        data_dir=args.data_dir,
        ocr_engine=args.ocr_engine,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
