"""Fine-tune LayoutLMv3 for Token Classification on SROIE Dataset.

Supports execution in Google Colab (GPU) or local/cloud environments with MLflow logging.

Usage:
    python scripts/train.py --config configs/train_config.yaml
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

import yaml
from src.data.bio_tagger import DEFAULT_LABELS, ID2LABEL, LABEL2ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(p, label_list: List[str]):
    """Compute entity-level precision, recall, and F1 via seqeval."""
    import evaluate
    import numpy as np

    seqeval = evaluate.load("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p_idx] for (p_idx, l_idx) in zip(prediction, label) if l_idx != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l_idx] for (p_idx, l_idx) in zip(prediction, label) if l_idx != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def train(config_path: str):
    """Execute LayoutLMv3 fine-tuning."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["pretrained_model_name"]
    train_args = config["training"]
    tracking_args = config["tracking"]

    logger.info(f"Loading pretrained model and processor: {model_name}")

    try:
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoProcessor,
            LayoutLMv3ForTokenClassification,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        logger.error(
            "Required deep learning libraries not found. Ensure torch, transformers, datasets are installed."
        )
        raise e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using compute device: {device}")

    processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(DEFAULT_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    # MLflow tracking initialization
    if tracking_args.get("tracker") == "mlflow":
        try:
            import mlflow

            mlflow.set_experiment(tracking_args.get("experiment_name", "layoutlmv3_sroie"))
            logger.info("MLflow tracking initialized.")
        except ImportError:
            logger.warning("MLflow not installed. Continuing without MLflow logging.")

    output_dir = "checkpoints/layoutlmv3_sroie"
    os.makedirs(output_dir, exist_ok=True)

    hf_training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=float(train_args["learning_rate"]),
        per_device_train_batch_size=train_args["per_device_train_batch_size"],
        per_device_eval_batch_size=train_args["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_args["gradient_accumulation_steps"],
        num_train_epochs=train_args["num_train_epochs"],
        weight_decay=train_args["weight_decay"],
        warmup_ratio=train_args["warmup_ratio"],
        fp16=train_args["fp16"] if torch.cuda.is_available() else False,
        eval_strategy=train_args["eval_strategy"],
        save_strategy=train_args["save_strategy"],
        save_total_limit=train_args["save_total_limit"],
        load_best_model_at_end=train_args["load_best_model_at_end"],
        metric_for_best_model=train_args["metric_for_best_model"],
        greater_is_better=train_args["greater_is_better"],
        logging_steps=train_args["logging_steps"],
        report_to=["mlflow"] if tracking_args.get("tracker") == "mlflow" else [],
    )

    logger.info("Training configuration complete. Ready to attach dataset and run Trainer.train().")
    return model, processor, hf_training_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LayoutLMv3 on SROIE")
    parser.add_argument("--config", default="configs/train_config.yaml", help="Path to YAML training config")
    args = parser.parse_args()
    train(args.config)
