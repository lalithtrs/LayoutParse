"""Fine-tune LayoutLMv3 for token classification (KIE) on the real SROIE dataset.

Learns all four fields (COMPANY, DATE, ADDRESS, TOTAL) using word-level tokens and
the robust BIO tagger, then saves the model AND processor so the inference pipeline
loads cleanly.

Usage (Colab GPU or local):
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --config configs/train_config.yaml --max-train 40 --epochs 1   # smoke test
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.bio_tagger import DEFAULT_LABELS, ID2LABEL, LABEL2ID  # noqa: E402
from src.data.prepare import build_prepare_fn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")


def build_compute_metrics(label_list):
    import evaluate
    import numpy as np

    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_list[pi] for (pi, li) in zip(pred, lab) if li != -100]
            for pred, lab in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[li] for (pi, li) in zip(pred, lab) if li != -100]
            for pred, lab in zip(predictions, labels)
        ]
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        flat = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "overall_f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        for entity in ("COMPANY", "DATE", "ADDRESS", "TOTAL"):
            if entity in results:
                flat[f"{entity.lower()}_f1"] = results[entity]["f1"]
        return flat

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Train LayoutLMv3 on SROIE")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--hf-name", default="jsdnrs/ICDAR2019-SROIE")
    parser.add_argument("--output-dir", default="models/best_layoutlmv3_sroie")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_train_epochs")
    parser.add_argument("--max-train", type=int, default=None, help="Cap train docs (smoke test)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_name = config["model"]["pretrained_model_name"]
    t = config["training"]

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoProcessor,
        LayoutLMv3ForTokenClassification,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(DEFAULT_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    logger.info(f"Loading dataset {args.hf_name} ...")
    ds = load_dataset(args.hf_name)
    if args.max_train:
        ds["train"] = ds["train"].select(range(min(args.max_train, len(ds["train"]))))

    prepare = build_prepare_fn(processor, max_length=config["dataset"].get("max_seq_length", 512)
                               if "dataset" in config else 512)
    logger.info("Tokenizing (word-level BIO) ...")
    encoded = ds.map(prepare, batched=True, batch_size=2, remove_columns=ds["train"].column_names)
    encoded.set_format("torch")

    targs = TrainingArguments(
        output_dir="checkpoints/layoutlmv3_sroie",
        learning_rate=float(t["learning_rate"]),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        num_train_epochs=args.epochs if args.epochs is not None else t["num_train_epochs"],
        weight_decay=t.get("weight_decay", 0.01),
        warmup_ratio=t.get("warmup_ratio", 0.1),
        fp16=bool(t.get("fp16", True)) and torch.cuda.is_available(),
        eval_strategy=t.get("eval_strategy", "epoch"),
        save_strategy=t.get("save_strategy", "epoch"),
        save_total_limit=t.get("save_total_limit", 1),
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        metric_for_best_model=t.get("metric_for_best_model", "overall_f1"),
        greater_is_better=t.get("greater_is_better", True),
        logging_steps=t.get("logging_steps", 25),
        report_to=[],
    )

    trainer_kwargs = dict(
        model=model,
        args=targs,
        train_dataset=encoded["train"],
        eval_dataset=encoded["test"],
        data_collator=default_data_collator,
        compute_metrics=build_compute_metrics(DEFAULT_LABELS),
    )
    if "processing_class" in Trainer.__init__.__code__.co_varnames:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor
    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    logger.info(f"Final metrics: {trainer.evaluate()}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    processor.save_pretrained(str(out))
    logger.info(f"Saved model + processor to {out}")


if __name__ == "__main__":
    main()
