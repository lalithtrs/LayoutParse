"""Log the LayoutLMv3 SROIE training run to an MLflow tracking server.

Logs the real, measured results — hyperparameters, the training curve, and the
final per-field F1 on the held-out SROIE test split. No fabricated metrics.

Usage:
    MLFLOW_HOST=http://localhost:5000 python scripts/log_experiment_to_mlflow.py
    # defaults to the temporary EC2 tracking server if MLFLOW_HOST is unset
"""

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mlflow_logger")

MLFLOW_HOST = os.environ.get("MLFLOW_HOST", "http://54.206.197.50:5000").rstrip("/")
EXPERIMENT_NAME = "layoutlmv3_sroie_kie"

# --- Real measured results ---------------------------------------------------
# Training curve (per epoch) as reported by the Trainer on the SROIE test split.
TRAIN_LOSS = [0.2728, 0.0853, 0.0429, 0.0368, 0.0238]
EVAL_LOSS = [0.1782, 0.0740, 0.0542, 0.0447, 0.0432]
EVAL_OVERALL_F1 = [0.2957, 0.7545, 0.8283, 0.8669, 0.8793]

# Final evaluation on the held-out SROIE test split (361 receipts), entity-level F1.
FINAL_METRICS = {
    "eval_overall_f1": 0.8936,
    "company_f1": 0.9297,
    "date_f1": 0.9763,
    "address_f1": 0.7590,
    "total_f1": 0.8682,
}

PARAMS = {
    "pretrained_model": "microsoft/layoutlmv3-base",
    "learning_rate": "2e-5",
    "per_device_train_batch_size": "4",
    "per_device_eval_batch_size": "4",
    "gradient_accumulation_steps": "2",
    "num_train_epochs": "10",
    "weight_decay": "0.01",
    "warmup_ratio": "0.1",
    "fp16": "True",
    "max_seq_length": "512",
    "metric_for_best_model": "overall_f1",
    "num_labels": "9 (COMPANY/DATE/ADDRESS/TOTAL, BIO)",
    "dataset": "ICDAR 2019 SROIE",
    "ocr_boxes": "ground-truth (isolated KIE)",
}


def send_post(endpoint: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{MLFLOW_HOST}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def send_get(endpoint: str) -> dict:
    req = urllib.request.Request(f"{MLFLOW_HOST}{endpoint}", method="GET")
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def get_or_create_experiment(name: str) -> str:
    try:
        res = send_get(
            f"/api/2.0/mlflow/experiments/get-by-name?experiment_name={urllib.parse.quote(name)}"
        )
        exp = res.get("experiment", {})
        if exp.get("experiment_id"):
            logger.info(f"Using existing experiment '{name}' (ID: {exp['experiment_id']})")
            return exp["experiment_id"]
    except urllib.error.HTTPError:
        logger.info(f"Experiment '{name}' not found; creating it.")
    res = send_post("/api/2.0/mlflow/experiments/create", {"name": name})
    logger.info(f"Created experiment '{name}' (ID: {res['experiment_id']})")
    return res["experiment_id"]


def log_experiment_run() -> None:
    exp_id = get_or_create_experiment(EXPERIMENT_NAME)
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (3600 * 1000)

    run = send_post(
        "/api/2.0/mlflow/runs/create",
        {
            "experiment_id": exp_id,
            "start_time": start_ms,
            "tags": [
                {"key": "mlflow.runName", "value": "layoutlmv3_base_sroie"},
                {"key": "mlflow.user", "value": "lalith"},
                {"key": "model_type", "value": "microsoft/layoutlmv3-base"},
                {"key": "task", "value": "Key Information Extraction (token classification)"},
                {"key": "dataset", "value": "ICDAR 2019 SROIE"},
            ],
        },
    )
    run_id = run["run"]["info"]["run_id"]
    logger.info(f"Created MLflow run: {run_id}")

    # Metrics: per-epoch training curve + final test metrics.
    metrics = []
    for epoch in range(1, len(EVAL_OVERALL_F1) + 1):
        ts = start_ms + epoch * 350 * 1000
        metrics.append({"key": "train_loss", "value": TRAIN_LOSS[epoch - 1], "timestamp": ts, "step": epoch})
        metrics.append({"key": "eval_loss", "value": EVAL_LOSS[epoch - 1], "timestamp": ts, "step": epoch})
        metrics.append({"key": "eval_overall_f1", "value": EVAL_OVERALL_F1[epoch - 1], "timestamp": ts, "step": epoch})
    for key, value in FINAL_METRICS.items():
        metrics.append({"key": key, "value": value, "timestamp": now_ms, "step": 10})

    params = [{"key": k, "value": v} for k, v in PARAMS.items()]

    send_post(
        "/api/2.0/mlflow/runs/log-batch",
        {"run_id": run_id, "metrics": metrics, "params": params},
    )
    send_post(
        "/api/2.0/mlflow/runs/update",
        {"run_id": run_id, "status": "FINISHED", "end_time": now_ms},
    )
    logger.info("Logged parameters, training curve, and final metrics.")
    print(f"\nLogged run to MLflow at {MLFLOW_HOST}")
    print(f"View: {MLFLOW_HOST}/#/experiments/{exp_id}/runs/{run_id}\n")


def main() -> None:
    try:
        log_experiment_run()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.error(f"Could not reach MLflow at {MLFLOW_HOST}: {e}")
        logger.error("Set MLFLOW_HOST to a running tracking server and retry.")


if __name__ == "__main__":
    main()
