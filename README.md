# DocuExtract: Document AI Invoice & Receipt Key Information Extraction

[![CI](https://github.com/lalithtrs/LayoutParse/actions/workflows/ci.yml/badge.svg)](https://github.com/lalithtrs/LayoutParse/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

> 🚀 **Live Production API (Swagger Docs):** [http://32.236.126.177:8000/docs](http://32.236.126.177:8000/docs)  
> 📊 **Live Telemetry & Drift Monitoring:** [http://32.236.126.177:8000/telemetry](http://32.236.126.177:8000/telemetry)  
> 📈 **MLflow Tracking Dashboard:** [http://32.236.126.177:5000](http://32.236.126.177:5000)

An end-to-end multimodal Document AI system for scanned invoice and receipt processing. Combines **PaddleOCR** for text localization/recognition with a fine-tuned **LayoutLMv3** (visual + spatial + textual token classification) to extract structured accounting data (`company`, `date`, `address`, `total`) into schema-validated JSON with confidence metrics and drift monitoring.

---

## 🎥 Live Interactive Demonstration

![DocuExtract Live Swagger Demo](docs/demo.gif)

> 📹 **Live Walkthrough:** Demonstrates real-time receipt inference on the AWS EC2 production deployment: uploading an invoice image, executing the combined OCR & LayoutLMv3 pipeline, and extracting structured key information with per-field confidence scores in under 5 seconds.

---

## 1. System Architecture

```text
  Scanned Receipt Image (JPG / PNG / PDF)
                    │
                    ▼
  [OCR Engine]  PaddleOCR (Text Localization + Angle Cls + Recognition)
                    │   Tokens + Bounding Boxes [x0, y0, x1, y1]
                    ▼
  [Feature Pipe] Coordinate Normalization [0-1000] + WordPiece Sub-token Alignment
                    │
                    ▼
  [Multimodal]  Fine-Tuned LayoutLMv3 (Visual Patches + Spatial Boxes + Text Tokens)
                    │   Logits / Softmax Predictions per Token
                    ▼
  [Post-Process] BIO Chunk Aggregation + Date (ISO) & Total Normalization + Confidence Scoring
                    │
                    ▼
  [FastAPI]     POST /extract  -->  Structured JSON + Confidence + Drift Telemetry
```

---

## 2. Benchmark Metrics (SROIE ICDAR 2019)

Evaluated on the unseen SROIE test benchmark (347 real receipt images):

| Field / Entity | Entity Precision | Entity Recall | Entity F1-Score | Exact Match % |
| :--- | :---: | :---: | :---: | :---: |
| **Company** | 93.1% | 91.7% | **92.4%** | 88.5% |
| **Date** | 96.2% | 95.4% | **95.8%** | 94.1% |
| **Address** | 86.2% | 83.1% | **84.6%** | 78.5% |
| **Total** | 90.5% | 89.1% | **89.8%** | 88.2% |
| **Overall Macro F1** | **91.5%** | **89.8%** | **90.6%** | — |
| **Full Document Accuracy (Strict)** | — | — | — | **76.2%** |

> **Full-Document Accuracy Definition:** A document is strictly scored as correct if and only if **all 4 extracted fields** match ground truth with 100% exact string equality (bounded above by the weakest field, Address, at 78.5% exact match). Under normalized string comparison (ignoring case, whitespace, and punctuation), document accuracy reaches **81.8%**.

### 2.1 Latency Breakdown (Stage-by-Stage Profiling)

Measured on the actual production deployment target (**AWS EC2 `t3.large`**, 2 vCPU, 8 GB RAM) compared to GPU:

| Pipeline Stage | EC2 `t3.large` (2 vCPU, Production) | GPU (NVIDIA T4, Benchmark) | Description |
| :--- | :---: | :---: | :--- |
| **OCR Text Detection (DBNet)** | ~1,850 ms | ~75 ms | PaddleOCR bounding box localization |
| **OCR Text Recognition (SVTR/CRNN)** | ~1,450 ms | ~45 ms | Text recognition across all detected lines |
| **LayoutLMv3 Model Forward Pass** | ~85 ms | ~20 ms | Multimodal token classification (512 tokens) |
| **Post-Processing & Validation** | ~5 ms | ~5 ms | BIO chunking, ISO date parsing, currency normalization |
| **Total End-to-End Latency** | **~3,390 ms (~3.4s)** | **~145 ms (0.15s)** | Full `/extract` request execution |

> **Note on Latency:** The neural LayoutLMv3 forward pass alone runs in **~85 ms on CPU**. Total request turnaround on EC2 is dominated by full-page CPU OCR detection/recognition (~3.3s), completing under 5 seconds end-to-end.

---

## 3. Engineering Decisions & End-to-End MLOps Lifecycle

This project was intentionally engineered to demonstrate senior-level ML system design, cost-conscious cloud architecture, and end-to-end MLOps governance rather than just a disconnected model script.

### 3.1 Why AWS EC2 (`t3.large`) for Serving? (Cost vs. Latency Optimization)
* **The Engineering Trade-Off:** While GPU instances (e.g. `g4dn.xlarge` at ~$0.53/hr ≈ $380/mo) provide sub-100ms inference, running dedicated cloud GPUs 24/7 for low-to-medium volume invoice processing is financially irresponsible.
* **Decoupled Compute Strategy:** We decoupled training from inference:
  * **Training (Heavy Compute):** Fine-tuned LayoutLMv3 on an accelerated GPU (Google Colab/Kaggle T4) in ~35 minutes.
  * **Serving (Cost-Efficient):** Deployed to an AWS EC2 `t3.large` instance (2 vCPU, 8 GB RAM at ~$0.083/hr ≈ $60/mo).
* **Container Optimization:** By substituting standard 7.5GB NVIDIA CUDA PyTorch wheels with targeted CPU wheels (`--index-url https://download.pytorch.org/whl/cpu`), we slashed the Docker image footprint from **8.5 GB down to 1.3 GB** and runtime memory usage to **<1.4 GB**, enabling an **~85 ms** LayoutLMv3 neural token classification pass on commodity CPUs (~3.4s full pipeline including OCR).

### 3.2 Why MLflow for Experiment Tracking & Model Governance?
* **Full Hyperparameter Lineage:** Logs complete experiment runs (`learning_rate: 2e-5`, `weight_decay: 0.01`, `warmup_ratio: 0.1`, `max_seq_length: 512`), training/eval loss curves across 10 epochs, and per-entity test metrics (`company_f1: 92.4%`, `date_f1: 95.8%`, `total_f1: 89.8%`, `address_f1: 84.6%`, `macro_f1: 90.6%`).
* **Auditability & Zero "Black Box" Deployments:** Rather than opaque `.safetensors` files, production artifacts are mapped back to exact run IDs (`33a94846ef4647c6b576e1a253f1fa3b`), ensuring complete auditability and reproducibility.
* **Isolated Architecture:** MLflow runs as an independent microservice container with SQLite and persistent Docker volume storage, decoupling model governance from live FastAPI request latency.

### 3.3 End-to-End MLOps Lifecycle Breakdown
```text
  [1. Ingestion]  -->  [2. Feature Eng.]  -->  [3. Training]  -->  [4. CI/CD & Test]  -->  [5. Serving]  -->  [6. Observability]
   SROIE ICDAR          Sub-sequence BIO        LayoutLMv3 GPU        GitHub Actions         FastAPI Docker        Real-Time Drift
   Dataset Loader       Tagging & Box Norm      + MLflow Tracking     Lint + 17 Unit Tests   AWS EC2 (t3.large)    Telemetry (/telemetry)
```

---

## 4. Repository Structure

```text
├── configs/
│   ├── base_config.yaml         # Global schema & BIO label mappings
│   ├── train_config.yaml        # LayoutLMv3 hyperparameters & MLflow tracking
│   └── infer_config.yaml        # OCR & normalization thresholds
├── data/
│   ├── samples/                 # Sample receipts & test annotations
│   ├── raw/                     # Raw SROIE dataset
│   └── processed/               # Tokenized splits
├── docker/
│   ├── Dockerfile               # Multi-stage production container
│   └── docker-compose.yml       # FastAPI + MLflow services
├── notebooks/
│   └── 01_train_layoutlmv3.ipynb       # One-click GPU training notebook for Colab
├── scripts/
│   ├── download_data.py         # SROIE acquisition script
│   └── train.py                 # PyTorch / HF Trainer pipeline
├── src/
│   ├── api/                     # FastAPI app, schemas, drift telemetry
│   ├── data/                    # Coordinate normalization & BIO tagging
│   ├── models/                  # LayoutLMv3 wrapper & postprocessor
│   ├── ocr/                     # PaddleOCR & SROIE OCR engines
│   └── pipeline/                # End-to-end DocumentExtractor
├── tests/                       # Unit & integration test suites
├── requirements.txt             # Production dependencies
└── pyproject.toml               # Tool configs (pytest, ruff, black)
```

---

## 5. Quickstart Guide

### 5.1 Local Setup

```bash
# Clone the repository
git clone https://github.com/lalithtrs/LayoutParse.git
cd LayoutParse

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Run Unit Tests

```bash
python3 -m unittest discover -s tests
# or with pytest:
pytest tests/
```

### 5.3 Training on Google Colab (GPU)

1. Open `notebooks/01_train_layoutlmv3.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Select Runtime: **GPU (T4)**.
3. Run all cells: It will automatically stream SROIE from Hugging Face, fine-tune `microsoft/layoutlmv3-base`, and download `best_layoutlmv3_sroie.zip`.
4. Unzip the downloaded checkpoint into `models/best_layoutlmv3_sroie/` in this repository.

### 5.4 Start the FastAPI Server

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs: `http://localhost:8000/docs`

---

## 6. API Reference

### Health Check
```bash
curl -X GET http://localhost:8000/health
```

### Extract Receipt Key Information
```bash
curl -X POST "http://localhost:8000/extract?ocr_engine=paddle" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/samples/sample_001.jpg"
```

**Response:**
```json
{
  "status": "success",
  "processing_time_ms": 3390.25,
  "data": {
    "company": {
      "value": "BOOK XPRESS SDN BHD",
      "raw_value": "BOOK XPRESS SDN BHD",
      "confidence": 0.965
    },
    "date": {
      "value": "2018-12-25",
      "raw_value": "25/12/2018",
      "confidence": 0.982
    },
    "address": {
      "value": "NO 12 JALAN SULTAN 50000 KUALA LUMPUR",
      "raw_value": "NO 12 JALAN SULTAN 50000 KUALA LUMPUR",
      "confidence": 0.912
    },
    "total": {
      "value": "45.50",
      "raw_value": "RM 45.50",
      "confidence": 0.941
    }
  },
  "overall_confidence": 0.950,
  "warnings": []
}
```

### Operational Telemetry & Drift Monitoring
```bash
curl -X GET http://localhost:8000/telemetry
```

---

## 7. Docker Deployment

```bash
# Build and run containerized service
docker compose -f docker/docker-compose.yml up --build -d

# Check container status
docker ps
```

---

## 8. License

This project is licensed under the Apache 2.0 License. The SROIE dataset is distributed under CC BY 4.0.
