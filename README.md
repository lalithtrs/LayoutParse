# DocuExtract: Document AI Invoice & Receipt Key Information Extraction

[![CI](https://github.com/lalithtrs/LayoutParse/actions/workflows/ci.yml/badge.svg)](https://github.com/lalithtrs/LayoutParse/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

> 🚀 **Live Production API (Swagger Docs):** [http://32.236.126.177:8000/docs](http://32.236.126.177:8000/docs)  
> 📊 **Live Telemetry & Drift Monitoring:** [http://32.236.126.177:8000/telemetry](http://32.236.126.177:8000/telemetry)  
> 📈 **MLflow Tracking Dashboard:** [http://32.236.126.177:5000](http://32.236.126.177:5000)

An end-to-end multimodal Document AI system for scanned invoice and receipt processing. Combines **PaddleOCR** for text localization/recognition with a fine-tuned **LayoutLMv3** (visual + spatial + textual token classification) to extract structured accounting data (`company`, `date`, `address`, `total`) into schema-validated JSON with confidence metrics and drift monitoring.

---

## 🎥 Live Interactive Demonstration

https://github.com/lalithtrs/LayoutParse/raw/main/docs/demo.webm

> 📹 **Live Walkthrough Video:** [**Click here to watch the full recording (`docs/demo.webm`)**](docs/demo.webm)  
> Demonstrates real-time receipt inference on the AWS EC2 production deployment: uploading an invoice image, executing the combined OCR & LayoutLMv3 pipeline, and extracting structured key information with per-field confidence scores in under 5 seconds.

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
| **Company** | 94.2% | 93.1% | **93.6%** | 89.4% |
| **Date** | 97.5% | 96.8% | **97.1%** | 95.2% |
| **Address** | 88.4% | 86.9% | **87.6%** | 78.5% |
| **Total** | 93.8% | 92.4% | **93.1%** | 91.0% |
| **Overall Macro F1** | **93.5%** | **92.3%** | **92.9%** | — |
| **Full Document Accuracy** | — | — | — | **81.8%** |

- **Inference Latency (GPU, NVIDIA T4):** ~210 ms / document
- **Inference Latency (CPU, Intel i5):** ~980 ms / document

---

## 3. Repository Structure

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
│   └── 01_colab_train_layoutlmv3.ipynb  # One-click GPU training notebook for Colab
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

## 4. Quickstart Guide

### 4.1 Local Setup

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

### 4.2 Run Unit Tests

```bash
python3 -m unittest discover -s tests
# or with pytest:
pytest tests/
```

### 4.3 Training on Google Colab (GPU)

1. Open `notebooks/01_colab_train_layoutlmv3.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Select Runtime: **GPU (T4)**.
3. Run all cells: It will automatically stream SROIE from Hugging Face, fine-tune `microsoft/layoutlmv3-base`, and download `best_layoutlmv3_sroie.zip`.
4. Unzip the downloaded checkpoint into `models/best_layoutlmv3_sroie/` in this repository.

### 4.4 Start the FastAPI Server

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs: `http://localhost:8000/docs`

---

## 5. API Reference

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
  "processing_time_ms": 284.5,
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

## 6. Docker Deployment

```bash
# Build and run containerized service
docker compose -f docker/docker-compose.yml up --build -d

# Check container status
docker ps
```

---

## 7. License

This project is licensed under the Apache 2.0 License. The SROIE dataset is distributed under CC BY 4.0.
