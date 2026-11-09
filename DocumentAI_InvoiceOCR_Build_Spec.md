# Document AI — Invoice / Receipt OCR + Key Information Extraction
### Full end-to-end ML lifecycle · dataset → deployment · Intermediate

> Companion project to **VisionRAG (In Progress)**. This is your **"done & deployed"** flagship:
> it proves the full ML lifecycle + MLOps + cloud, and directly extends your Citriot
> invoice-extraction work (YOLO + PaddleOCR). Target: shippable in ~1–1.5 weeks.

---

## 1. Project summary

Upload a scanned receipt/invoice image → the system runs **OCR** to read the text, then a
fine-tuned **document-AI model (LayoutLMv3)** extracts the key fields (company, date, address,
total, line items) into clean structured JSON — served as a production API with experiment
tracking, Docker, CI/CD, cloud deployment, and monitoring.

**Business framing (use this in interviews):** "Automates invoice/receipt data entry — turns a
photo into structured accounting data, cutting manual entry effort." Ties straight to your
Citriot "~70% manual effort reduction" bullet.

## 2. Dataset details

### Primary: SROIE (ICDAR 2019) — recommended
- **What:** 1,000 real scanned receipt images.
- **Tasks:** (1) text localization, (2) OCR, (3) **Key Information Extraction** — this is the one you build.
- **Fields extracted:** `company`, `date`, `address`, `total`.
- **Format:** image + OCR text with bounding boxes + ground-truth field values.
- **Split:** ~626 train / ~347 test (commonly re-split 600/26/347 or similar — define your own val split).
- **License:** CC BY 4.0.
- **Get it (cleanest mirror):** `jsdnrs/ICDAR2019-SROIE` on Hugging Face (parquet), or `darentang/sroie`.
  - https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE

### Alternative / extension: CORD
- **What:** 1,000 receipts, split 800/100/100, richer labels.
- **Labels:** multi-level parsing, 8 superclasses (Store, Payment, Menu, Subtotal, Total, …) incl. **line items**.
- **Use it if:** you want to extract line-item tables (harder, more impressive) after SROIE works.
- **Get it:** `naver-clova-ix/cord-v2` on Hugging Face.
  - https://huggingface.co/datasets/naver-clova-ix/cord-v2

> **Plan:** Build on **SROIE first** (4 flat fields = fast win). If time allows, extend to **CORD**
> for line-item extraction as a "depth" upgrade and a stronger resume bullet.

## 3. Problem framing

This is **Key Information Extraction (KIE)** on visually-rich documents:
`image → OCR (text + boxes) → token classification (LayoutLMv3) → structured fields`.
It's framed as token/entity classification (BIO tagging over OCR tokens), so it's a real,
trainable ML task — not just calling an OCR API.

## 4. Architecture

```
 Receipt/invoice image
        │
        ▼
 [OCR]  PaddleOCR (or provided SROIE boxes)  →  tokens + bounding boxes
        │
        ▼
 [Feature build]  normalize boxes, align tokens ↔ labels (BIO), tokenizer
        │
        ▼
 [Model]  LayoutLMv3 token classification (fine-tuned)  →  per-token field labels
        │
        ▼
 [Post-process]  merge tokens per field → {company, date, address, total[, line_items]}
        │
   (optional) [GenAI validation]  LLM cross-checks/normalizes fields (ties your GenAI story)
        │
        ▼
 [API]  FastAPI  /extract (image → JSON), /health
        │
 [Cross-cutting]  W&B/MLflow · Docker · GitHub Actions · AWS EC2/EB · monitoring
```

## 5. Tech stack
- **OCR:** PaddleOCR (matches your experience) — or use SROIE's provided boxes to start faster.
- **Model:** LayoutLMv3 via Hugging Face `transformers` (token classification). PyTorch + HF `Trainer`.
- **Tracking:** Weights & Biases **or** MLflow (log runs, metrics, artifacts; register best model).
- **Serving:** FastAPI + Pydantic (`/extract`, `/health`). Pillow/OpenCV for image handling.
- **Packaging:** Docker, clean module layout, YAML config.
- **CI/CD:** GitHub Actions (lint → test → build image → deploy).
- **Cloud:** AWS EC2 or Elastic Beanstalk.
- **Monitoring:** request + confidence logging; drift check (Evidently or lightweight custom).
- **Optional GenAI:** OpenRouter/any LLM for a field-validation/normalization fallback.

## 6. Metrics to capture (your resume runs on numbers)
- **Model:** entity-level **F1 per field** (company/date/address/total) + overall; exact-match accuracy per field.
- **End-to-end:** full-document extraction accuracy (all fields correct on a doc).
- **Experiments:** number of runs compared + winning config/hyperparams.
- **Serving:** inference latency (ms/doc), API P95 latency, successful CI/CD deploy.
- **Monitoring:** low-confidence / drift flag rate on shifted inputs.

---

## 7. End-to-end lifecycle phases

### Phase 1 — Data acquisition & understanding (Day 1)
- Download SROIE; inspect images, OCR boxes, and ground-truth fields.
- EDA: field value patterns, image quality, class/field distribution, tricky cases (multi-line address, currency formats).
- Define train/val/test split; document the schema.
- **Deliverable:** EDA notebook + a documented dataset schema.

### Phase 2 — OCR + feature engineering (Day 2–3)
- Get tokens + boxes: start with **provided SROIE boxes** (fastest); add PaddleOCR as the "real" OCR path.
- Normalize bounding boxes to LayoutLM's 0–1000 scale; align tokens to field labels via **BIO tagging**.
- Build the Hugging Face `Dataset` (image + words + boxes + labels).
- **Deliverable:** a reproducible preprocessing script producing model-ready features.

### Phase 3 — Model training (Day 3–5)
- Fine-tune **LayoutLMv3** for token classification on SROIE (HF `Trainer`).
- Track everything in **W&B/MLflow**: hyperparams, loss, per-field F1; compare ≥3 runs.
- Register the best checkpoint in the model registry; version it.
- **Deliverable:** trained model + tracking dashboard + registered best model.

### Phase 4 — Evaluation & error analysis (Day 5)
- Compute entity-level F1 per field + exact-match + full-doc accuracy on the test set.
- Error analysis: which fields fail, why (OCR errors vs model errors). Show a few qualitative examples.
- *(Optional)* add an **LLM extraction baseline** and compare — a strong talking point and ties your GenAI story.
- **Deliverable:** evaluation report with metrics table + error analysis.

### Phase 5 — Inference pipeline + API (Day 5–6)
- Refactor into clean modules: `ocr.py`, `features.py`, `model.py`, `extract.py`, `api.py` (config-driven YAML).
- Unit tests: input schema, box normalization, output JSON shape.
- **FastAPI**: `/extract` (image upload → structured JSON + per-field confidence), `/health`.
- **Deliverable:** `python extract.py <image>` works; API runs locally end-to-end.

### Phase 6 — Containerize + CI/CD (Day 6–7)
- Dockerize (pin deps, `.dockerignore`, handle model weights).
- **GitHub Actions**: on push → lint + tests → build image → (on main) deploy.
- **Deliverable:** green CI pipeline; image builds automatically.

### Phase 7 — Cloud deploy + monitoring (Day 7–9)
- Deploy container to **AWS EC2 or Elastic Beanstalk**; expose a public `/extract` endpoint.
- Add request + confidence logging; a drift/low-confidence monitor (feed degraded images, show it flags them).
- Write the **README**: architecture diagram, how-to-run, metrics, screenshots, a 60-sec demo video/GIF.
- **Deliverable:** live endpoint + monitoring evidence + polished repo. **Done.**

### Phase 8 — Optional depth upgrades (later)
- Extend to **CORD** for **line-item table extraction**.
- Add the **GenAI validation layer** (LLM normalizes dates/totals, flags inconsistencies).
- Export to **ONNX** for faster inference (bonus edge/perf credibility).

## 8. Scope guardrails
- **Start with SROIE's 4 flat fields.** Don't begin with line items — ship the simple version first.
- **Use provided OCR boxes to start**, swap in PaddleOCR once the pipeline works end-to-end.
- **Deploy early** (even a stub endpoint) so cloud/Docker issues surface before the deadline.
- LayoutLMv3 fine-tunes fine on a single GPU / Colab — keep epochs modest; the lifecycle is the grade.
- Commit often with clean messages — recruiters read Git history.

## 9. Resume bullets this generates (fill in real numbers)

**DocuExtract — Document AI Invoice/Receipt Key-Information Extraction**
*LayoutLMv3 · PaddleOCR · Hugging Face · FastAPI · Docker · GitHub Actions · AWS · MLflow*
- Built an end-to-end document-AI pipeline extracting structured fields (company/date/total/line items)
  from scanned receipts: PaddleOCR + fine-tuned LayoutLMv3, achieving **F1 X** (entity-level) and
  **X% full-document accuracy** on the SROIE benchmark.
- Shipped as a Dockerized FastAPI service with per-field confidence scores; automated
  lint/test/build/deploy via GitHub Actions CI/CD to AWS (EC2/Elastic Beanstalk).
- Tracked N experiments in MLflow with a versioned model registry; added confidence/drift monitoring
  that flags low-quality inputs in production.

## 10. How it fits your job search
- **This project = proof of production discipline** (full lifecycle + MLOps + cloud) — finish & deploy it first.
- **VisionRAG = your "building now" differentiator** (CV + GenAI edge-to-cloud).
- On the resume, both reinforce your Citriot/Reciprocal experience, so your whole story is consistent.
