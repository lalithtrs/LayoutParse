"""FastAPI Application for Document AI Invoice/Receipt Extraction."""

import io
import logging
import time
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from src.api.monitoring import telemetry
from src.api.schemas import ExtractionResponse, HealthResponse
from src.pipeline.extractor import DocumentExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api")

app = FastAPI(
    title="DocuExtract API",
    description="Document AI Key Information Extraction from Invoices & Receipts (LayoutLMv3 + PaddleOCR)",
    version="0.1.0",
)

# Enable CORS for web UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy extractor instance
extractor: Optional[DocumentExtractor] = None


def get_extractor(ocr_engine: str = "mock") -> DocumentExtractor:
    global extractor
    if extractor is None or extractor.ocr_engine_type != ocr_engine:
        extractor = DocumentExtractor(ocr_engine_type=ocr_engine)
    return extractor


@app.get("/health", response_model=HealthResponse)
def health():
    """System health and readiness check."""
    ext = get_extractor()
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=(ext.model is not None),
        ocr_engine=ext.ocr_engine_type,
        device=ext.device,
    )


@app.get("/telemetry")
def get_telemetry():
    """Retrieve runtime operational metrics and drift statistics."""
    return telemetry.get_summary()


@app.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(
    file: UploadFile = File(...),
    ocr_engine: str = Query("mock", pattern="^(mock|paddle|sroie_precomputed)$"),
):
    """Extract structured key information fields from an uploaded invoice image."""
    start_time = time.perf_counter()

    if not file.content_type.startswith("image/"):
        telemetry.record_request(latency_ms=0, overall_confidence=0.0, success=False)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload an image file (JPEG, PNG).",
        )

    try:
        contents = await file.read()
        image_bytes = io.BytesIO(contents)

        # Get image dimensions if PIL available
        width, height = 1000, 1000
        try:
            from PIL import Image

            with Image.open(image_bytes) as img:
                width, height = img.size
                image_bytes.seek(0)
        except Exception:
            pass

        pipeline = get_extractor(ocr_engine=ocr_engine)
        result = pipeline.extract(image_input=image_bytes, width=width, height=height)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        result["processing_time_ms"] = round(elapsed_ms, 2)

        # Record telemetry and append any drift warnings
        warnings = telemetry.record_request(
            latency_ms=elapsed_ms,
            overall_confidence=result["overall_confidence"],
            success=True,
        )
        result["warnings"].extend(warnings)

        return result

    except Exception as e:
        logger.error(f"Extraction error: {e}", exc_info=True)
        telemetry.record_request(latency_ms=0, overall_confidence=0.0, success=False)
        raise HTTPException(status_code=500, detail=f"Internal extraction error: {str(e)}")
