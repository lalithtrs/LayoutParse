"""Pydantic schemas and data contracts for Document AI API."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class FieldResult(BaseModel):
    """Extraction result for an individual document field."""

    value: Optional[str] = Field(None, description="Normalized field value (e.g. ISO date, decimal total)")
    raw_value: Optional[str] = Field(None, description="Original unnormalized OCR text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")


class ExtractionData(BaseModel):
    """Structured fields extracted from the receipt."""

    company: FieldResult
    date: FieldResult
    address: FieldResult
    total: FieldResult


class ExtractionResponse(BaseModel):
    """Complete API response for /extract endpoint."""

    status: str = Field("success", description="Status string: success or error")
    processing_time_ms: float = Field(..., description="End-to-end processing time in milliseconds")
    data: ExtractionData = Field(..., description="Extracted key information fields")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Mean confidence across extracted fields")
    warnings: List[str] = Field(default_factory=list, description="Quality or drift warnings")


class HealthResponse(BaseModel):
    """System health check response."""

    status: str
    version: str
    model_loaded: bool
    ocr_engine: str
    device: str
