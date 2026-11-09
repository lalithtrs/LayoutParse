"""Post-processor for LayoutLMv3 token predictions.

Aggregates BIO-tagged tokens into structured fields, computes per-field
confidence metrics, and normalizes date and total values.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExtractedField:
    """Represents an extracted document field."""

    value: Optional[str]
    confidence: float
    raw_tokens: List[str] = field(default_factory=list)
    normalized_value: Optional[Any] = None

    def to_dict(self) -> dict:
        return {
            "value": self.normalized_value if self.normalized_value is not None else self.value,
            "raw_value": self.value,
            "confidence": round(self.confidence, 4),
        }


def parse_date(date_str: str) -> Optional[str]:
    """Normalize extracted date string to ISO-8601 YYYY-MM-DD format."""
    if not date_str:
        return None

    cleaned = re.sub(r"[^\w\s\-\/\.]", "", date_str).strip()

    formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%d/%m/%y",
        "%d-%m-%y",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%B %d %Y",
        "%d %b %y",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            # Basic sanity check on reasonable receipt years (2000 - 2035)
            if 2000 <= dt.year <= 2035:
                return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Regex search for DD/MM/YYYY or YYYY-MM-DD inside string
    match = re.search(r"(\d{2})[\/\-\.](\d{2})[\/\-\.](\d{2,4})", cleaned)
    if match:
        day, month, year = match.groups()
        if len(year) == 2:
            year = f"20{year}"
        try:
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    return date_str  # Return original if parsing unsuccessful


def parse_total(total_str: str) -> Optional[str]:
    """Clean monetary currency text and format to standard decimal string."""
    if not total_str:
        return None

    # Strip currency identifiers like RM, $, USD, EUR, etc.
    cleaned = re.sub(r"(?i)[^\d\.\,]", "", total_str).strip()

    # Match numeric amounts with decimals e.g. 1,234.50 or 45.50
    match = re.search(r"(\d+(?:[\,\.]\d{2,3})*(?:[\.\,]\d{2}))", cleaned)
    if match:
        raw_num = match.group(1)
        # Normalize European comma decimal if needed
        if "," in raw_num and "." not in raw_num:
            raw_num = raw_num.replace(",", ".")
        elif "," in raw_num and "." in raw_num:
            raw_num = raw_num.replace(",", "")  # Thousands separator
        try:
            val = float(raw_num)
            return f"{val:.2f}"
        except ValueError:
            pass

    # Fallback to any digits
    digits_only = re.search(r"\d+", cleaned)
    if digits_only:
        return digits_only.group(0)

    return total_str


class DocumentPostProcessor:
    """Converts token predictions and confidence scores into structured document fields."""

    def __init__(self, target_fields: Optional[List[str]] = None):
        self.target_fields = target_fields or ["company", "date", "address", "total"]

    def aggregate_entities(
        self,
        tokens: List[str],
        predicted_tags: List[str],
        probabilities: Optional[List[float]] = None,
    ) -> Dict[str, ExtractedField]:
        """Aggregate BIO-tagged tokens into field values with confidence calculation.

        Args:
            tokens: List of word tokens.
            predicted_tags: Corresponding BIO tags (e.g. 'B-COMPANY', 'I-COMPANY', 'O').
            probabilities: Softmax probability for each predicted tag (defaults to 1.0).

        Returns:
            Dict mapping field names to ExtractedField instances.
        """
        probs = probabilities if probabilities is not None else [1.0] * len(tokens)

        # Collect raw chunks per entity type: { 'COMPANY': [ (chunk_tokens, chunk_probs), ... ] }
        chunks: Dict[str, List[Tuple[List[str], List[float]]]] = {
            f.upper(): [] for f in self.target_fields
        }

        current_entity: Optional[str] = None
        current_tokens: List[str] = []
        current_probs: List[float] = []

        for token, tag, prob in zip(tokens, predicted_tags, probs):
            if tag.startswith("B-"):
                # Save previous entity if exists
                if current_entity and current_tokens:
                    chunks[current_entity].append((current_tokens, current_probs))

                current_entity = tag[2:]
                current_tokens = [token]
                current_probs = [prob]

            elif tag.startswith("I-") and current_entity == tag[2:]:
                current_tokens.append(token)
                current_probs.append(prob)

            else:
                if current_entity and current_tokens:
                    chunks[current_entity].append((current_tokens, current_probs))
                current_entity = None
                current_tokens = []
                current_probs = []

        # Catch tail entity
        if current_entity and current_tokens:
            chunks[current_entity].append((current_tokens, current_probs))

        # Select best chunk per field
        results: Dict[str, ExtractedField] = {}

        for field_name in self.target_fields:
            key = field_name.upper()
            candidate_chunks = chunks.get(key, [])

            if not candidate_chunks:
                results[field_name] = ExtractedField(
                    value=None,
                    confidence=0.0,
                    raw_tokens=[],
                    normalized_value=None,
                )
                continue

            # For address, concatenate chunks; for others, pick chunk with highest mean confidence
            if field_name == "address":
                all_tokens: List[str] = []
                all_probs: List[float] = []
                for c_toks, c_prb in candidate_chunks:
                    all_tokens.extend(c_toks)
                    all_probs.extend(c_prb)
                avg_conf = sum(all_probs) / max(1, len(all_probs))
                val = " ".join(all_tokens).strip()
                results[field_name] = ExtractedField(
                    value=val,
                    confidence=avg_conf,
                    raw_tokens=all_tokens,
                    normalized_value=val,
                )
            else:
                best_tokens, best_probs = max(
                    candidate_chunks,
                    key=lambda item: sum(item[1]) / max(1, len(item[1])),
                )
                avg_conf = sum(best_probs) / max(1, len(best_probs))
                raw_val = " ".join(best_tokens).strip()

                norm_val: Optional[str] = raw_val
                if field_name == "date":
                    norm_val = parse_date(raw_val)
                elif field_name == "total":
                    norm_val = parse_total(raw_val)

                results[field_name] = ExtractedField(
                    value=raw_val,
                    confidence=avg_conf,
                    raw_tokens=best_tokens,
                    normalized_value=norm_val,
                )

        return results
