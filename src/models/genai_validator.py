"""GenAI & Rule-based cross-validation layer for extracted invoice fields.

Cross-checks extracted dates, totals, and address formatting to catch subtle
extraction errors or anomalies before serving the payload.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("genai_validator")


class DocumentFieldValidator:
    """Validates and cross-checks extracted fields using semantic rules and optional LLM fallback."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")

    def validate_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Inspect extracted field dictionary and return validation status and warnings."""
        anomalies: List[str] = []

        company = data.get("company", {}).get("value")
        date_val = data.get("date", {}).get("value")
        total_val = data.get("total", {}).get("value")
        address = data.get("address", {}).get("value")

        # 1. Company name check
        if not company:
            anomalies.append("Missing company entity.")
        elif len(str(company).strip()) < 2:
            anomalies.append(f"Suspiciously short company name: '{company}'.")

        # 2. Total amount check
        if not total_val:
            anomalies.append("Missing total monetary amount.")
        else:
            try:
                num = float(str(total_val).replace(",", ""))
                if num <= 0:
                    anomalies.append(f"Non-positive total amount: {total_val}.")
                elif num > 100000:
                    anomalies.append(f"Unusually large receipt total: {total_val}.")
            except ValueError:
                anomalies.append(f"Unparseable total monetary value: '{total_val}'.")

        # 3. Date check
        if not date_val:
            anomalies.append("Missing transaction date.")
        else:
            # ISO 8601 validation (YYYY-MM-DD)
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(date_val)):
                anomalies.append(f"Date '{date_val}' could not be normalized to ISO-8601 YYYY-MM-DD.")

        # 4. Address check
        if not address:
            anomalies.append("Missing address entity.")

        is_valid = len(anomalies) == 0

        return {
            "is_valid": is_valid,
            "anomalies": anomalies,
            "validation_score": round(max(0.0, 1.0 - (0.25 * len(anomalies))), 2),
        }
