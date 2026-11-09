"""BIO (Beginning, Inside, Outside) tagging and token alignment module.

Aligns OCR token sequences with key information ground-truth strings
(e.g., company, date, address, total) to create training labels for LayoutLMv3.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

# Standard SROIE BIO label schema
DEFAULT_LABELS = [
    "O",
    "B-COMPANY",
    "I-COMPANY",
    "B-DATE",
    "I-DATE",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-TOTAL",
    "I-TOTAL",
]

LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(DEFAULT_LABELS)}
ID2LABEL: Dict[int, str] = {i: label for i, label in enumerate(DEFAULT_LABELS)}


def clean_text(text: str) -> str:
    """Normalize text for robust matching by lowercasing and stripping non-alphanumeric chars."""
    if not text:
        return ""
    return re.sub(r"[^\w\s]", "", text).lower().strip()


def find_subsequence(sequence: List[str], target: List[str]) -> List[Tuple[int, int]]:
    """Find all start and end index tuples where target appears in sequence."""
    matches: List[Tuple[int, int]] = []
    if not target or not sequence or len(target) > len(sequence):
        return matches

    target_len = len(target)
    for i in range(len(sequence) - target_len + 1):
        if sequence[i : i + target_len] == target:
            matches.append((i, i + target_len))
    return matches


class BIOTagger:
    """Assigns BIO tags to OCR tokens given document ground-truth field values."""

    def __init__(
        self,
        label_list: Optional[List[str]] = None,
        entity_keys: Optional[List[str]] = None,
    ):
        self.labels = label_list or DEFAULT_LABELS
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.entity_keys = entity_keys or ["company", "date", "address", "total"]

    def tag_tokens(
        self,
        tokens: Sequence[str],
        ground_truth: Dict[str, str],
    ) -> List[str]:
        """Align OCR tokens against ground truth field values and return BIO tags.

        Args:
            tokens: List of extracted word strings in reading order.
            ground_truth: Dict mapping entity key (e.g. 'company') to its string value.

        Returns:
            List of BIO tag strings (e.g. ['B-COMPANY', 'I-COMPANY', 'O', ...]).
        """
        n_tokens = len(tokens)
        tags = ["O"] * n_tokens
        cleaned_tokens = [clean_text(t) for t in tokens]

        # Sort entity keys by length of value descending (longer matches prioritize)
        sorted_entities = sorted(
            [(k, ground_truth[k]) for k in self.entity_keys if k in ground_truth and ground_truth[k]],
            key=lambda x: len(x[1]),
            reverse=True,
        )

        occupied = [False] * n_tokens

        for key, val_str in sorted_entities:
            val_tokens = [clean_text(w) for w in val_str.split() if clean_text(w)]
            if not val_tokens:
                continue

            entity_name = key.upper()
            b_tag = f"B-{entity_name}"
            i_tag = f"I-{entity_name}"

            # Ensure tags exist in target vocabulary
            if b_tag not in self.label2id or i_tag not in self.label2id:
                continue

            # Exact token sub-sequence matching
            matches = find_subsequence(cleaned_tokens, val_tokens)

            # Fallback: if no direct multi-token match, try matching individual tokens or substrings
            if not matches and len(val_tokens) == 1:
                target_word = val_tokens[0]
                for idx, c_token in enumerate(cleaned_tokens):
                    if not occupied[idx] and (target_word == c_token or target_word in c_token):
                        matches.append((idx, idx + 1))
                        break

            for start_idx, end_idx in matches:
                # Check if span is already occupied
                if any(occupied[idx] for idx in range(start_idx, end_idx)):
                    continue

                tags[start_idx] = b_tag
                occupied[start_idx] = True
                for idx in range(start_idx + 1, end_idx):
                    tags[idx] = i_tag
                    occupied[idx] = True

                # Match found for this entity, avoid duplicate tagging for same entity
                break

        return tags

    def tags_to_ids(self, tags: Sequence[str]) -> List[int]:
        """Convert list of BIO string labels to integer IDs."""
        return [self.label2id.get(tag, self.label2id.get("O", 0)) for tag in tags]

    def ids_to_tags(self, ids: Sequence[int]) -> List[str]:
        """Convert list of integer label IDs to BIO string labels."""
        return [self.id2label.get(int(i), "O") for i in ids]
