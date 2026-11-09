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

            # Locate the best span for this entity:
            # exact contiguous match -> single-token substring -> fuzzy multi-token run.
            span = self._find_span(cleaned_tokens, val_tokens, occupied)
            if span is None:
                continue

            start_idx, end_idx = span
            tags[start_idx] = b_tag
            occupied[start_idx] = True
            for idx in range(start_idx + 1, end_idx):
                tags[idx] = i_tag
                occupied[idx] = True

        return tags

    def _find_span(
        self,
        cleaned_tokens: List[str],
        val_tokens: List[str],
        occupied: List[bool],
        gap: int = 1,
    ) -> Optional[Tuple[int, int]]:
        """Find the [start, end) token span that best matches an entity value.

        Multi-word fields (company, address) rarely appear as a perfectly contiguous
        OCR sub-sequence — addresses in particular span several lines and carry OCR
        noise. We therefore fall back from exact matching to a gap-tolerant run of the
        value's own word tokens, which is what lets company/address get labelled at all.
        """
        n = len(cleaned_tokens)

        # 1. Exact contiguous sub-sequence.
        for start_idx, end_idx in find_subsequence(cleaned_tokens, val_tokens):
            if not any(occupied[i] for i in range(start_idx, end_idx)):
                return (start_idx, end_idx)

        # 2. Single-token value: first free token that contains / equals it.
        if len(val_tokens) == 1:
            target = val_tokens[0]
            if len(target) < 2:
                return None
            for idx, c_token in enumerate(cleaned_tokens):
                if not occupied[idx] and c_token and (target == c_token or target in c_token or c_token in target):
                    return (idx, idx + 1)
            return None

        # 3. Fuzzy multi-token: longest gap-tolerant run of the value's word tokens.
        val_set = set(val_tokens)
        matched = [i for i in range(n) if not occupied[i] and cleaned_tokens[i] in val_set]
        if not matched:
            return None

        runs: List[List[int]] = []
        current = [matched[0]]
        for i in matched[1:]:
            if i - current[-1] - 1 <= gap:
                current.append(i)
            else:
                runs.append(current)
                current = [i]
        runs.append(current)

        def coverage(run: List[int]) -> int:
            return len({cleaned_tokens[i] for i in run} & val_set)

        best = max(runs, key=coverage)
        # Require at least half of the value's distinct tokens to be present.
        if coverage(best) < max(2, (len(val_set) + 1) // 2):
            return None

        start_idx, end_idx = best[0], best[-1] + 1
        if any(occupied[i] for i in range(start_idx, end_idx)):
            return None
        return (start_idx, end_idx)

    def tags_to_ids(self, tags: Sequence[str]) -> List[int]:
        """Convert list of BIO string labels to integer IDs."""
        return [self.label2id.get(tag, self.label2id.get("O", 0)) for tag in tags]

    def ids_to_tags(self, ids: Sequence[int]) -> List[str]:
        """Convert list of integer label IDs to BIO string labels."""
        return [self.id2label.get(int(i), "O") for i in ids]
