"""Corrected Colab Training snippet for LayoutLMv3 on SROIE.

Fixes the root cause where doc_entities was searched for 'label'/'box'
keys (which don't exist in SROIE), causing 100% of tokens to be labeled 'O'.
"""

import re


def clean_text(text):
    """Normalize text by lowercasing and stripping non-alphanumeric chars."""
    return re.sub(r"[^\w\s]", "", str(text)).lower().strip()


def find_subsequence(sequence, target):
    """Find start and end index where target words appear consecutively in sequence."""
    matches = []
    if not target or not sequence or len(target) > len(sequence):
        return matches
    t_len = len(target)
    for i in range(len(sequence) - t_len + 1):
        if sequence[i : i + t_len] == target:
            matches.append((i, i + t_len))
    return matches


def assign_bio_tags(doc_words, doc_entities):
    """Align OCR words with SROIE ground truth field values (company, date, address, total)."""
    tags = ["O"] * len(doc_words)
    cleaned_tokens = [clean_text(w) for w in doc_words]
    occupied = [False] * len(doc_words)

    # Sort fields by length descending so longer entities (e.g. address) match before shorter substrings
    entity_items = [
        (k.upper(), str(v))
        for k, v in doc_entities.items()
        if v and k.lower() in ["company", "date", "address", "total"]
    ]
    entity_items.sort(key=lambda x: len(x[1]), reverse=True)

    for entity_name, entity_val in entity_items:
        val_tokens = [clean_text(w) for w in entity_val.split() if clean_text(w)]
        if not val_tokens:
            continue

        b_tag = f"B-{entity_name}"
        i_tag = f"I-{entity_name}"

        # Match exact word sequence
        matches = find_subsequence(cleaned_tokens, val_tokens)

        # Fallback for single-word entities (or partial match like total amount)
        if not matches and len(val_tokens) == 1:
            target_word = val_tokens[0]
            for idx, c_tok in enumerate(cleaned_tokens):
                if not occupied[idx] and (target_word == c_tok or (len(target_word) > 2 and target_word in c_tok)):
                    matches.append((idx, idx + 1))
                    break

        for start_idx, end_idx in matches:
            if any(occupied[idx] for idx in range(start_idx, end_idx)):
                continue
            tags[start_idx] = b_tag
            occupied[start_idx] = True
            for idx in range(start_idx + 1, end_idx):
                tags[idx] = i_tag
                occupied[idx] = True
            break

    return tags


def prepare_examples(examples):
    # Ensure all PIL images are converted to RGB mode to handle grayscale images correctly
    images = [img.convert("RGB") for img in examples["image"]]
    words = examples["words"]
    boxes = examples["bboxes"]
    entities_list = examples["entities"]

    # Normalize and clamp coordinates strictly between 0 and 1000
    clamped_boxes = []
    for doc_boxes in boxes:
        normalized_doc_boxes = []
        for box in doc_boxes:
            x1 = max(0, min(1000, int(box[0])))
            y1 = max(0, min(1000, int(box[1])))
            x2 = max(x1, min(1000, int(box[2])))
            y2 = max(y1, min(1000, int(box[3])))
            normalized_doc_boxes.append([x1, y1, x2, y2])
        clamped_boxes.append(normalized_doc_boxes)

    batch_word_labels = []
    for batch_idx in range(len(words)):
        doc_words = words[batch_idx]
        doc_entities = entities_list[batch_idx]

        # Robust text-based BIO alignment directly against SROIE entity strings
        word_labels = assign_bio_tags(doc_words, doc_entities)

        # Convert word-level string labels to category IDs
        word_label_ids = [label2id.get(l, 0) for l in word_labels]
        batch_word_labels.append(word_label_ids)

    encoding = processor(
        images,
        words,
        boxes=clamped_boxes,
        word_labels=batch_word_labels,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    return encoding
