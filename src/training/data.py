"""Data loading, cleaning, tokenisation, and label encoding for NER."""

import json
import pickle
import re
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_token(token: str) -> str:
    """Apply all cleaning rules to a single token string."""
    # Strip YAML front-matter artefacts  ---...---
    token = re.sub(r"^---.*?---$", "", token, flags=re.DOTALL)
    # Remove citation markers  [@...]
    token = re.sub(r"\[@[^\]]*\]", "", token)
    # Remove LaTeX commands  \cmd{...} or \cmd
    token = re.sub(r"\\[a-zA-Z]+(?:\{[^}]*\})?", "", token)
    # Replace non-alphanumeric characters (keep letters, digits, hyphens, apostrophes)
    token = re.sub(r"[^a-zA-Z0-9\-\']", " ", token)
    # Collapse multiple whitespace to single space and strip
    token = re.sub(r"\s+", " ", token).strip()
    return token


def _clean_example(tokens: list[str], ner_tags: list[str]) -> tuple[list[str], list[str]]:
    """Clean all tokens in an example and drop empty ones (keeping labels aligned)."""
    clean_tokens, clean_tags = [], []
    for tok, tag in zip(tokens, ner_tags):
        cleaned = _clean_token(tok)
        if cleaned:
            clean_tokens.append(cleaned)
            clean_tags.append(tag)
    return clean_tokens, clean_tags


# ---------------------------------------------------------------------------
# Label building
# ---------------------------------------------------------------------------

def build_label_maps(dataset_split, max_types: int = config.MAX_ENTITY_TYPES,
                     rare_threshold: int = config.RARE_ENTITY_THRESHOLD,
                     rare_policy: str = config.RARE_ENTITY_POLICY):
    """Derive label2id / id2label from the dataset, capping at *max_types* most frequent."""
    tag_counter: Counter = Counter()
    for example in dataset_split:
        tag_counter.update(example["ner_tags"])

    # Always keep the O tag
    top_tags = {tag for tag, cnt in tag_counter.most_common(max_types) if cnt >= rare_threshold}
    top_tags.add("O")

    # Sort deterministically (O first, then B- I- pairs alphabetically)
    sorted_tags = sorted(top_tags, key=lambda t: (t != "O", t))
    label2id = {tag: idx for idx, tag in enumerate(sorted_tags)}
    id2label = {idx: tag for tag, idx in label2id.items()}
    return label2id, id2label


# ---------------------------------------------------------------------------
# Tokenisation with subword–word label alignment
# ---------------------------------------------------------------------------

def tokenize_and_align_labels(examples, tokenizer, label2id: dict):
    """HuggingFace tokenizer + label alignment using word_ids()."""
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512,
    )
    all_labels = []
    for i, label_seq in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        aligned = []
        for word_idx in word_ids:
            if word_idx is None:
                aligned.append(-100)
            elif word_idx != previous_word_idx:
                raw_tag = label_seq[word_idx]
                aligned.append(label2id.get(raw_tag, label2id.get("O", 0)))
            else:
                # For continuation subwords, propagate label but mask from loss
                aligned.append(-100)
            previous_word_idx = word_idx
        all_labels.append(aligned)
    tokenized["labels"] = all_labels
    return tokenized


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def load_and_prepare_data(force_rebuild: bool = False):
    """Load, clean, tokenise, and encode the NER dataset.

    Returns (train_dataset, val_dataset, label2id, id2label).
    Caches the processed result to PROCESSED_DATA_PATH to speed up reruns.
    """
    if not force_rebuild and config.PROCESSED_DATA_PATH.exists():
        logger.info("Loading cached processed data from %s", config.PROCESSED_DATA_PATH)
        with open(config.PROCESSED_DATA_PATH, "rb") as f:
            cache = pickle.load(f)
        return cache["train"], cache["val"], cache["label2id"], cache["id2label"]

    logger.info("Downloading dataset: %s", config.DATASET_NAME)
    raw = load_dataset(config.DATASET_NAME)

    train_raw = raw["train"].select(range(min(config.TRAIN_SIZE, len(raw["train"]))))
    val_raw = raw["validation"].select(range(min(config.VALIDATION_SIZE, len(raw["validation"]))))

    logger.info("Train size: %d | Val size: %d", len(train_raw), len(val_raw))

    # Clean tokens in-place
    def _apply_cleaning(example):
        tokens, tags = _clean_example(example["tokens"], example["ner_tags"])
        example["tokens"] = tokens
        example["ner_tags"] = tags
        return example

    train_raw = train_raw.map(_apply_cleaning)
    val_raw = val_raw.map(_apply_cleaning)

    # Filter rows with length mismatch (shouldn't happen after cleaning, but be safe)
    train_raw = train_raw.filter(lambda ex: len(ex["tokens"]) == len(ex["ner_tags"]) > 0)
    val_raw = val_raw.filter(lambda ex: len(ex["tokens"]) == len(ex["ner_tags"]) > 0)

    logger.info("After cleaning — Train: %d | Val: %d", len(train_raw), len(val_raw))

    # Build label vocabulary from training split
    label2id, id2label = build_label_maps(train_raw)
    logger.info("Label space size: %d", len(label2id))

    # Persist label mapping
    with open(config.ID2LABEL_PATH, "w") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)
    logger.info("id2label saved to %s", config.ID2LABEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    train_ds = train_raw.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer, label2id),
        batched=True,
        remove_columns=train_raw.column_names,
    )
    val_ds = val_raw.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer, label2id),
        batched=True,
        remove_columns=val_raw.column_names,
    )

    # Cache to disk
    with open(config.PROCESSED_DATA_PATH, "wb") as f:
        pickle.dump({"train": train_ds, "val": val_ds, "label2id": label2id, "id2label": id2label}, f)
    logger.info("Processed data cached to %s", config.PROCESSED_DATA_PATH)

    return train_ds, val_ds, label2id, id2label
