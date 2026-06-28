"""Data loading, cleaning, and tokenisation for Emotion Detection.

Dataset : dair-ai/emotion
Task    : Sequence Classification (6 classes)
Labels  : sadness(0), joy(1), love(2), anger(3), fear(4), surprise(5)
"""

import json
import pickle
import re

from datasets import load_dataset
from transformers import AutoTokenizer

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning  (designed for short social-media style text)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalise a raw text sample.

    Steps:
      1. Lower-case
      2. Remove URLs
      3. Remove @mentions and #hashtag symbols (keep the word)
      4. Remove non-ASCII characters
      5. Replace non-alphanumeric characters (keep space, apostrophe, period,
         exclamation mark, question mark — punctuation that carries sentiment)
      6. Collapse multiple whitespace to a single space
      7. Strip leading/trailing whitespace
    """
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove @mentions (keep nothing — they carry no emotion signal)
    text = re.sub(r"@\w+", "", text)
    # Remove # symbol but keep the hashtag word
    text = re.sub(r"#(\w+)", r"\1", text)
    # Remove non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode()
    # Keep letters, digits, and a small set of sentiment-bearing punctuation
    text = re.sub(r"[^a-z0-9\s'.,!?]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _apply_cleaning(batch):
    batch[config.TEXT_COLUMN] = [clean_text(t) for t in batch[config.TEXT_COLUMN]]
    return batch


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch[config.TEXT_COLUMN],
        truncation=True,
        padding="max_length",
        max_length=config.MAX_SEQ_LENGTH,
    )


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def load_and_prepare_data(force_rebuild: bool = False):
    """Load, clean, and tokenise the emotion dataset.

    Returns (train_dataset, val_dataset, test_dataset, id2label, label2id).
    Caches the processed result to PROCESSED_DATA_PATH to speed up reruns.
    """
    if not force_rebuild and config.PROCESSED_DATA_PATH.exists():
        logger.info("Loading cached processed data from %s", config.PROCESSED_DATA_PATH)
        with open(config.PROCESSED_DATA_PATH, "rb") as f:
            cache = pickle.load(f)
        return cache["train"], cache["val"], cache["test"], cache["id2label"], cache["label2id"]

    logger.info("Downloading dataset: %s", config.DATASET_NAME)
    raw = load_dataset(config.DATASET_NAME)

    train_raw = raw["train"].select(range(min(config.TRAIN_SIZE, len(raw["train"]))))
    val_raw = raw["validation"].select(range(min(config.VALIDATION_SIZE, len(raw["validation"]))))
    test_raw = raw["test"].select(range(min(config.TEST_SIZE, len(raw["test"]))))

    logger.info("Raw sizes — Train: %d | Val: %d | Test: %d",
                len(train_raw), len(val_raw), len(test_raw))

    # Clean text
    train_raw = train_raw.map(_apply_cleaning, batched=True)
    val_raw = val_raw.map(_apply_cleaning, batched=True)
    test_raw = test_raw.map(_apply_cleaning, batched=True)

    # Drop empty rows after cleaning
    train_raw = train_raw.filter(lambda ex: len(ex[config.TEXT_COLUMN].strip()) > 0)
    val_raw = val_raw.filter(lambda ex: len(ex[config.TEXT_COLUMN].strip()) > 0)
    test_raw = test_raw.filter(lambda ex: len(ex[config.TEXT_COLUMN].strip()) > 0)

    logger.info("After cleaning — Train: %d | Val: %d | Test: %d",
                len(train_raw), len(val_raw), len(test_raw))

    # Label maps (fixed by dataset definition)
    id2label = config.EMOTION_LABELS
    label2id = {v: k for k, v in id2label.items()}

    # Persist label mapping
    with open(config.ID2LABEL_PATH, "w") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)
    logger.info("id2label saved to %s", config.ID2LABEL_PATH)

    # Tokenise
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    def tok(batch):
        return tokenize_batch(batch, tokenizer)

    cols_to_remove = [c for c in train_raw.column_names if c != config.LABEL_COLUMN]
    train_ds = train_raw.map(tok, batched=True, remove_columns=cols_to_remove)
    val_ds = val_raw.map(tok, batched=True, remove_columns=cols_to_remove)
    test_ds = test_raw.map(tok, batched=True, remove_columns=cols_to_remove)

    # Rename label column to what Trainer expects
    if config.LABEL_COLUMN != "labels":
        train_ds = train_ds.rename_column(config.LABEL_COLUMN, "labels")
        val_ds = val_ds.rename_column(config.LABEL_COLUMN, "labels")
        test_ds = test_ds.rename_column(config.LABEL_COLUMN, "labels")

    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    # Cache
    with open(config.PROCESSED_DATA_PATH, "wb") as f:
        pickle.dump({
            "train": train_ds, "val": val_ds, "test": test_ds,
            "id2label": id2label, "label2id": label2id,
        }, f)
    logger.info("Processed data cached to %s", config.PROCESSED_DATA_PATH)

    return train_ds, val_ds, test_ds, id2label, label2id
