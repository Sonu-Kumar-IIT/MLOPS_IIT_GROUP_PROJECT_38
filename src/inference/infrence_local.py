"""Run NER inference from a locally saved model checkpoint.

Usage:
    INPUT_TEXT="Google launched a new model in California." \\
        python -m src.inference.infrence_local

Environment variables:
    MODEL_CHECKPOINT_DIR   — Path to local model directory (default: config.MODEL_CHECKPOINT_DIR)
    INPUT_TEXT             — Text to run NER on (required)
"""

import json
import os
import sys
from pathlib import Path

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """Return the best checkpoint directory inside *checkpoint_dir*."""
    candidates = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return candidates[-1]


def run_inference_local(input_text: str, model_dir: Path | None = None):
    """Load model from a local checkpoint and return token-label pairs."""
    model_dir = model_dir or config.MODEL_CHECKPOINT_DIR

    if not Path(model_dir).exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # If it looks like a checkpoint parent, navigate into the latest checkpoint
    checkpoint_path = model_dir
    if not (Path(model_dir) / "config.json").exists():
        checkpoint_path = find_best_checkpoint(Path(model_dir))

    logger.info("Loading local model from: %s", checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    model = AutoModelForTokenClassification.from_pretrained(str(checkpoint_path))

    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )
    results = ner_pipeline(input_text)
    return results


def main():
    input_text = os.getenv("INPUT_TEXT", "").strip()
    if not input_text:
        logger.error("INPUT_TEXT environment variable is not set or empty.")
        sys.exit(1)

    model_dir_env = os.getenv("MODEL_CHECKPOINT_DIR")
    model_dir = Path(model_dir_env) if model_dir_env else config.MODEL_CHECKPOINT_DIR

    logger.info("Input text: %s", input_text)
    results = run_inference_local(input_text, model_dir)

    output = {
        "model_dir": str(model_dir),
        "input_text": input_text,
        "token_labels": [
            {"token": r["word"], "label": r["entity_group"], "score": round(r["score"], 4)}
            for r in results
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
