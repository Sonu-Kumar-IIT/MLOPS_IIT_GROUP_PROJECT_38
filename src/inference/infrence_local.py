"""Run Emotion Detection inference from a locally saved model checkpoint.

Usage:
    INPUT_TEXT="I feel so happy today!" \\
        python -m src.inference.infrence_local

Environment variables:
    MODEL_CHECKPOINT_DIR  -- local model dir (default: config.MODEL_CHECKPOINT_DIR)
    INPUT_TEXT            -- Text to classify (required)
"""

import json
import os
import sys
from pathlib import Path

from transformers import pipeline

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def find_best_checkpoint(checkpoint_dir):
    candidates = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return candidates[-1]


def run_inference_local(input_text, model_dir=None):
    model_dir = Path(model_dir) if model_dir else config.MODEL_CHECKPOINT_DIR
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    checkpoint_path = model_dir
    if not (model_dir / "config.json").exists():
        checkpoint_path = find_best_checkpoint(model_dir)
    logger.info("Loading local model from: %s", checkpoint_path)
    classifier = pipeline(
        "text-classification",
        model=str(checkpoint_path),
        tokenizer=str(checkpoint_path),
        top_k=None,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
    )
    return classifier(input_text)


def main():
    input_text = os.getenv("INPUT_TEXT", "").strip()
    if not input_text:
        logger.error("INPUT_TEXT environment variable is not set or empty.")
        sys.exit(1)
    model_dir_env = os.getenv("MODEL_CHECKPOINT_DIR")
    model_dir = Path(model_dir_env) if model_dir_env else config.MODEL_CHECKPOINT_DIR
    logger.info("Input text: %s", input_text)
    results = run_inference_local(input_text, model_dir)
    preds = results[0] if isinstance(results[0], list) else results
    preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)
    output = {
        "model_dir": str(model_dir),
        "input_text": input_text,
        "predicted_emotion": preds_sorted[0]["label"],
        "confidence": round(preds_sorted[0]["score"], 4),
        "all_scores": [{"emotion": p["label"], "score": round(p["score"], 4)} for p in preds_sorted],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
