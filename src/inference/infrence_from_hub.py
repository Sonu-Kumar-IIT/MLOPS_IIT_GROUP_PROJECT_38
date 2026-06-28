"""Run Emotion Detection inference by loading the fine-tuned model from Hugging Face Hub.

Usage:
    HF_TOKEN=<token> INPUT_TEXT="I feel so happy today!" \\
        python -m src.inference.infrence_from_hub

Environment variables:
    HF_MODEL_NAME   — HF repo id  (default: value in config.py)
    HF_TOKEN        — HF access token (required for private repos)
    INPUT_TEXT      — Text to classify (required)
"""

import json
import os
import sys

from transformers import pipeline

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def run_inference(input_text: str, model_name: str | None = None, hf_token: str | None = None):
    """Download model from Hub and return emotion prediction(s)."""
    model_name = model_name or os.getenv("HF_MODEL_NAME", config.HF_REPO_ID)
    hf_token = hf_token or os.getenv("HF_TOKEN", config.HF_TOKEN) or None

    logger.info("Loading model from Hub: %s", model_name)
    classifier = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        token=hf_token,
        top_k=None,       # return scores for all 6 emotion classes
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
    )

    logger.info("Running prediction …")
    results = classifier(input_text)
    return results


def main():
    input_text = os.getenv("INPUT_TEXT", "").strip()
    if not input_text:
        logger.error("INPUT_TEXT environment variable is not set or empty.")
        sys.exit(1)

    logger.info("Input text: %s", input_text)
    results = run_inference(input_text)

    # results is a list-of-lists when top_k=None
    predictions = results[0] if isinstance(results[0], list) else results
    predictions_sorted = sorted(predictions, key=lambda x: x["score"], reverse=True)

    output = {
        "model_name": os.getenv("HF_MODEL_NAME", config.HF_REPO_ID),
        "input_text": input_text,
        "predicted_emotion": predictions_sorted[0]["label"],
        "confidence": round(predictions_sorted[0]["score"], 4),
        "all_scores": [
            {"emotion": p["label"], "score": round(p["score"], 4)}
            for p in predictions_sorted
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
