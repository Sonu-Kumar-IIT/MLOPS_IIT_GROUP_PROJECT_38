"""Run NER inference by loading the fine-tuned model directly from Hugging Face Hub.

Usage:
    HF_TOKEN=<token> INPUT_TEXT="Google launched a new model in California." \\
        python -m src.inference.infrence_from_hub

Environment variables:
    HF_MODEL_NAME   — HF repo id  (default: value in config.py)
    HF_TOKEN        — HF access token (required for private repos)
    INPUT_TEXT      — Text to run NER on (required)
"""

import json
import os
import sys

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def run_inference(input_text: str, model_name: str | None = None, hf_token: str | None = None):
    """Download model from Hub and return token-label pairs."""
    model_name = model_name or os.getenv("HF_MODEL_NAME", config.HF_REPO_ID)
    hf_token = hf_token or os.getenv("HF_TOKEN", config.HF_TOKEN) or None

    logger.info("Loading model from Hub: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForTokenClassification.from_pretrained(model_name, token=hf_token)

    logger.info("Model loaded. Running prediction …")
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

    logger.info("Input text: %s", input_text)
    results = run_inference(input_text)

    output = {
        "model_name": os.getenv("HF_MODEL_NAME", config.HF_REPO_ID),
        "input_text": input_text,
        "token_labels": [
            {"token": r["word"], "label": r["entity_group"], "score": round(r["score"], 4)}
            for r in results
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
