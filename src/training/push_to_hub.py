"""Push best model and tokenizer to Hugging Face Hub."""

import os
import wandb

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def push_model_to_hub(model, tokenizer):
    """Push model + tokenizer to HF Hub; record URL in W&B summary if active."""
    hf_token = os.getenv("HF_TOKEN", config.HF_TOKEN)
    repo_id = os.getenv("HF_REPO_ID", config.HF_REPO_ID)

    if not hf_token:
        logger.warning("HF_TOKEN not set — skipping Hub push.")
        return

    logger.info("Pushing model to Hub: %s", repo_id)
    model.push_to_hub(repo_id, token=hf_token)
    tokenizer.push_to_hub(repo_id, token=hf_token)
    hub_url = f"https://huggingface.co/{repo_id}"
    logger.info("Model available at: %s", hub_url)

    if wandb.run is not None:
        wandb.run.summary["huggingface_model"] = hub_url

    return hub_url
