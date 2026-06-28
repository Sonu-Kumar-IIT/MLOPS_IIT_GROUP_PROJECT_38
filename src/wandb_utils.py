"""Weights & Biases initialisation helper."""

import os
import wandb
from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def init_wandb(run_name: str | None = None, extra_config: dict | None = None):
    """Initialise a W&B run.

    Environment variable WANDB_API_KEY must be set before calling this function.
    """
    api_key = os.getenv("WANDB_API_KEY", config.WANDB_API_KEY)
    if not api_key:
        logger.warning("WANDB_API_KEY is not set — W&B logging will be disabled.")
        os.environ["WANDB_DISABLED"] = "true"
        return None

    wandb.login(key=api_key)

    run_config = {
        "model_name": config.MODEL_NAME,
        "dataset": config.DATASET_NAME,
        "task": "emotion-detection",
        "num_labels": config.NUM_LABELS,
        "train_size": config.TRAIN_SIZE,
        "validation_size": config.VALIDATION_SIZE,
        "test_size": config.TEST_SIZE,
        "max_seq_length": config.MAX_SEQ_LENGTH,
        "learning_rate": config.LEARNING_RATE,
        "epochs": config.NUM_TRAIN_EPOCHS,
        "batch_size": config.PER_DEVICE_TRAIN_BATCH_SIZE,
        "weight_decay": config.WEIGHT_DECAY,
        "warmup_ratio": config.WARMUP_RATIO,
    }
    if extra_config:
        run_config.update(extra_config)

    run = wandb.init(
        project=config.WANDB_PROJECT,
        name=run_name or config.WANDB_RUN_NAME,
        config=run_config,
        reinit=True,
    )
    logger.info("W&B run initialised: %s", run.url)
    return run


def finish_wandb():
    """Safely finish the active W&B run."""
    if wandb.run is not None:
        wandb.finish()
