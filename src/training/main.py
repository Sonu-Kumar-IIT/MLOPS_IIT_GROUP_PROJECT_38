"""Orchestrates the full NER training pipeline.

Usage:
    python -m src.training.main
"""

from src import config
from src.training.data import load_and_prepare_data
from src.training.train import run_training
from src.training.eval import save_classification_report
from src.training.push_to_hub import push_model_to_hub
from src.wandb_utils import init_wandb, finish_wandb
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    logger.info("=== MLOps NER Pipeline — Group 38 ===")
    logger.info("Model       : %s", config.MODEL_NAME)
    logger.info("Dataset     : %s", config.DATASET_NAME)
    logger.info("Train size  : %d | Val size: %d", config.TRAIN_SIZE, config.VALIDATION_SIZE)
    logger.info("LR          : %s | Epochs: %d", config.LEARNING_RATE, config.NUM_TRAIN_EPOCHS)

    # 1. Initialise W&B (no-op if key not set)
    init_wandb()

    # 2. Load and prepare data
    train_ds, val_ds, label2id, id2label = load_and_prepare_data()

    # 3. Train
    trainer, model, tokenizer = run_training(train_ds, val_ds, label2id, id2label)

    # 4. Save detailed classification report
    save_classification_report(trainer, val_ds, id2label)

    # 5. Push to Hugging Face Hub
    push_model_to_hub(model, tokenizer)

    # 6. Finish W&B run
    finish_wandb()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
