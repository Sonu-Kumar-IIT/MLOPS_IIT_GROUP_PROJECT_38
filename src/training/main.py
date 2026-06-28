"""Orchestrates the full Emotion Detection training pipeline.

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
    logger.info("=== MLOps Emotion Detection Pipeline — Group 38 ===")
    logger.info("Dataset     : %s", config.DATASET_NAME)
    logger.info("Model       : %s", config.MODEL_NAME)
    logger.info("Train size  : %d | Val size: %d", config.TRAIN_SIZE, config.VALIDATION_SIZE)
    logger.info("LR          : %s | Epochs: %d | Batch: %d",
                config.LEARNING_RATE, config.NUM_TRAIN_EPOCHS, config.PER_DEVICE_TRAIN_BATCH_SIZE)

    # 1. Initialise W&B (no-op if WANDB_API_KEY not set)
    init_wandb()

    # 2. Load and prepare data
    train_ds, val_ds, test_ds, id2label, label2id = load_and_prepare_data()

    # 3. Train
    trainer, model, tokenizer = run_training(train_ds, val_ds, id2label, label2id)

    # 4. Evaluate on held-out test split and save reports + confusion matrix
    save_classification_report(trainer, test_ds, id2label)

    # 5. Push best model to Hugging Face Hub
    push_model_to_hub(model, tokenizer)

    # 6. Finish W&B run
    finish_wandb()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
