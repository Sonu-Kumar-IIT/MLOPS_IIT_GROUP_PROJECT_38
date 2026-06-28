"""Training loop — wraps HuggingFace Trainer for the NER pipeline."""

import os

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from src import config
from src.training.eval import compute_metrics_fn
from src.utils import get_logger

logger = get_logger(__name__)


def build_model(label2id: dict, id2label: dict):
    """Load the pretrained model with a fresh classification head."""
    hf_token = os.getenv("HF_TOKEN", config.HF_TOKEN) or None
    model = AutoModelForTokenClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(label2id),
        id2label={int(k): v for k, v in id2label.items()},
        label2id=label2id,
        ignore_mismatched_sizes=True,
        token=hf_token,
    )
    return model


def build_tokenizer():
    hf_token = os.getenv("HF_TOKEN", config.HF_TOKEN) or None
    return AutoTokenizer.from_pretrained(config.MODEL_NAME, token=hf_token)


def run_training(train_dataset, val_dataset, label2id: dict, id2label: dict):
    """Build model, Trainer, and run training.

    Returns (trainer, model, tokenizer).
    """
    tokenizer = build_tokenizer()
    model = build_model(label2id, id2label)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Determine whether W&B is available
    report_to = "wandb" if os.getenv("WANDB_API_KEY") else "none"

    training_args = TrainingArguments(
        output_dir=str(config.MODEL_CHECKPOINT_DIR),
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        eval_strategy=config.EVAL_STRATEGY,
        save_strategy=config.SAVE_STRATEGY,
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        logging_steps=config.LOGGING_STEPS,
        report_to=report_to,
        run_name=config.WANDB_RUN_NAME,
        fp16=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn(id2label),
    )

    logger.info("Starting training …")
    trainer.train()
    logger.info("Training complete.")

    return trainer, model, tokenizer
