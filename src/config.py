"""Central configuration for the MLOps NER pipeline — Group 38."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
EVAL_RESULTS_DIR = ARTIFACTS_DIR / "eval_results"
MODEL_CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

ID2LABEL_PATH = ARTIFACTS_DIR / "id2label.json"
PROCESSED_DATA_PATH = ARTIFACTS_DIR / "processed_ner_data.pkl"
CLASSIFICATION_REPORT_JSON = EVAL_RESULTS_DIR / "classification_report.json"
CLASSIFICATION_REPORT_TXT = EVAL_RESULTS_DIR / "classification_report.txt"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "yongsun-yoon/open-ner-english"
TRAIN_SIZE = int(os.getenv("TRAIN_SIZE", "2000"))
VALIDATION_SIZE = int(os.getenv("VALIDATION_SIZE", "400"))

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("HF_MODEL_NAME", "dslim/bert-base-NER")
HF_REPO_ID = os.getenv("HF_REPO_ID", "mlops-group38-ner")

# ---------------------------------------------------------------------------
# Label policy
# ---------------------------------------------------------------------------
RARE_ENTITY_POLICY = os.getenv("RARE_ENTITY_POLICY", "O")  # map rare tags -> O
RARE_ENTITY_THRESHOLD = int(os.getenv("RARE_ENTITY_THRESHOLD", "50"))
MAX_ENTITY_TYPES = int(os.getenv("MAX_ENTITY_TYPES", "500"))

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "3e-5"))
NUM_TRAIN_EPOCHS = int(os.getenv("NUM_TRAIN_EPOCHS", "3"))
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "16"))
PER_DEVICE_EVAL_BATCH_SIZE = int(os.getenv("PER_DEVICE_EVAL_BATCH_SIZE", "16"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.1"))
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "50"))
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "f1"

# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "ner-mlops-group38-project")
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", "run-v1")

# ---------------------------------------------------------------------------
# Secrets (injected via environment — never hardcoded)
# ---------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
