"""Central configuration for the MLOps Emotion Detection pipeline — Group 38."""

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
PROCESSED_DATA_PATH = ARTIFACTS_DIR / "processed_emotion_data.pkl"
CLASSIFICATION_REPORT_JSON = EVAL_RESULTS_DIR / "classification_report.json"
CLASSIFICATION_REPORT_TXT = EVAL_RESULTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = EVAL_RESULTS_DIR / "confusion_matrix.png"

# ---------------------------------------------------------------------------
# Dataset  — dair-ai/emotion
#   6 emotion classes: sadness(0), joy(1), love(2), anger(3), fear(4), surprise(5)
#   Full split sizes: train=16000, validation=2000, test=2000
# ---------------------------------------------------------------------------
DATASET_NAME = "dair-ai/emotion"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

TRAIN_SIZE = int(os.getenv("TRAIN_SIZE", "16000"))
VALIDATION_SIZE = int(os.getenv("VALIDATION_SIZE", "2000"))
TEST_SIZE = int(os.getenv("TEST_SIZE", "2000"))

# Fixed label mapping — will also be written to artifacts/id2label.json at runtime
EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}
NUM_LABELS = len(EMOTION_LABELS)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("HF_MODEL_NAME", "distilbert-base-uncased")
HF_REPO_ID = os.getenv("HF_REPO_ID", "Sonu-Kumar-IIT/mlops-group38-emotion")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "128"))

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
NUM_TRAIN_EPOCHS = int(os.getenv("NUM_TRAIN_EPOCHS", "4"))
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "32"))
PER_DEVICE_EVAL_BATCH_SIZE = int(os.getenv("PER_DEVICE_EVAL_BATCH_SIZE", "64"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.06"))
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "100"))
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "f1"

# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "emotion-mlops-group38-project")
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", "run-v1")

# ---------------------------------------------------------------------------
# Secrets (injected via environment — never hardcoded)
# ---------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
