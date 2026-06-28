"""Evaluation metrics for Emotion Detection (sequence classification)."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

EMOTION_NAMES = list(config.EMOTION_LABELS.values())


def compute_metrics_fn(id2label: dict):
    """Return a compute_metrics callable compatible with HuggingFace Trainer."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        report = classification_report(
            labels, predictions,
            target_names=[id2label[i] for i in range(len(id2label))],
            output_dict=True,
            zero_division=0,
        )
        return {
            "accuracy": report.get("accuracy", 0.0),
            "f1": report.get("weighted avg", {}).get("f1-score", 0.0),
            "precision": report.get("weighted avg", {}).get("precision", 0.0),
            "recall": report.get("weighted avg", {}).get("recall", 0.0),
            "macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
            "macro_precision": report.get("macro avg", {}).get("precision", 0.0),
            "macro_recall": report.get("macro avg", {}).get("recall", 0.0),
        }

    return compute_metrics


def save_classification_report(trainer, test_dataset, id2label: dict):
    """Run a full evaluation pass on the test split and save reports."""
    logger.info("Generating classification report on test split …")
    output = trainer.predict(test_dataset)
    logits = output.predictions
    labels = output.label_ids

    predictions = np.argmax(logits, axis=-1)
    label_names = [id2label[i] for i in range(len(id2label))]

    report_txt = classification_report(labels, predictions, target_names=label_names, zero_division=0)
    report_dict = classification_report(labels, predictions, target_names=label_names,
                                        output_dict=True, zero_division=0)

    with open(config.CLASSIFICATION_REPORT_TXT, "w") as f:
        f.write(report_txt)
    with open(config.CLASSIFICATION_REPORT_JSON, "w") as f:
        json.dump(report_dict, f, indent=2)

    # Confusion matrix plot
    cm = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Emotion Detection")
    plt.tight_layout()
    fig.savefig(config.CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)

    logger.info("Reports saved to %s", config.EVAL_RESULTS_DIR)
    logger.info("\n%s", report_txt)
    return report_dict
