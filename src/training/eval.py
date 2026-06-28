"""Evaluation metrics computation for token-classification (NER)."""

import json

import numpy as np
from sklearn.metrics import classification_report

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def compute_metrics_fn(id2label: dict):
    """Return a compute_metrics callable compatible with HuggingFace Trainer."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_labels, pred_labels = [], []
        for pred_seq, label_seq in zip(predictions, labels):
            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    continue
                true_labels.append(id2label.get(int(label_id), "O"))
                pred_labels.append(id2label.get(int(pred_id), "O"))

        report_dict = classification_report(
            true_labels, pred_labels,
            output_dict=True,
            zero_division=0,
        )
        return {
            "accuracy": report_dict.get("accuracy", 0.0),
            "f1": report_dict.get("weighted avg", {}).get("f1-score", 0.0),
            "precision": report_dict.get("weighted avg", {}).get("precision", 0.0),
            "recall": report_dict.get("weighted avg", {}).get("recall", 0.0),
            "macro_f1": report_dict.get("macro avg", {}).get("f1-score", 0.0),
            "macro_precision": report_dict.get("macro avg", {}).get("precision", 0.0),
            "macro_recall": report_dict.get("macro avg", {}).get("recall", 0.0),
        }

    return compute_metrics


def save_classification_report(trainer, val_dataset, id2label: dict):
    """Run a full evaluation pass and persist the classification report."""
    logger.info("Generating classification report …")
    predictions_output = trainer.predict(val_dataset)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids

    predictions = np.argmax(logits, axis=-1)
    true_labels, pred_labels = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            true_labels.append(id2label.get(int(label_id), "O"))
            pred_labels.append(id2label.get(int(pred_id), "O"))

    report_txt = classification_report(true_labels, pred_labels, zero_division=0)
    report_dict = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)

    with open(config.CLASSIFICATION_REPORT_TXT, "w") as f:
        f.write(report_txt)

    with open(config.CLASSIFICATION_REPORT_JSON, "w") as f:
        json.dump(report_dict, f, indent=2)

    logger.info("Classification report saved to %s", config.EVAL_RESULTS_DIR)
    return report_dict
