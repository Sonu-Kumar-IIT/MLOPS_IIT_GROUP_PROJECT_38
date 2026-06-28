"""Shared helper utilities for the NER pipeline."""

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist; return the path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
