"""Logging utilities for the tactical analysis system."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project logger.

    Parameters
    ----------
    log_file:
        Optional path to a log file.
    level:
        Logging verbosity.
    """
    logger = logging.getLogger("ai_tactical_analyst")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
