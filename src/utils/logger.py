"""Logging utility module for Face Mask Detection System.

Configures formatted console and optional file logging with standard logging levels.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "face_mask_detection", log_file: Optional[Path] = None) -> logging.Logger:
    """Get or configure a logger with consistent formatting.

    Args:
        name: Name of the logger, typically __name__ or module name.
        log_file: Optional Path to write logs to disk.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console stream handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
