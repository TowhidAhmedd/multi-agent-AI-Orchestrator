"""
Centralized logging configuration.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", name: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        name: Logger name; defaults to root logger.

    Returns:
        Configured logger instance.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name or __name__)
    logger.setLevel(log_level)

    if not logger.handlers:
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger inheriting the root configuration.

    Args:
        name: Module or component name.

    Returns:
        Logger instance.
    """
    return setup_logging(name=name)
