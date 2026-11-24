"""Structured logging configuration for Retrieva AI."""

import logging
import sys


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure root and application loggers with structured formatting."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    )

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Suppress overly chatty third-party loggers if needed
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    logger = logging.getLogger("retrieva_ai")
    logger.setLevel(numeric_level)
    return logger


def get_logger(name: str = "retrieva_ai") -> logging.Logger:
    """Return a logger instance for the given component name."""
    if not name.startswith("retrieva_ai"):
        return logging.getLogger(f"retrieva_ai.{name}")
    return logging.getLogger(name)
