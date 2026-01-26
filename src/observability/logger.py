"""Observability logger utilities."""

from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str = "modular-rag", log_level: Optional[str] = None) -> logging.Logger:
    """Get a configured logger.

    Args:
        name: Logger name.
        log_level: Optional log level string (e.g., "INFO").

    Returns:
        Configured logger instance.
    """

    if log_level:
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = logging.INFO

    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return logging.getLogger(name)