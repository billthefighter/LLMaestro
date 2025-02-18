"""Centralized logging configuration for the project."""

import logging
import sys
from pathlib import Path
from typing import Optional


def configure_logging(
    level: int = logging.INFO, log_file: Optional[Path] = None, module_name: Optional[str] = None
) -> logging.Logger:
    """Configure logging with consistent formatting across the project.

    Args:
        level: The logging level to use
        log_file: Optional path to write logs to a file
        module_name: Optional module name to use for the logger

    Returns:
        Configured logger instance
    """
    # Create formatter
    formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Get logger for the module
    logger = logging.getLogger(module_name if module_name else __name__)
    logger.setLevel(level)

    return logger
