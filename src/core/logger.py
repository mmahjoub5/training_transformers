# src/core/logger.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    run_dir: Path,
    *,
    name: str = "train",
    level: str = "info",
    log_file: str = "train.log",
    console: bool = True,
) -> logging.Logger:
    """
    Create a logger that writes to:
      - run_dir / log_file
      - stdout (optional)

    Args:
      run_dir: directory for this run (e.g., runs/2025-01-01_debug/)
      name: logger name
      level: "debug" | "info" | "warning" | "error"
      log_file: filename inside run_dir
      console: whether to also log to stdout

    Returns:
      configured logging.Logger
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    logger.propagate = False  # avoid duplicate logs if root logger exists

    # Clear old handlers (important for notebooks / repeated runs)
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(run_dir / log_file)
    file_handler.setLevel(level.upper())
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level.upper())
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
