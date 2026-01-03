# src/data/dataset_loader.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from datasets import load_dataset, Dataset, DatasetDict



def load_dataset_generic(cfg: dict) -> DatasetDict:
    """
    Load a Hugging Face dataset in a task-agnostic way.

    Returns:
      DatasetDict with available splits:
        - "train"
        - "validation"
        - "test" (if requested)
    """
    if "data_file" in cfg.keys():
        ds: DatasetDict = load_dataset(
            cfg["name"],
            cfg["config_name"],
            data_files=cfg["data_file"]
        )
    else:
        ds: DatasetDict = load_dataset(
            cfg["name"],
            cfg["config_name"],
            split_name=cfg.get("split", None),
        )


    out: Dict[str, Dataset] = {}

    # ---- Train ----
    if cfg["train_split"]:
        train_ds = ds[cfg["train_split"]]
        if cfg["shuffle_train"]:
            train_ds = train_ds.shuffle(seed=cfg["seed"])
        if cfg["max_train_samples"] is not None:
            train_ds = train_ds.select(range(min(cfg["max_train_samples"], len(train_ds))))
        out["train"] = train_ds

    # ---- Validation ----
    if cfg["val_split"] and cfg["val_split"] in ds:
        val_ds = ds[cfg["val_split"]]
        if cfg["max_val_samples"] is not None:
            val_ds = val_ds.select(range(min(cfg["max_val_samples"], len(val_ds))))
        out["validation"] = val_ds

    # ---- Test ----
    if cfg["test_split"] and cfg["test_split"] in ds:
        test_ds = ds[cfg["test_split"]]
        if cfg["max_test_samples"] is not None:
            test_ds = test_ds.select(range(min(cfg["max_test_samples"], len(test_ds))))
        out["test"] = test_ds

    return DatasetDict(out)
