from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DataConfig:
    """Configuration for dataset loading and splitting."""
    name: str
    preprocessor: str
    data_file: Optional[str] = None
    config_name: Optional[str] = None
    split: Optional[str] = None
    train_split: str = "train"
    val_split: Optional[str] = "validation"
    test_split: Optional[str] = None
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    shuffle_train: bool = True
    seed: int = 42

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "DataConfig":
        data_cfg = cfg.get("data", {})
        return cls(
            name=data_cfg["name"],
            preprocessor=data_cfg["preprocessor"],
            data_file=data_cfg.get("data_file", None),
            config_name=data_cfg.get("config_name", None),
            split=data_cfg.get("split", None),
            train_split=data_cfg.get("train_split", "train"),
            val_split=data_cfg.get("val_split", "validation"),
            test_split=data_cfg.get("test_split", None),
            max_train_samples=data_cfg.get("max_train_samples", None),
            max_val_samples=data_cfg.get("max_val_samples", None),
            max_test_samples=data_cfg.get("max_test_samples", None),
            shuffle_train=data_cfg.get("shuffle_train", True),
            seed=data_cfg.get("seed", 42),
        )
