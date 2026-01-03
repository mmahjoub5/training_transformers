from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TrainingConfig:
    """Configuration for training loop and optimization."""
    batch_size: int
    lr: float
    epochs: int
    output_dir: str
    precision: str = "fp32"
    eval_steps: int = 0
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    max_steps: int = -1
    weight_decay: float = 0.0
    num_workers: int = 4
    optim: str = "adamw_torch"
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.0

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "TrainingConfig":
        training_cfg = cfg.get("training", {})
        return cls(
            batch_size=training_cfg["batch_size"],
            lr=float(training_cfg["lr"]),
            epochs=training_cfg["epochs"],
            output_dir=training_cfg["output_dir"],
            precision=training_cfg.get("precision", "fp32"),
            eval_steps=training_cfg.get("eval_steps", 0),
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
            gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
            max_steps=training_cfg.get("max_steps", -1),
            weight_decay=training_cfg.get("weight_decay", 0.0),
            num_workers=training_cfg.get("num_workers", 4),
            optim=training_cfg.get("optim", "adamw_torch"),
            max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
            warmup_ratio=training_cfg.get("warmup_ratio", 0.0),
        )
