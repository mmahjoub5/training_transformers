from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TrainingConfig:
    """Configuration for training loop and optimization."""
    batch_size: int
    lr: float
    epochs: int
    output_dir: str
    eval_strategy: str
    max_length: float
    lr_scheduler_type: str = "linear"
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
    load_best_model_at_end: bool = False
    early_stopping_patience: int | None = None
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "TrainingConfig":
        training_cfg = cfg.get("training", {})
        if training_cfg.get("eval_strategy", "epoch") is False:
            training_cfg["eval_strategy"] = "no"
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
            eval_strategy=training_cfg.get("eval_strategy", "epoch"),
            lr_scheduler_type=training_cfg.get("lr_scheduler_type", "linear"),
            max_length=training_cfg.get("max_length", 2048),
            load_best_model_at_end=training_cfg.get("load_best_model_at_end", False),
            early_stopping_patience=training_cfg.get("early_stopping_patience", None),
            metric_for_best_model=training_cfg.get("metric_for_best_model", "eval_loss"),
            greater_is_better=training_cfg.get("greater_is_better", False),
        )
