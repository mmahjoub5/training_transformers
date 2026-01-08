from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class LoggingConfig:
    """Configuration for logging and checkpointing."""
    logging_steps: int = 100
    save_steps: int = 100
    report_to: List[str] | str | None = None
    logging_dir: str = "./runs"
    save_strategy: str = "epoch"
    level: str = "info"
    save_total_limit: int | None = None

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "LoggingConfig":
        logging_cfg = cfg.get("logging", {})
        return cls(
            logging_steps=logging_cfg.get("logging_steps", 100),
            save_steps=logging_cfg.get("save_steps", 100),
            report_to=logging_cfg.get("report_to", []),
            logging_dir=logging_cfg.get("logging_dir", "./runs"),
            save_strategy=logging_cfg.get("save_strategy", "epoch"),
            level=logging_cfg.get("level", "info"),
            save_total_limit=logging_cfg.get("save_total_limit", None),
        )
