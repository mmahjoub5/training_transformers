from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer behavior."""

    max_length: int = 2048

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "TokenizerConfig":
        tok_cfg = cfg.get("tokenizer", {})
        return cls(max_length=tok_cfg.get("max_length", 2048))
