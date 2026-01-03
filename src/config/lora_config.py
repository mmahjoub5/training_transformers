from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class LoraConfigSpec:
    """Configuration for LoRA adapter setup."""
    rank: int = 16
    lora_alpha: int = 12
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = None

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "LoraConfigSpec":
        lora_cfg = cfg.get("lora", {})
        return cls(
            rank=lora_cfg.get("rank", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 12),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
            target_modules=lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
