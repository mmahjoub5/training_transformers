from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_name: str
    precision: str = "fp16"
    low_cpu_mem_usage: bool = True
    device_map: Optional[str] = None
    kind: str = "clm"
    attn_implementation: str = "sdpa"

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], *, precision: Optional[str] = None) -> "ModelConfig":
        model_cfg = cfg.get("model", {})
        return cls(
            model_name=model_cfg["name"],
            precision=precision or model_cfg.get("precision", "fp16"),
            low_cpu_mem_usage=model_cfg.get("low_cpu_mem_usage", True),
            device_map=model_cfg.get("device_map"),
            kind=model_cfg.get("kind", "clm"),
            attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        )
