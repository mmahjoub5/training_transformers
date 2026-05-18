from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GenerationConfig:
    """Canonical generation configuration for evaluation and inference."""

    max_new_tokens: int = 200
    temperature: float = 0.2
    top_p: float = 1.0
    do_sample: Optional[bool] = None

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "GenerationConfig":
        """Build a GenerationConfig from a raw dict (e.g., YAML section)."""
        gen_cfg = cfg.get("generation", cfg) if isinstance(cfg, dict) else {}
        return cls(
            max_new_tokens=gen_cfg.get("max_new_tokens", 200),
            temperature=gen_cfg.get("temperature", 0.2),
            top_p=gen_cfg.get("top_p", 1.0),
            do_sample=gen_cfg.get("do_sample", None),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dict for model.generate()."""
        out = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.do_sample is not None:
            out["do_sample"] = self.do_sample
        return out
