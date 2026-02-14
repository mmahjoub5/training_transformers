from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DeepSpeedConfig:
    """Configuration for DeepSpeed integration."""

    enabled: bool = False
    config_path: Optional[str] = None
    zero_stage: int = 2

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "DeepSpeedConfig":
        ds_cfg = cfg.get("deepspeed", {})
        if not ds_cfg:
            return cls()
        return cls(
            enabled=ds_cfg.get("enabled", False),
            config_path=ds_cfg.get("config_path", None),
            zero_stage=ds_cfg.get("zero_stage", 2),
        )

    def resolve_config_path(self) -> Optional[str]:
        """Return the DeepSpeed JSON config path, or None when disabled."""
        if not self.enabled:
            return None
        if self.config_path:
            return self.config_path
        return f"configs/deepspeed/ds_z{self.zero_stage}.json"
