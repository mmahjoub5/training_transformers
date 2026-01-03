from typing import Optional

class ModelConfig:
    """Configuration for model loading."""
    def __init__(
        self,
        model_name: str,
        precision: str = "fp16",
        low_cpu_mem_usage: bool = True,
        device_map: Optional[str] = None,
        kind: str = "clm",
        attn_implementation: str = "sdpa"
    ):
        self.model_name = model_name
        self.precision = precision
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.device_map = device_map
        self.kind = kind
        self.attn_implementation = attn_implementation