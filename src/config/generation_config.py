from typing import Dict, Any
class GenerationConfig:
    """Configuration for text generation."""
    def __init__(
        self,
        max_new_tokens: int = 200,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dict for model.generate()."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }