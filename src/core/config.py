from typing import Any, Dict

import yaml

REQUIRED_TOP_LEVEL_KEYS = ("model", "data", "tokenizer", "training")


class ConfigValidationError(ValueError):
    """Raised when a loaded YAML config is missing required sections or fields."""


def validate_config(raw: Dict[str, Any]) -> None:
    """Validate that required top-level config sections are present.

    Args:
        raw: The dict produced by yaml.safe_load.

    Raises:
        ConfigValidationError: If any required section is missing.
    """
    if not isinstance(raw, dict):
        raise ConfigValidationError(
            f"Config root must be a mapping, got {type(raw).__name__}"
        )
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in raw:
            raise ConfigValidationError(f"Missing required config section: '{key}'")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    validate_config(raw)
    return raw
