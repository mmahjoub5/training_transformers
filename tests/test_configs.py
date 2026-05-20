"""Tests for YAML config loading, schema validation, and typed config dataclasses."""

import textwrap

import pytest

from src.config.tokenizer_config import TokenizerConfig
from src.config.training_config import TrainingConfig
from src.core.config import ConfigValidationError, load_config, validate_config


def _write_yaml(tmp_path, content):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


def test_validate_config_accepts_complete_config():
    raw = {
        "model": {"name": "x"},
        "data": {"name": "y"},
        "tokenizer": {"max_length": 128},
        "training": {"batch_size": 1, "lr": 1e-4, "epochs": 1, "output_dir": "out"},
    }
    validate_config(raw)  # no raise


def test_validate_config_raises_when_training_missing():
    raw = {"model": {}, "data": {}, "tokenizer": {}}
    with pytest.raises(ConfigValidationError, match="training"):
        validate_config(raw)


def test_validate_config_raises_when_model_missing():
    raw = {"data": {}, "tokenizer": {}, "training": {}}
    with pytest.raises(ConfigValidationError, match="model"):
        validate_config(raw)


def test_validate_config_raises_when_root_is_not_dict():
    with pytest.raises(ConfigValidationError, match="mapping"):
        validate_config([1, 2, 3])


def test_load_config_calls_validation(tmp_path):
    path = _write_yaml(tmp_path, """
    model:
      name: x
    data:
      name: y
    tokenizer:
      max_length: 128
    # training missing
    """)
    with pytest.raises(ConfigValidationError, match="training"):
        load_config(path)


def _minimal_training_cfg():
    return {
        "training": {
            "batch_size": 4,
            "lr": 1e-4,
            "epochs": 1,
            "output_dir": "/tmp/out",
            "eval_strategy": "epoch",
            "max_length": 2048,
        }
    }


def test_tokenizer_config_from_dict():
    cfg = TokenizerConfig.from_dict({"tokenizer": {"max_length": 4096}})
    assert cfg.max_length == 4096


def test_tokenizer_config_default_when_missing():
    cfg = TokenizerConfig.from_dict({})
    assert cfg.max_length == 2048


def test_training_config_seed_from_data():
    raw = _minimal_training_cfg()
    raw["data"] = {"seed": 123}
    cfg = TrainingConfig.from_dict(raw)
    assert cfg.seed == 123


def test_training_config_seed_from_training_section_overrides_data():
    raw = _minimal_training_cfg()
    raw["data"] = {"seed": 123}
    raw["training"]["seed"] = 7
    cfg = TrainingConfig.from_dict(raw)
    assert cfg.seed == 7


def test_training_config_seed_default():
    cfg = TrainingConfig.from_dict(_minimal_training_cfg())
    assert cfg.seed == 42


def test_build_training_args_signature_uses_tokenizer_config():
    import inspect

    from src.training.trainer_factory import build_training_args

    sig = inspect.signature(build_training_args)
    params = list(sig.parameters)
    assert "tokenizer_config" in params
    assert "config" not in params, "raw config dict should be removed"
