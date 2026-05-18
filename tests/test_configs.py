"""Tests for YAML config loading and schema validation (issue #17)."""

import textwrap

import pytest

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
