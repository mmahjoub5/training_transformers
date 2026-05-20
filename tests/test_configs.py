"""Tests for YAML config loading, schema validation, and config dataclass parsing."""

import textwrap

import pytest

from src.config.data_config import DataConfig
from src.config.deepspeed_config import DeepSpeedConfig
from src.config.profiling_config import ProfilingConfig
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


class TestDataConfig:
    def test_required_fields_parsed(self):
        cfg = DataConfig.from_dict({
            "data": {
                "name": "json",
                "preprocessor": "SocraticPreprocessor",
                "data_file": "data/custom_dataset/final_data.json",
            }
        })
        assert cfg.name == "json"
        assert cfg.preprocessor == "SocraticPreprocessor"
        assert cfg.data_file == "data/custom_dataset/final_data.json"

    def test_optional_fields_have_defaults(self):
        cfg = DataConfig.from_dict({
            "data": {"name": "squad", "preprocessor": "SQuADPreprocessor"}
        })
        assert cfg.config_name is None
        assert cfg.split is None
        assert cfg.train_split == "train"
        assert cfg.val_split == "validation"
        assert cfg.test_split is None
        assert cfg.max_train_samples is None
        assert cfg.max_val_samples is None
        assert cfg.max_test_samples is None
        assert cfg.shuffle_train is True
        assert cfg.seed == 42

    def test_full_config_round_trip(self):
        raw = {
            "data": {
                "name": "json",
                "preprocessor": "SocraticPreprocessor",
                "data_file": "data.json",
                "train_split": "train",
                "val_split": "validation",
                "test_split": None,
                "max_train_samples": 100,
                "max_val_samples": 20,
                "max_test_samples": None,
                "shuffle_train": False,
                "seed": 7,
            }
        }
        cfg = DataConfig.from_dict(raw)
        assert cfg.max_train_samples == 100
        assert cfg.shuffle_train is False
        assert cfg.seed == 7

    def test_omitting_optional_key_does_not_raise(self):
        """Any omitted optional key must use its default, not raise KeyError."""
        cfg = DataConfig.from_dict({"data": {"name": "json", "preprocessor": "SocraticPreprocessor"}})
        assert cfg.seed == 42
        assert cfg.shuffle_train is True


class TestProfilingConfig:
    def test_enabled_round_trip(self, sample_config_with_profiling):
        cfg = ProfilingConfig.from_dict(sample_config_with_profiling)
        assert cfg.enabled is True
        assert cfg.log_every_n_steps == 5
        assert cfg.trace_dir == "/tmp/traces"
        assert cfg.enable_torch_profiler is False

    def test_disabled_defaults(self, sample_config_minimal):
        cfg = ProfilingConfig.from_dict(sample_config_minimal)
        assert cfg.enabled is False
        assert cfg.log_every_n_steps == 10
        assert cfg.trace_dir is None

    def test_empty_dict_returns_defaults(self):
        cfg = ProfilingConfig.from_dict({})
        assert cfg.enabled is False
        assert cfg.log_every_n_steps == 10

    def test_to_dict(self, sample_config_with_profiling):
        cfg = ProfilingConfig.from_dict(sample_config_with_profiling)
        d = cfg.to_dict()
        assert d["enabled"] is True
        assert d["log_every_n_steps"] == 5


class TestDeepSpeedConfig:
    def test_enabled_with_zero_stage(self, sample_config_with_profiling):
        cfg = DeepSpeedConfig.from_dict(sample_config_with_profiling)
        assert cfg.enabled is True
        assert cfg.zero_stage == 2
        assert cfg.config_path is None

    def test_disabled_defaults(self, sample_config_minimal):
        cfg = DeepSpeedConfig.from_dict(sample_config_minimal)
        assert cfg.enabled is False
        assert cfg.zero_stage == 2

    def test_resolve_config_path_disabled(self):
        cfg = DeepSpeedConfig(enabled=False)
        assert cfg.resolve_config_path() is None

    def test_resolve_config_path_auto(self):
        cfg = DeepSpeedConfig(enabled=True, zero_stage=3)
        assert cfg.resolve_config_path() == "configs/deepspeed/ds_z3.json"

    def test_resolve_config_path_explicit(self):
        cfg = DeepSpeedConfig(enabled=True, config_path="/custom/ds.json")
        assert cfg.resolve_config_path() == "/custom/ds.json"

    def test_resolve_all_stages(self):
        for stage in (1, 2, 3):
            cfg = DeepSpeedConfig(enabled=True, zero_stage=stage)
            assert cfg.resolve_config_path() == f"configs/deepspeed/ds_z{stage}.json"
