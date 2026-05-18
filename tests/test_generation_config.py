"""Tests for the canonical GenerationConfig (issue #7)."""

from src.config.generation_config import GenerationConfig


def test_default_values():
    cfg = GenerationConfig()
    assert cfg.max_new_tokens == 200
    assert cfg.temperature == 0.2
    assert cfg.top_p == 1.0
    assert cfg.do_sample is None


def test_from_dict_with_generation_section():
    cfg = GenerationConfig.from_dict({"generation": {"max_new_tokens": 50, "temperature": 0.7, "top_p": 0.9, "do_sample": True}})
    assert cfg.max_new_tokens == 50
    assert cfg.temperature == 0.7
    assert cfg.top_p == 0.9
    assert cfg.do_sample is True


def test_from_dict_uses_defaults_when_missing():
    cfg = GenerationConfig.from_dict({})
    assert cfg.max_new_tokens == 200
    assert cfg.temperature == 0.2


def test_to_dict_omits_none_do_sample():
    cfg = GenerationConfig()
    out = cfg.to_dict()
    assert "do_sample" not in out
    assert out["max_new_tokens"] == 200


def test_to_dict_includes_do_sample_when_set():
    cfg = GenerationConfig(do_sample=False)
    out = cfg.to_dict()
    assert out["do_sample"] is False


def test_evaluator_reexports_same_class():
    from metrics_eval.evaluator import GenerationConfig as EvalGenCfg
    assert EvalGenCfg is GenerationConfig
