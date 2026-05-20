"""Tests that metrics_eval can be imported from src.metrics_eval (issue #6)."""


def test_evaluator_importable_from_src():
    from src.metrics_eval.evaluator import EvalConfig, GenerationConfig, compute_metrics, evaluate
    assert callable(evaluate)
    assert callable(compute_metrics)
    assert EvalConfig is not None
    assert GenerationConfig is not None


def test_heuristics_importable_from_src():
    from src.metrics_eval.heuristics import is_numeric_violation, is_withhold_violation
    assert callable(is_withhold_violation)
    assert callable(is_numeric_violation)


def test_callback_importable_from_src():
    from src.metrics_eval.callback import MetricsEvalCallback
    assert MetricsEvalCallback is not None


def test_package_reexports():
    import src.metrics_eval as me
    assert hasattr(me, "evaluate")
    assert hasattr(me, "EvalConfig")
    assert hasattr(me, "is_withhold_violation")
