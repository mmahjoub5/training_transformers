"""Tests for MetricsEvalCallback initialization and lifecycle."""

from metrics_eval.callback import MetricsEvalCallback


def test_on_log_before_on_step_end_does_not_raise():
    callback = MetricsEvalCallback(eval_json="placeholder.json", eval_steps=10)
    # Calling on_log before any on_step_end fires must not raise AttributeError.
    logs = {}
    control = callback.on_log(args=None, state=None, control=object(), logs=logs)
    assert control is not None
    assert logs == {}


def test_pending_logs_initialized_to_none():
    callback = MetricsEvalCallback(eval_json="placeholder.json")
    assert callback._pending_logs is None


def test_on_log_merges_pending_logs_when_set():
    callback = MetricsEvalCallback(eval_json="placeholder.json")
    callback._pending_logs = {"custom/foo": 0.5}
    logs = {"loss": 1.0}
    callback.on_log(args=None, state=None, control=object(), logs=logs)
    assert logs == {"loss": 1.0, "custom/foo": 0.5}
    assert callback._pending_logs is None
