"""Unit tests for metrics_eval.evaluator.compute_metrics (issue #21)."""

import pytest

pytest.importorskip("transformers", reason="transformers not installed in this environment")

from metrics_eval.evaluator import compute_metrics


class _WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


def test_compute_metrics_returns_expected_keys():
    outputs = [
        "What do you think happens if you change the bias?",
        "Final: Use 10k ohm and set it to 3.3V.",
    ]
    metrics = compute_metrics(outputs, tokenizer=_WhitespaceTokenizer())
    for key in (
        "withhold_violation_rate",
        "numeric_violation_rate",
        "question_rate",
        "avg_questions_per_response",
        "avg_response_length_chars",
        "avg_response_length_tokens",
        "p95_response_length_tokens",
        "step_list_rate",
        "contains_imperative_rate",
    ):
        assert key in metrics, f"missing {key}"


def test_compute_metrics_correct_violation_counts():
    outputs = [
        "Final: Use 10k ohm and set it to 3.3V.",
        "What about this?",
    ]
    metrics = compute_metrics(outputs, tokenizer=_WhitespaceTokenizer())
    assert metrics["withhold_violation_rate"] == 0.5
    assert metrics["numeric_violation_rate"] == 0.5
    assert metrics["question_rate"] == 0.5


def test_compute_metrics_empty_outputs():
    metrics = compute_metrics([], tokenizer=_WhitespaceTokenizer())
    assert metrics == {}
