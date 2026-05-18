"""Test that ELI5Preprocessor_QA no longer emits debug print() (issue #15)."""

import io
from contextlib import redirect_stdout

from src.data.eli_preprocess import ELI5Preprocessor_QA


class _FakeTokenizer:
    pad_token = None
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(5))}

    def decode(self, ids, **kwargs):
        return "decoded"


def test_call_emits_no_stdout():
    preprocessor = ELI5Preprocessor_QA(_FakeTokenizer(), max_length=20)
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = preprocessor({"title": "q", "answers.text": ["a"]})
    assert buf.getvalue() == ""
    assert "input_ids" in result
    assert "labels" in result
