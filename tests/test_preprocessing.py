"""Tests for data preprocessors."""

from unittest.mock import MagicMock

from src.data.socratic_dialog_preprocess import SocraticPreprocessor


def _make_tokenizer():
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token = "<eos>"
    return tok


def _make_example(withhold=True, avoid_numeric=False):
    return {
        "turns": [
            {"role": "junior_engineer", "content": "How does gradient descent work?"},
            {"role": "senior_engineer", "content": "What do you think happens to loss as we adjust weights?"},
        ],
        "policy": {
            "withhold_final_answer": withhold,
            "avoid_numeric_rules_of_thumb": avoid_numeric,
        },
    }


class TestSocraticPreprocessor:
    def test_returns_messages_key(self):
        preprocessor = SocraticPreprocessor(_make_tokenizer())
        result = preprocessor(_make_example())
        assert "messages" in result
        assert isinstance(result["messages"], list)

    def test_message_roles(self):
        preprocessor = SocraticPreprocessor(_make_tokenizer())
        result = preprocessor(_make_example())
        roles = [m["role"] for m in result["messages"]]
        assert roles[0] == "system"
        assert "user" in roles
        assert "assistant" in roles

    def test_junior_maps_to_user_senior_maps_to_assistant(self):
        preprocessor = SocraticPreprocessor(_make_tokenizer())
        result = preprocessor(_make_example())
        non_system = [m for m in result["messages"] if m["role"] != "system"]
        assert non_system[0]["role"] == "user"
        assert non_system[0]["content"] == "How does gradient descent work?"
        assert non_system[1]["role"] == "assistant"

    def test_no_instance_state_mutation_between_calls(self):
        """Parallel calls must not bleed messages across examples."""
        preprocessor = SocraticPreprocessor(_make_tokenizer())
        ex1 = _make_example(withhold=True)
        ex2 = _make_example(withhold=False, avoid_numeric=True)

        result1 = preprocessor(ex1)
        result2 = preprocessor(ex2)

        # Each call should return independent message lists
        assert result1["messages"] is not result2["messages"]
        # result1 should not contain result2's messages
        assert len(result1["messages"]) == 3  # system + user + assistant
        assert len(result2["messages"]) == 3

    def test_policy_withhold_reflected_in_system_prompt(self):
        preprocessor = SocraticPreprocessor(_make_tokenizer())
        result = preprocessor(_make_example(withhold=True))
        system_content = result["messages"][0]["content"]
        assert "withhold the final answer" in system_content

    def test_policy_no_flags_gives_direct_answer_prompt(self):
        preprocessor = SocraticPreprocessor(_make_tokenizer())
        example = _make_example(withhold=False, avoid_numeric=False)
        result = preprocessor(example)
        system_content = result["messages"][0]["content"]
        assert "direct answers" in system_content

    def test_no_self_messages_attribute_after_call(self):
        """Ensure instance state is not modified by __call__."""
        preprocessor = SocraticPreprocessor(_make_tokenizer())
        assert not hasattr(preprocessor, "messages")
        preprocessor(_make_example())
        assert not hasattr(preprocessor, "messages")


"""Tests for data preprocessing classes."""

from src.data.eli_preprocess import ELI5Preprocessor_CLM


class FakeTokenizer:
    """Minimal tokenizer that returns deterministic token ids based on text length."""

    def __init__(self, tokens_per_call=10):
        self.tokens_per_call = tokens_per_call

    def __call__(self, text):
        n = max(1, len(text.split()))
        return {"input_ids": list(range(n))}


class TestELI5PreprocessorCLM:
    def test_short_sequence_returns_empty_lists_without_crash(self, caplog):
        tokenizer = FakeTokenizer()
        preprocessor = ELI5Preprocessor_CLM(tokenizer)
        preprocessor.block_size = 512

        example = {"title": "short title", "answers.text": ["short answer"]}
        with caplog.at_level("WARNING"):
            result = preprocessor(example)

        assert result == {"input_ids": [], "attention_mask": [], "labels": []}
        assert any("Skipping example" in rec.message for rec in caplog.records)

    def test_long_sequence_chunks_into_block_size(self):
        tokenizer = FakeTokenizer()
        preprocessor = ELI5Preprocessor_CLM(tokenizer)
        preprocessor.block_size = 4

        long_text = " ".join(["word"] * 20)
        example = {"title": long_text, "answers.text": [long_text]}
        result = preprocessor(example)

        assert len(result["input_ids"]) >= 1
        for chunk in result["input_ids"]:
            assert len(chunk) == preprocessor.block_size
        assert result["labels"] == result["input_ids"]
        assert len(result["attention_mask"]) == len(result["input_ids"])
