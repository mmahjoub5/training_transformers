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
