"""Unit tests for src/metrics_eval/heuristics.py functions (issue #21)."""

from metrics_eval.heuristics import (
    avg_questions_per_response,
    contains_imperative,
    count_questions,
    count_step_lines,
    is_mostly_questions,
    is_numeric_violation,
    is_withhold_violation,
    question_rate,
)


class TestCountQuestions:
    def test_no_questions(self):
        assert count_questions("hello world") == 0

    def test_multiple_questions(self):
        assert count_questions("What? Why? How?") == 3

    def test_single_question(self):
        assert count_questions("What now?") == 1


class TestCountStepLines:
    def test_numbered_list(self):
        text = "1. Do X\n2. Do Y\n3. Do Z"
        assert count_step_lines(text) == 3

    def test_dash_list(self):
        assert count_step_lines("- a\n- b") == 2

    def test_no_list(self):
        assert count_step_lines("hello world") == 0


class TestIsWithholdViolation:
    def test_explicit_answer_phrase(self):
        assert is_withhold_violation("The answer is 42") is True

    def test_final_heading(self):
        assert is_withhold_violation("Final: Use this approach") is True

    def test_clean_question_passes(self):
        assert is_withhold_violation("What happens when bias changes?") is False


class TestIsNumericViolation:
    def test_numeric_with_unit(self):
        assert is_numeric_violation("Set it to 3.3V") is True

    def test_range_with_unit(self):
        assert is_numeric_violation("Use 10 to 20 ohm") is True

    def test_pure_text_no_violation(self):
        assert is_numeric_violation("What unit makes sense here?") is False


class TestQuestionRate:
    def test_all_questions(self):
        assert question_rate(["What?", "Why?"]) == 1.0

    def test_no_questions(self):
        assert question_rate(["hello", "world"]) == 0.0

    def test_empty_list(self):
        assert question_rate([]) == 0.0


class TestAvgQuestionsPerResponse:
    def test_multiple_questions_per_response(self):
        assert avg_questions_per_response(["What? Why?", "How?"]) == 1.5

    def test_no_responses(self):
        assert avg_questions_per_response([]) == 0.0


class TestIsMostlyQuestions:
    def test_only_questions(self):
        assert is_mostly_questions("What? Why?") is True

    def test_only_statements(self):
        assert is_mostly_questions("Hello. World.") is False


class TestContainsImperative:
    def test_imperative_at_start(self):
        assert contains_imperative("Do this thing") is True

    def test_no_imperative(self):
        assert contains_imperative("What do you think?") is False
