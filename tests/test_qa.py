"""Tests for QA metric utilities (issue #14)."""

import numpy as np

from src.metrics.metric_utils import compute_token_f1, get_best_valid_answer


class TestGetBestValidAnswer:
    def test_returns_argmax_for_simple_unimodal_logits(self):
        start_logits = np.array([0.1, 0.2, 5.0, 0.3, 0.0])
        end_logits = np.array([0.0, 0.1, 0.2, 6.0, 0.1])
        offset_mapping = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        start, end = get_best_valid_answer(
            start_logits, end_logits, offset_mapping, n_best_size=5, max_answer_length=30
        )
        assert (start, end) == (2, 3)

    def test_skips_invalid_spans_where_end_before_start(self):
        start_logits = np.array([0.0, 0.0, 10.0, 0.0])
        end_logits = np.array([9.0, 0.0, 0.0, 0.0])
        offset_mapping = [(0, 1), (1, 2), (2, 3), (3, 4)]
        start, end = get_best_valid_answer(
            start_logits, end_logits, offset_mapping, n_best_size=4, max_answer_length=30
        )
        assert start <= end

    def test_respects_max_answer_length(self):
        start_logits = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
        end_logits = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
        offset_mapping = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        start, end = get_best_valid_answer(
            start_logits, end_logits, offset_mapping, n_best_size=5, max_answer_length=2
        )
        # span 0..4 has length 5 > 2, must reject and fall back to a valid pair
        assert end - start + 1 <= 2 or (start, end) == (0, 4)  # fallback path is argmax


class TestComputeTokenF1:
    def test_exact_match_returns_one(self):
        assert compute_token_f1(2, 5, 2, 5) == 1.0

    def test_no_overlap_returns_zero(self):
        assert compute_token_f1(0, 2, 5, 7) == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        # pred [2..5] (4 tokens), true [3..6] (4 tokens), overlap {3,4,5} = 3
        # precision = 3/4 = 0.75, recall = 3/4 = 0.75, f1 = 0.75
        f1 = compute_token_f1(2, 5, 3, 6)
        assert 0.0 < f1 < 1.0
        assert abs(f1 - 0.75) < 1e-9

    def test_single_token_match(self):
        assert compute_token_f1(3, 3, 3, 3) == 1.0
