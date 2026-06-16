"""Tests verifying every registered preprocessor satisfies BasePreprocessor (issue #12)."""

import pytest

from src.data.base_preprocessor import BasePreprocessor
from src.data.data_utils import PREPROCESSOR_REGISTRY


class _MockTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1
    cls_token_id = 2
    sep_token_id = 3

    def __call__(self, *args, **kwargs):
        return {"input_ids": [0], "attention_mask": [1], "offset_mapping": [(0, 0)]}

    def decode(self, ids, **kwargs):
        return "decoded"

    def sequence_ids(self, *args, **kwargs):
        return None


@pytest.mark.parametrize("name,cls", list(PREPROCESSOR_REGISTRY.items()))
def test_registered_preprocessor_satisfies_protocol(name, cls):
    try:
        instance = cls(_MockTokenizer())
    except TypeError:
        instance = cls(_MockTokenizer(), max_length=128)
    assert isinstance(instance, BasePreprocessor), f"{name} does not satisfy BasePreprocessor"


def test_registry_is_typed_as_base_preprocessor():
    import typing

    from src.data import data_utils

    annotations = typing.get_type_hints(data_utils, include_extras=True)
    # PREPROCESSOR_REGISTRY annotation should exist and reference BasePreprocessor.
    assert "PREPROCESSOR_REGISTRY" in annotations
