"""Tests that model_loader uses logger.debug instead of print banners (issue #8)."""

import pathlib

_SOURCE = pathlib.Path("src/models/model_loader.py").read_text()


def test_no_print_banner_calls_in_module_source():
    assert "+++++++++++++++++++" not in _SOURCE, "Debug ++++ banner should be removed"


def test_no_print_calls_in_module_source():
    assert "print(" not in _SOURCE


def test_logger_is_defined():
    assert "logger" in _SOURCE
