"""Tests that model_loader uses logger.debug instead of print banners (issue #8)."""

import inspect

from src.models import model_loader


def test_no_print_banner_calls_in_module_source():
    source = inspect.getsource(model_loader)
    assert "+++++++++++++++++++" not in source, "Debug ++++ banner should be removed"


def test_no_print_calls_in_module_source():
    source = inspect.getsource(model_loader)
    # Module should no longer use print(); logger.debug is the replacement.
    assert "print(" not in source


def test_logger_is_defined():
    assert hasattr(model_loader, "logger")
