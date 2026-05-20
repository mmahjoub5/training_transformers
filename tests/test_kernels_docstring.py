"""Ensure src/kernels/__init__.py has a real docstring (issue #20)."""

import src.kernels as kernels


def test_kernels_module_has_docstring():
    assert kernels.__doc__ is not None
    assert len(kernels.__doc__.strip()) > 0
    assert "Placeholder" in kernels.__doc__ or "kernel" in kernels.__doc__.lower()
