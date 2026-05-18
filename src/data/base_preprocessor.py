from typing import Protocol, runtime_checkable


@runtime_checkable
class BasePreprocessor(Protocol):
    """Structural type contract for preprocessors in PREPROCESSOR_REGISTRY.

    Any class that implements `__call__(self, example: dict) -> dict` satisfies
    this protocol via structural typing — no inheritance required.
    """

    def __call__(self, example: dict) -> dict: ...
