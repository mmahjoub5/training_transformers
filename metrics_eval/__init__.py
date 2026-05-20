from metrics_eval.heuristics import is_numeric_violation, is_withhold_violation

_EVALUATOR_EXPORTS = frozenset({"EvalConfig", "GenerationConfig", "evaluate", "run_eval"})


def __getattr__(name: str):
    if name in _EVALUATOR_EXPORTS:
        from metrics_eval import evaluator as _ev

        symbols = {n: getattr(_ev, n) for n in _EVALUATOR_EXPORTS}
        globals().update(symbols)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "is_numeric_violation",
    "is_withhold_violation",
]
