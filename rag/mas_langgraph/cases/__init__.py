"""Case packages for the reusable MAS backbone."""

from __future__ import annotations

__all__ = [
    "MCQCaseDefinition",
    "McqOutcomeEvaluator",
]


def __getattr__(name: str):
    if name in {"MCQCaseDefinition", "McqOutcomeEvaluator"}:
        from .mcq import MCQCaseDefinition, McqOutcomeEvaluator

        mapping = {
            "MCQCaseDefinition": MCQCaseDefinition,
            "McqOutcomeEvaluator": McqOutcomeEvaluator,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
