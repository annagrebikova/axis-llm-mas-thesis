"""AXIS interrogation-mode constants and validation.

Constrained (default) enumerates exact executable queries for the LLM to copy.
Hybrid exposes a controlled vocabulary and lets the LLM compose one query,
which is then parsed, resolved, vocabulary-checked, and executed on the same
replay engine.

New interrogation is counterfactual-only: REMOVE / ADD / WHATIF / DONE.
WHAT remains parseable for historical experiment JSON, but is not advertised
or executed in new AXIS runs. Factual information comes from the execution
trace and provenance model.
"""

from __future__ import annotations

from typing import Any, FrozenSet

AXIS_INTERROGATION_MODE_CONSTRAINED = "constrained"
AXIS_INTERROGATION_MODE_HYBRID = "hybrid"
DEFAULT_AXIS_INTERROGATION_MODE = AXIS_INTERROGATION_MODE_CONSTRAINED
AXIS_INTERROGATION_MODES: FrozenSet[str] = frozenset(
    {
        AXIS_INTERROGATION_MODE_CONSTRAINED,
        AXIS_INTERROGATION_MODE_HYBRID,
    }
)
DEFAULT_HYBRID_MAX_ALTERNATIVES_PER_TARGET = 8

# Scientific interrogation policy. Distinct from hybrid vs constrained composition.
AXIS_INTERROGATION_POLICY_COUNTERFACTUAL_ONLY = "counterfactual_only"
AXIS_INTERROGATION_POLICY_WHAT_ENABLED = "what_enabled"
DEFAULT_AXIS_INTERROGATION_POLICY = AXIS_INTERROGATION_POLICY_COUNTERFACTUAL_ONLY
AXIS_INTERROGATION_POLICIES: FrozenSet[str] = frozenset(
    {
        AXIS_INTERROGATION_POLICY_COUNTERFACTUAL_ONLY,
        AXIS_INTERROGATION_POLICY_WHAT_ENABLED,
    }
)
WHAT_DISABLED_IN_INTERROGATION = (
    "WHAT is not in the advertised interrogation vocabulary. "
    "Factual workflow information is supplied by the execution trace and "
    "provenance model. Compose REMOVE, ADD, WHATIF, or DONE."
)


def normalize_axis_interrogation_mode(value: Any = None) -> str:
    """Return a canonical mode. ``None`` / blank → constrained (default)."""
    if value is None:
        return DEFAULT_AXIS_INTERROGATION_MODE
    text = str(value).strip().lower()
    if not text:
        return DEFAULT_AXIS_INTERROGATION_MODE
    if text not in AXIS_INTERROGATION_MODES:
        allowed = ", ".join(sorted(AXIS_INTERROGATION_MODES))
        raise ValueError(
            f"Invalid axis_interrogation_mode {value!r}. "
            f"Expected one of: {allowed}."
        )
    return text


def is_hybrid_interrogation_mode(value: Any = None) -> bool:
    return normalize_axis_interrogation_mode(value) == AXIS_INTERROGATION_MODE_HYBRID


def normalize_axis_interrogation_policy(value: Any = None) -> str:
    """Return a canonical interrogation policy.

    Blank / omitted → ``counterfactual_only`` (current generalized AXIS).
    Historical fingerprints without this key are not equivalent.
    """
    if value is None:
        return DEFAULT_AXIS_INTERROGATION_POLICY
    text = str(value).strip().lower()
    if not text:
        return DEFAULT_AXIS_INTERROGATION_POLICY
    if text not in AXIS_INTERROGATION_POLICIES:
        allowed = ", ".join(sorted(AXIS_INTERROGATION_POLICIES))
        raise ValueError(
            f"Invalid axis_interrogation_policy {value!r}. "
            f"Expected one of: {allowed}."
        )
    return text


def is_counterfactual_only_interrogation_policy(value: Any = None) -> bool:
    return (
        normalize_axis_interrogation_policy(value)
        == AXIS_INTERROGATION_POLICY_COUNTERFACTUAL_ONLY
    )
