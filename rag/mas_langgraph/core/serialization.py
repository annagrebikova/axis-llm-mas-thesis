"""Serialization helpers for checkpoint-safe workflow state."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Set


_RUNTIME_KEYS = frozenset(
    {
        "agent_runner",
        "needs",
        "rag_agents",
        "orchestrator_config",
        "workflow_runtime",
    }
)


def strip_runtime_keys(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of state without runtime service handles."""
    return {k: v for k, v in state.items() if k not in _RUNTIME_KEYS}


def is_checkpoint_safe_value(value: Any) -> bool:
    """Heuristic: reject obvious non-serializable service objects."""
    if value is None:
        return True
    if isinstance(value, (str, int, float, bool, bytes)):
        return True
    if isinstance(value, (list, tuple, set)):
        return all(is_checkpoint_safe_value(v) for v in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, (str, int)) and is_checkpoint_safe_value(v)
            for k, v in value.items()
        )
    # dataclasses / simple objects with __dict__ of safe values — allow CooperativeState
    module = getattr(type(value), "__module__", "")
    name = type(value).__name__
    if name in {"CooperativeState", "OrchestratorConfig"} or module.startswith(
        "mas_langgraph.core"
    ):
        return True
    if name.endswith("AgentRunner") or name == "RAGAgents":
        return False
    if "NeedsEvaluator" in name:
        return False
    return True


def deep_snapshot(obj: Any) -> Any:
    """Deep-copy a value for genuine before/after snapshots."""
    try:
        return deepcopy(obj)
    except Exception:
        return obj
