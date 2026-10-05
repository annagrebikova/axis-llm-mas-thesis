"""Helpers for copy-safe state snapshots (Prompt 4 compatibility)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, MutableMapping

from .state_cloning import clone_cooperative_state

__all__ = [
    "capture_state_after",
    "capture_state_before",
    "clone_cooperative_state",
    "snapshots_share_refs",
]


def capture_state_before(state: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Capture a deep snapshot *before* domain logic mutates cooperative state."""
    cs = state.get("cooperative_state")
    if cs is not None and hasattr(cs, "snapshot_counts"):
        return deepcopy(cs.snapshot_counts())
    return deepcopy(dict(state))


def capture_state_after(
    state: MutableMapping[str, Any],
    update: MutableMapping[str, Any],
) -> Dict[str, Any]:
    """Capture snapshot after applying returned update conceptually.

    Prefer the cooperative_state referenced by the update when present.
    """
    cs = update.get("cooperative_state", state.get("cooperative_state"))
    if cs is not None and hasattr(cs, "snapshot_counts"):
        return deepcopy(cs.snapshot_counts())
    merged = {**dict(state), **dict(update)}
    return deepcopy({k: v for k, v in merged.items() if k != "cooperative_state"})


def snapshots_share_refs(before: Any, after: Any) -> bool:
    """Return True if before/after are the same object (undesired)."""
    return before is after
