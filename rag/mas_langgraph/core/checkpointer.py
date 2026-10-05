"""Default LangGraph checkpointer for AXIS-ready factual execution."""

from __future__ import annotations

from typing import Any


def default_factual_checkpointer() -> Any:
    """Return a fresh in-memory checkpointer for factual workflow runs.

    Uses JSON-plus serde with pickle fallback so ``CooperativeState`` and other
    MCQ dataclass payloads can be persisted in checkpoints.
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return MemorySaver(serde=JsonPlusSerializer(pickle_fallback=True))
