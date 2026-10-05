"""Node-local state binding — legacy services without shared mutable workflow state."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Optional


_MISSING = object()


def _is_bound_callable(attr: Any) -> bool:
    if not callable(attr):
        return False
    # Avoid wrapping class objects / constructors accidentally used as attrs.
    return not isinstance(attr, type)


@dataclass
class StateBoundRAGAgents:
    """Per-node RAG view; ``.state`` is the working ``CooperativeState`` only.

    Legacy ``RAGAgents`` methods close over ``self.state`` on the service
    instance. ``__getattr__`` would otherwise return those bound methods
    unchanged, so outline/LO writes landed on the original runtime blackboard
    while LangGraph cloned a different ``working_cs``.
    """

    service: Any
    state: Any

    def __getattr__(self, name: str) -> Any:
        service = object.__getattribute__(self, "service")
        attr = getattr(service, name)
        if not _is_bound_callable(attr):
            return attr
        bound_state = object.__getattribute__(self, "state")

        @wraps(attr)
        def _call(*args: Any, **kwargs: Any) -> Any:
            previous = getattr(service, "state", _MISSING)
            service.state = bound_state
            try:
                return attr(*args, **kwargs)
            finally:
                if previous is _MISSING:
                    try:
                        delattr(service, "state")
                    except Exception:
                        service.state = bound_state
                else:
                    service.state = previous

        return _call


def bind_rag_agents(service: Any, cooperative_state: Any) -> Optional[StateBoundRAGAgents]:
    """Bind legacy RAG helpers to a node-local cooperative state."""
    if service is None:
        return None
    return StateBoundRAGAgents(service=service, state=cooperative_state)
