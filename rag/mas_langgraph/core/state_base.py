"""Abstract base for cooperative blackboard implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseCooperativeState(ABC):
    """Subclass for a new MAS domain; implement snapshot_counts() and domain storage."""

    def __init__(self) -> None:
        # Checkpointed counters so generate_id() is replay-stable from any fork.
        self._id_counters: Dict[str, int] = {}

    @abstractmethod
    def snapshot_counts(self) -> Dict[str, Any]:
        """Lightweight metrics for orchestration progress and logging."""

    def generate_id(self, prefix: str) -> str:
        """Deterministic ID from checkpointed counters (not random UUID)."""
        counters = getattr(self, "_id_counters", None)
        if not isinstance(counters, dict):
            counters = {}
            self._id_counters = counters
        n = int(counters.get(prefix, 0)) + 1
        counters[prefix] = n
        return f"{prefix}_{n:08d}"
