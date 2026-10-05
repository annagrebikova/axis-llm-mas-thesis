"""Abstract base for orchestration condition evaluators."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict


class BaseNeedsEvaluator(ABC):
    """Subclass and implement condition methods; register via as_dict()."""

    def __init__(self, state: Any, coverage_plan: Any = None, **kwargs: Any) -> None:
        self.state = state
        self.coverage_plan = coverage_plan
        self._extra: Dict[str, Any] = kwargs

    @abstractmethod
    def as_dict(self) -> Dict[str, Callable[..., bool]]:
        """Map condition name (from orchestration_rules) -> callable(topic=...)."""

    def coverage_topic(self, topic: str) -> str:
        """Resolve input_template='coverage_topic' in orchestration rules."""
        return topic

    def register(self, conditions: Dict[str, Callable[..., bool]]) -> Dict[str, Callable[..., bool]]:
        """Merge subclass conditions with universal 'always'."""
        out = dict(conditions)
        out.setdefault("always", lambda **_: True)
        return out
