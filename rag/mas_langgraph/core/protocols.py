"""Protocols for pluggable core and orchestrator components.

Implement these in custom modules and register classes in input/profile.py.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class StateProtocol(Protocol):
    """Minimal shared blackboard interface required by the orchestrator."""

    def snapshot_counts(self) -> Dict[str, Any]: ...

    def generate_id(self, prefix: str) -> str: ...


@runtime_checkable
class NeedsEvaluatorProtocol(Protocol):
    """Condition evaluator for orchestration_rules.py condition names."""

    def as_dict(self) -> Dict[str, Callable[..., bool]]: ...

    def coverage_topic(self, topic: str) -> str: ...


@runtime_checkable
class WorkflowOrchestratorProtocol(Protocol):
    """Rule-based orchestration loop."""

    orchestration_log: List[Any]

    def get_next_orchestration_action(
        self,
        last_agent_id: Optional[str] = None,
        step: int = 0,
    ) -> Tuple[Optional[str], str, Optional[str]]: ...

    def run_full_workflow(self) -> Dict[str, Any]: ...
