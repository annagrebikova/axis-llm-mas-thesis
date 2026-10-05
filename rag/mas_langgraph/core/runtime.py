"""Workflow runtime services — never serialized into LangGraph checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class CaseDefinition(Protocol):
    """Bridge between generic infrastructure and a concrete LangGraph case.

    AXIS-specific adapters belong on ``AxisCaseDefinition``, not here.
    """

    case_id: str

    def build_graph(self, checkpointer: Any = None) -> Any:
        """Return the compiled LangGraph for this case."""
        ...

    def load_artifact_types(self) -> Any:
        ...

    def load_provenance_schema(self) -> Any:
        ...

    def load_outcome_evaluator(self) -> Any:
        ...

    def load_intervention_contracts(self) -> Any:
        ...

    def state_class(self) -> type:
        """Return the cooperative blackboard class for this case."""
        ...

    def project_state_for_artifact_resolution(
        self,
        state: Any,
        *,
        node_id: str = "",
    ) -> Any:
        """Optional: return resolver-filtered workflow state for a node."""
        return state


@runtime_checkable
class AxisCaseDefinition(CaseDefinition, Protocol):
    """Case that supports AXIS interrogation and grounded explanation."""

    def load_verbalizer(self) -> Any:
        ...

    def load_prompt_templates(self) -> Any:
        ...

    def load_provenance_slice_adapter(self) -> Any:
        ...

    def load_run_observation_adapter(self) -> Any:
        ...

    def load_provenance_instrumentation_adapter(self) -> Any:
        ...

    def load_explained_case_resolver(self) -> Any:
        ...

@dataclass(frozen=True)
class WorkflowRuntime:
    """Externally managed services for a workflow run.

    These objects must not be stored inside checkpointed LangGraph state.
    """

    case_definition: Any = None
    agent_runner: Any = None
    rag_agents: Any = None
    needs_evaluator: Any = None
    orchestrator_config: Any = None
    llm: Any = None
    cross_cove_verifier_llm: Any = None
    vector_store: Any = None
    rag_client: Any = None
    tool_clients: dict = field(default_factory=dict)
    artifact_registry: Any = None
    artifact_resolver: Any = None
    provenance_recorder: Any = None
    extra: dict = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            if val is not None:
                return val
        return self.extra.get(key, default)

    def prompt_dir(self) -> Optional[Path]:
        case = self.case_definition
        if case is None:
            return None
        if hasattr(case, "prompt_dir"):
            return case.prompt_dir()
        templates = case.load_prompt_templates()
        if templates:
            return None
        return None
