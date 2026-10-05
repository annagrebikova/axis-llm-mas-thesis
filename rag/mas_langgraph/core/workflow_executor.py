"""High-level workflow executor with run_id / checkpoint history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..artifacts.channel_state import ensure_artifact_channels
from ..core.case_config import CaseConfigurationError
from ..provenance.channel_state import ensure_provenance_channels
from ..replay.executor import LangGraphReplayExecutor, WorkflowRunResult
from .checkpointer import default_factual_checkpointer
from .runtime import WorkflowRuntime

# Re-export for callers that import WorkflowRunResult via the executor module.
__all__ = ["WorkflowExecutor", "WorkflowRunResult"]


@dataclass
class WorkflowExecutor:
    """Factual LangGraph execution with run_id, runtime context, and history."""

    runtime: Optional[WorkflowRuntime] = None
    checkpointer: Any = None
    compiled_graph: Any = None
    strict: bool = True
    legacy_mcq_fallback: bool = False
    _replay: Optional[LangGraphReplayExecutor] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.checkpointer is None:
            self.checkpointer = default_factual_checkpointer()
        if self.compiled_graph is None and self.runtime is not None:
            case = getattr(self.runtime, "case_definition", None)
            if case is not None and hasattr(case, "build_graph"):
                if self.strict:
                    self.compiled_graph = case.build_graph(
                        checkpointer=self.checkpointer
                    )
                else:
                    try:
                        self.compiled_graph = case.build_graph(
                            checkpointer=self.checkpointer
                        )
                    except Exception:
                        self.compiled_graph = None
        if self.compiled_graph is None and self.legacy_mcq_fallback:
            from .graph_builder import build_mcq_workflow_graph

            self.compiled_graph = build_mcq_workflow_graph(
                checkpointer=self.checkpointer
            )
        if self.compiled_graph is None:
            raise CaseConfigurationError(
                "WorkflowExecutor requires compiled_graph or "
                "runtime.case_definition.build_graph(); no implicit case graph is built"
            )
        self._replay = LangGraphReplayExecutor(
            compiled_graph=self.compiled_graph,
            checkpointer=self.checkpointer,
        )

    def run(
        self,
        initial_state: Dict[str, Any],
        runtime: Optional[WorkflowRuntime] = None,
        run_id: Optional[str] = None,
        *,
        resume_after_node: Optional[str] = None,
    ) -> WorkflowRunResult:
        """Invoke a factual run. ``thread_id`` equals ``run_id``.

        Runtime services are passed via LangGraph context — never injected into state.
        Optional ``resume_after_node`` continues from a seeded mid-graph checkpoint.
        """
        run_id = run_id or f"run_{uuid4().hex[:12]}"
        rt = runtime or self.runtime or WorkflowRuntime()
        state = ensure_provenance_channels(ensure_artifact_channels(dict(initial_state)))
        state["run_id"] = run_id
        state.setdefault("branch_id", "factual")
        state.setdefault("branch_type", "factual")
        state.setdefault("parent_branch_id", "")
        state.setdefault("origin_checkpoint_alias", "")
        state.setdefault("intervention_id", None)
        assert self._replay is not None
        case = getattr(rt, "case_definition", None)
        if case is not None and hasattr(case, "load_intervention_contracts"):
            if self.strict:
                self._replay.contracts = case.load_intervention_contracts()
            else:
                try:
                    self._replay.contracts = case.load_intervention_contracts()
                except Exception:
                    pass
        return self._replay.run_factual(
            state,
            run_id=run_id,
            runtime=rt,
            resume_after_node=resume_after_node,
        )

    def get_state_history(self, run_id: str) -> List[Any]:
        assert self._replay is not None
        return self._replay.get_state_history(run_id)

    def get_checkpoint_configurations(self, run_id: str) -> List[Dict[str, Any]]:
        assert self._replay is not None
        index = self._replay.get_checkpoint_index(run_id)
        return [
            {
                "alias": r.alias,
                "checkpoint_id": r.checkpoint_id,
                "thread_id": r.thread_id,
                "raw_config": r.raw_config,
            }
            for r in index.records
        ]

    def get_checkpoint_index(self, run_id: str):
        assert self._replay is not None
        return self._replay.get_checkpoint_index(run_id)

    def get_final_state(self, run_id: str) -> Dict[str, Any]:
        assert self._replay is not None
        return dict(self._replay.get_factual_result(run_id).final_state)

    def get_factual_result(self, run_id: str) -> WorkflowRunResult:
        assert self._replay is not None
        return self._replay.get_factual_result(run_id)

    @property
    def replay_executor(self) -> LangGraphReplayExecutor:
        assert self._replay is not None
        return self._replay
