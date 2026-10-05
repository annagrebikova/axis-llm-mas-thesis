"""High-level case-neutral AXIS explanation service."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from ..artifacts.models import ArtifactRef
from ..artifacts.registry import ArtifactRegistry
from ..core.case_config import load_axis_case_capabilities
from ..replay.checkpoints import CheckpointIndex
from ..replay.executor import LangGraphReplayExecutor, WorkflowRunResult
from .explainer import AxisExplanationController, ExplanationResult
from .explained_case import ExplainedCaseResolver
from .comparison import NoOpReplayCache
from .interrogation_mode import (
    DEFAULT_AXIS_INTERROGATION_MODE,
    normalize_axis_interrogation_mode,
)
from .models import ExplainedCase


class AxisExplanationService:
    """Public coordinator: factual run → ExplainedCase → AxisExplanationController.

    Does not implement interrogation logic; the controller remains authoritative.
    """

    def __init__(
        self,
        *,
        case_definition: Any,
        factual_run_result: WorkflowRunResult,
        replay_executor: LangGraphReplayExecutor,
        checkpoint_index: Optional[CheckpointIndex] = None,
        llm_generate: Optional[Callable[[str], str]] = None,
        intervention_registry: Optional[ArtifactRegistry] = None,
        intervention_alternatives: Optional[Mapping[str, ArtifactRef]] = None,
        max_rounds: int = 5,
        max_attempts: Optional[int] = None,
        strict_factual_branch: bool = True,
        axis_interrogation_mode: str = DEFAULT_AXIS_INTERROGATION_MODE,
        diagnose_hybrid_add: bool = False,
        lecturebank_id: str = "",
    ) -> None:
        if factual_run_result is None:
            raise ValueError("factual_run_result is required")
        if strict_factual_branch and factual_run_result.branch_id != "factual":
            raise ValueError(
                f"AxisExplanationService requires branch_id='factual'; "
                f"got {factual_run_result.branch_id!r}"
            )
        idx = checkpoint_index
        if idx is None:
            idx = factual_run_result.checkpoint_index
        if idx is None:
            try:
                idx = replay_executor.get_checkpoint_index(factual_run_result.run_id)
            except Exception as exc:
                raise ValueError(
                    "checkpoint_index is required when factual_run_result "
                    "has no frozen checkpoint index"
                ) from exc
        thread_id = getattr(idx, "thread_id", None)
        if thread_id is not None and thread_id != factual_run_result.run_id:
            raise ValueError(
                f"checkpoint_index.thread_id {thread_id!r} != "
                f"factual_run_result.run_id {factual_run_result.run_id!r}"
            )
        # Replay executor must already know this factual run (same checkpoint tree).
        try:
            replay_idx = replay_executor.get_checkpoint_index(factual_run_result.run_id)
        except KeyError as exc:
            raise ValueError(
                f"replay_executor has no factual checkpoint index for "
                f"run_id={factual_run_result.run_id!r}"
            ) from exc
        if replay_idx.thread_id != factual_run_result.run_id:
            raise ValueError("replay executor checkpoint index run identity mismatch")

        caps = load_axis_case_capabilities(case_definition, strict=True)
        resolver = caps.explained_case_resolver
        if resolver is None:
            raise ValueError("case capabilities missing explained_case_resolver")

        registry = factual_run_result.artifact_registry
        if registry is None:
            raise ValueError("factual_run_result has no artifact_registry")

        self.case_definition = case_definition
        self.factual_run_result = factual_run_result
        self.replay_executor = replay_executor
        self.checkpoint_index = idx
        self.capabilities = caps
        self._resolver: ExplainedCaseResolver = resolver
        self.llm_generate = llm_generate
        self.intervention_registry = intervention_registry
        self.intervention_alternatives = dict(intervention_alternatives or {})
        self.max_rounds = int(max_rounds)
        self.max_attempts = int(
            max_attempts if max_attempts is not None else max(10, self.max_rounds * 2)
        )
        self.axis_interrogation_mode = normalize_axis_interrogation_mode(
            axis_interrogation_mode
        )
        self.diagnose_hybrid_add = bool(diagnose_hybrid_add)
        self.lecturebank_id = str(lecturebank_id or "")
        self._registry = registry
        # One NOOP cache per service lifetime (shared across explain() calls /
        # explained MCQs for the same factual run + replay executor).
        self.noop_cache = NoOpReplayCache()

    def list_explainable_artifacts(self) -> Tuple[ArtifactRef, ...]:
        refs = self._resolver.list_explainable_artifacts(
            factual_run_result=self.factual_run_result
        )
        return tuple(refs)

    def resolve_case(self, artifact_ref: ArtifactRef) -> ExplainedCase:
        return self._resolver.resolve(
            factual_run_result=self.factual_run_result,
            artifact_ref=artifact_ref,
        )

    def explain(
        self,
        *,
        artifact_ref: ArtifactRef,
        question: str,
        scripted_queries: Optional[Sequence[str]] = None,
    ) -> ExplanationResult:
        explained = self.resolve_case(artifact_ref)
        controller = AxisExplanationController(
            explained_case=explained,
            registry=self._registry,
            occurrences=self.factual_run_result.occurrences,
            provenance_relations=self.factual_run_result.provenance_relations,
            checkpoint_index=self.checkpoint_index,
            replay_executor=self.replay_executor,
            case_definition=self.case_definition,
            factual_run_result=self.factual_run_result,
            llm_generate=self.llm_generate,
            intervention_registry=self.intervention_registry,
            intervention_alternatives=self.intervention_alternatives,
            max_rounds=self.max_rounds,
            max_attempts=self.max_attempts,
            noop_cache=self.noop_cache,
            axis_interrogation_mode=self.axis_interrogation_mode,
            diagnose_hybrid_add=self.diagnose_hybrid_add,
            lecturebank_id=getattr(self, "lecturebank_id", "") or "",
        )
        return controller.run(question, scripted_queries=scripted_queries)
