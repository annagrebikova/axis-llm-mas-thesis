"""Workflow execution with LangGraph checkpointer and real fork-and-replay."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from uuid import uuid4

from ..artifacts.models import ArtifactRef
from ..artifacts.registry import ArtifactRegistry
from ..artifacts.resolver import ArtifactResolver
from .branch_artifacts import (
    SOURCE_INTERVENTION,
    SOURCE_ORIGIN,
    artifact_exists_at_checkpoint,
    envelope_type_at_checkpoint,
    rebase_artifact_channels_for_branch,
    resolve_intervention_source_envelope,
    validate_no_substitution_cycle,
)
from .branches import BranchMetadata, BranchStore
from .checkpoints import (
    BoundarySelector,
    CheckpointIndex,
    CheckpointRecord,
    extract_checkpoint_id,
    extract_thread_id,
)
from .events import InterventionEvent
from .interventions import ArtifactIntervention, InterventionContractRegistry
from .llm_determinism import (
    WorkflowLLMReplayLedger,
    WorkflowLLMReplayMissError,
    install_replay_aware_llm,
    workflow_llm_execution_context,
)

if TYPE_CHECKING:
    from ..core.runtime import WorkflowRuntime


def _branch_key(run_id: str, branch_id: str) -> Tuple[str, str]:
    return (run_id, branch_id)


def _replay_span(name: str):
    """No-op unless an AXIS replay profile scope is active."""
    try:
        from ..experiment.timing import timed_replay_span

        return timed_replay_span(name)
    except Exception:
        from contextlib import nullcontext

        return nullcontext()


def _normalize_workflow_branch_type(
    branch_type: Any,
    *,
    branch_id: str = "",
) -> str:
    """Map metadata/branch_id to factual|noop|counterfactual.

    Critical: never use substring ``\"factual\" in \"counterfactual\"``.
    """
    raw = str(branch_type or "").strip().lower()
    bid = str(branch_id or "").strip().lower()
    if raw in {"noop", "no_op", "control"} or bid.startswith("noop"):
        return "noop"
    if raw in {"counterfactual", "cf"} or bid.startswith("cf"):
        return "counterfactual"
    if raw == "factual" or bid == "factual" or bid.startswith("factual"):
        return "factual"
    if bid.startswith("noop"):
        return "noop"
    if bid.startswith("cf"):
        return "counterfactual"
    # Default for unknown fork ids: treat as counterfactual when not factual/noop.
    if raw:
        return "counterfactual"
    return "factual"


@dataclass
class WorkflowRunResult:
    """Result of a factual or counterfactual workflow invocation."""

    run_id: str
    branch_id: str
    final_state: Dict[str, Any]
    step_logs: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint_index: Optional[CheckpointIndex] = None
    error: Optional[str] = None
    raw_result: Any = None
    branch_metadata: Optional[BranchMetadata] = None
    branch_history: List[Any] = field(default_factory=list)
    origin_checkpoint: Optional[CheckpointRecord] = None
    llm_replay_diagnostics: Dict[str, Any] = field(default_factory=dict)
    # Target-bounded replay diagnostics (backward-compatible optional fields).
    replay_completion_mode: str = "graph_end"
    stopped_after_node: Optional[str] = None
    skipped_downstream_steps: Optional[int] = None
    target_observable: Optional[bool] = None
    executed_nodes: List[str] = field(default_factory=list)
    node_timings: List[Dict[str, Any]] = field(default_factory=list)
    node_profile_unattributed: Dict[str, Any] = field(default_factory=dict)
    replay_span_profile: Dict[str, Any] = field(default_factory=dict)

    @property
    def occurrences(self) -> List[Any]:
        """Prefer index-enriched occurrences for factual runs."""
        if (
            self.branch_id == "factual"
            and self.checkpoint_index is not None
            and self.checkpoint_index.resolved_occurrences
        ):
            return list(self.checkpoint_index.resolved_occurrences)
        return list(self.final_state.get("occurrence_records") or [])

    @property
    def resolved_occurrences(self) -> List[Any]:
        if self.checkpoint_index is None:
            return []
        return list(self.checkpoint_index.resolved_occurrences)

    @property
    def branch_occurrences(self) -> List[Any]:
        """Occurrences whose ``branch_id`` matches this result branch."""
        return [
            o
            for o in (self.final_state.get("occurrence_records") or [])
            if getattr(o, "branch_id", None) == self.branch_id
        ]

    @property
    def inherited_occurrences(self) -> List[Any]:
        return [
            o
            for o in (self.final_state.get("occurrence_records") or [])
            if getattr(o, "branch_id", None) != self.branch_id
        ]

    @property
    def branch_provenance_relations(self) -> List[Any]:
        bid = self.branch_id
        out = []
        for rel in self.final_state.get("run_provenance_relations") or []:
            cons = getattr(rel, "consumer_occurrence_id", "") or ""
            if f":{bid}:" in cons or cons.startswith(f"{self.run_id}:{bid}:"):
                out.append(rel)
        return out

    @property
    def provenance_relations(self) -> List[Any]:
        return list(self.final_state.get("run_provenance_relations") or [])

    @property
    def produced_by(self) -> List[Any]:
        return list(self.final_state.get("produced_by") or [])

    @property
    def artifact_derivations(self) -> List[Any]:
        return list(self.final_state.get("artifact_derivations") or [])

    @property
    def intervention_events(self) -> List[Any]:
        return list(self.final_state.get("intervention_events") or [])

    @property
    def artifact_registry(self) -> Optional[ArtifactRegistry]:
        """Return the artifact registry view from final state."""
        from ..artifacts.channel_state import load_artifact_channels

        try:
            registry, _resolver = load_artifact_channels(self.final_state)
            return registry
        except Exception:
            return None

    @property
    def provenance_violations(self) -> List[str]:
        return list(self.final_state.get("provenance_violations") or [])


class LangGraphReplayExecutor:
    """Runs factual workflows and forks real LangGraph NOOP / REMOVE replays."""

    def __init__(
        self,
        *,
        compiled_graph: Any = None,
        checkpointer: Any = None,
        contracts: Optional[InterventionContractRegistry] = None,
    ) -> None:
        self.compiled_graph = compiled_graph
        self.checkpointer = checkpointer
        self.contracts = contracts or InterventionContractRegistry()
        self.branches = BranchStore()
        self._factual_histories: Dict[str, List[Any]] = {}
        self._factual_results: Dict[str, Dict[str, Any]] = {}
        self._factual_indices: Dict[str, CheckpointIndex] = {}
        self._runtimes: Dict[str, Any] = {}
        self._branch_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._branch_histories: Dict[Tuple[str, str], List[Any]] = {}
        self._branch_fork_configs: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._resolvers: Dict[Tuple[str, str], ArtifactResolver] = {}
        # Run-scoped workflow LLM response ledgers (deterministic replay).
        self._llm_ledgers: Dict[str, WorkflowLLMReplayLedger] = {}
        # Compatibility aliases used by older unit tests
        self._histories = self._factual_histories
        self._final_states = self._factual_results
        self._indices = self._factual_indices

    def get_or_create_llm_ledger(self, run_id: str) -> WorkflowLLMReplayLedger:
        if run_id not in self._llm_ledgers:
            self._llm_ledgers[run_id] = WorkflowLLMReplayLedger(run_id=run_id)
        return self._llm_ledgers[run_id]

    def get_workflow_llm_diagnostics(self, run_id: str) -> Dict[str, Any]:
        ledger = self._llm_ledgers.get(run_id)
        if ledger is None:
            return {"workflow_llm_determinism_enabled": False}
        return ledger.diagnostics()

    def _ensure_workflow_llm_wrapped(self, runtime: Any) -> None:
        """Install replay-aware wrapper on runtime rag_agents if present."""
        if runtime is None:
            return
        rag = getattr(runtime, "rag_agents", None)
        if rag is None:
            rag = getattr(getattr(runtime, "agent_runner", None), "rag_agents", None)
        if rag is None or getattr(rag, "llm", None) is None:
            return
        try:
            install_replay_aware_llm(rag, temperature=0)
        except Exception:
            return
        # Keep agent_runner.llm in sync if it caches a direct reference.
        runner = getattr(runtime, "agent_runner", None)
        if runner is not None and hasattr(runner, "llm"):
            runner.llm = rag.llm
            if hasattr(runner, "base_llm"):
                runner.base_llm = getattr(rag, "base_llm", None)

    def _ensure_cross_cove_verifier_llm(self, runtime: Any) -> Any:
        """Install Cross-CoVe verifier once per runtime (never inside a node)."""
        if runtime is None:
            return runtime
        from ..domains.quiz.cross_cove import (
            build_cross_cove_verifier_llm,
            llm_model_identity,
            resolve_cross_cove_verifier_model,
            resolve_cross_cove_verifier_temperature,
        )
        from .llm_determinism import ReplayAwareWorkflowLLM

        existing = getattr(runtime, "cross_cove_verifier_llm", None)
        if existing is not None:
            if not isinstance(existing, ReplayAwareWorkflowLLM):
                wrapped = ReplayAwareWorkflowLLM(existing)
                runtime = replace(runtime, cross_cove_verifier_llm=wrapped)
                rag = getattr(runtime, "rag_agents", None)
                if rag is not None:
                    try:
                        rag.cross_cove_verifier_llm = wrapped
                    except Exception:
                        pass
            return runtime

        rag = getattr(runtime, "rag_agents", None)
        if rag is None:
            rag = getattr(getattr(runtime, "agent_runner", None), "rag_agents", None)
        generator_llm = getattr(rag, "llm", None) if rag is not None else None
        if generator_llm is None:
            generator_llm = getattr(runtime, "llm", None)
        generator_model = llm_model_identity(generator_llm)
        if not generator_model and rag is not None:
            generator_model = (
                getattr(rag, "llm_model", None)
                or getattr(rag, "model_name", None)
                or getattr(getattr(rag, "config", None), "llm_model", None)
            )
            generator_model = str(generator_model) if generator_model else None
        config = getattr(runtime, "orchestrator_config", None)
        verifier_model = resolve_cross_cove_verifier_model(
            config, generator_model=generator_model
        )
        verifier_temp = resolve_cross_cove_verifier_temperature(config)
        verifier_llm, _, _ = build_cross_cove_verifier_llm(
            verifier_model=verifier_model,
            generator_llm=generator_llm,
            generator_model=generator_model,
            temperature=verifier_temp,
        )
        if verifier_llm is None:
            return runtime
        if not isinstance(verifier_llm, ReplayAwareWorkflowLLM):
            verifier_llm = ReplayAwareWorkflowLLM(verifier_llm)
        runtime = replace(runtime, cross_cove_verifier_llm=verifier_llm)
        if rag is not None:
            try:
                rag.cross_cove_verifier_llm = verifier_llm
            except Exception:
                pass
        return runtime

    def _model_meta_for_ledger(self, runtime: Any) -> Tuple[str, Any]:
        rag = None
        if runtime is not None:
            rag = getattr(runtime, "rag_agents", None)
            if rag is None:
                rag = getattr(getattr(runtime, "agent_runner", None), "rag_agents", None)
        llm = getattr(rag, "llm", None) if rag is not None else None
        base = getattr(rag, "base_llm", None) if rag is not None else None
        model = (
            getattr(base, "model_name", None)
            or getattr(base, "model", None)
            or getattr(llm, "model_name", None)
            or getattr(llm, "model", None)
            or ""
        )
        temp = getattr(base, "temperature", None)
        if temp is None:
            temp = getattr(llm, "temperature", None)
        return str(model), temp

    def _bind_ledger_models(self, ledger: WorkflowLLMReplayLedger, runtime: Any) -> None:
        model_name, temperature = self._model_meta_for_ledger(runtime)
        ledger.model_name = model_name or ledger.model_name
        ledger.temperature = temperature if temperature is not None else ledger.temperature
        cc_llm = getattr(runtime, "cross_cove_verifier_llm", None) if runtime is not None else None
        if cc_llm is None:
            return
        try:
            from ..domains.quiz.cross_cove import llm_model_identity

            ledger.cross_cove_verifier_model = llm_model_identity(cc_llm) or (
                ledger.cross_cove_verifier_model
            )
        except Exception:
            return

    def run_factual(
        self,
        initial_state: Dict[str, Any],
        *,
        run_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        runtime: Optional["WorkflowRuntime"] = None,
        allow_synthetic_history: bool = False,
        resume_after_node: Optional[str] = None,
    ) -> WorkflowRunResult:
        """Invoke the compiled graph for a factual run.

        When ``resume_after_node`` is set, the initial state is written via
        ``update_state(..., as_node=...)`` and execution continues with
        ``invoke(None)``. This supports deterministic seeded fixtures without
        changing graph topology.
        """
        run_id = run_id or f"run_{uuid4().hex[:12]}"
        factual = BranchMetadata.factual(run_id)
        if self.branches.get_optional(run_id, "factual") is None:
            self.branches.register(factual)

        if runtime is not None:
            self._runtimes[run_id] = runtime
            case = getattr(runtime, "case_definition", None)
            if case is not None and hasattr(case, "load_intervention_contracts"):
                self.contracts = case.load_intervention_contracts()

        self._ensure_workflow_llm_wrapped(runtime)
        runtime = self._ensure_cross_cove_verifier_llm(runtime)
        if runtime is not None:
            self._runtimes[run_id] = runtime
        ledger = self.get_or_create_llm_ledger(run_id)
        self._bind_ledger_models(ledger, runtime)
        ledger.begin_branch("factual")

        invoke_config = {
            "configurable": {"thread_id": run_id},
            **(config or {}),
        }
        state = dict(initial_state)
        state.setdefault("run_id", run_id)
        state.setdefault("branch_id", "factual")
        state.setdefault("branch_type", "factual")
        state.setdefault("parent_branch_id", "")
        state.setdefault("origin_checkpoint_alias", "")
        state.setdefault("intervention_id", None)

        raw = None
        error = None
        if self.compiled_graph is not None:
            try:
                invoke_kwargs: Dict[str, Any] = {"config": invoke_config}
                if runtime is not None:
                    invoke_kwargs["context"] = runtime
                with workflow_llm_execution_context(
                    run_id=run_id,
                    branch_id="factual",
                    branch_type="factual",
                    ledger=ledger,
                ):
                    if resume_after_node:
                        self.compiled_graph.update_state(
                            invoke_config, state, as_node=resume_after_node
                        )
                        raw = self.compiled_graph.invoke(None, **invoke_kwargs)
                    else:
                        raw = self.compiled_graph.invoke(state, **invoke_kwargs)
                final = dict(raw) if isinstance(raw, dict) else {"result": raw}
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                final = dict(state)
        else:
            final = dict(state)

        factual_llm_diag = ledger.end_branch("factual")

        history: List[Any] = []
        if self.compiled_graph is not None and hasattr(
            self.compiled_graph, "get_state_history"
        ):
            try:
                history = list(self.compiled_graph.get_state_history(invoke_config))
                history = list(reversed(history))
            except Exception:  # noqa: BLE001
                history = []

        if not history:
            if allow_synthetic_history:
                history = [
                    _SyntheticSnapshot(
                        i, run_id, final, parent_index=i - 1 if i else None
                    )
                    for i in range(max(1, len(final.get("step_logs") or []) or 1))
                ]
            elif self.compiled_graph is not None:
                error = error or (
                    "No LangGraph checkpoint history available for factual run. "
                    "Ensure the graph was compiled with a checkpointer."
                )

        index = CheckpointIndex(thread_id=run_id)
        if history:
            index.build_from_history(
                history,
                allow_synthetic=allow_synthetic_history,
                source="langgraph" if self.compiled_graph is not None else "synthetic",
            )
            occs = list(final.get("occurrence_records") or [])
            if not occs and history:
                from .checkpoints import extract_occurrence_records

                occs = extract_occurrence_records(history[-1])
            if occs:
                index.enrich_from_occurrences(occs)

        self._factual_histories[run_id] = history
        self._factual_results[run_id] = deepcopy(final)
        self._factual_indices[run_id] = index
        self._resolvers[_branch_key(run_id, "factual")] = ArtifactResolver(
            registry=ArtifactRegistry(),
            branch_id="factual",
        )

        return WorkflowRunResult(
            run_id=run_id,
            branch_id="factual",
            final_state=final,
            step_logs=list(final.get("step_logs") or []),
            checkpoint_index=index,
            error=error,
            raw_result=raw,
            branch_metadata=factual,
            branch_history=list(history),
            llm_replay_diagnostics=dict(factual_llm_diag or {}),
        )

    def get_state_history(self, run_id: str) -> List[Any]:
        return list(self._factual_histories.get(run_id, []))

    def get_checkpoint_index(self, run_id: str) -> CheckpointIndex:
        if run_id not in self._factual_indices:
            raise KeyError(f"Unknown run_id: {run_id}")
        return self._factual_indices[run_id]

    def get_factual_result(self, run_id: str) -> WorkflowRunResult:
        """Read-only factual WorkflowRunResult (no rerun)."""
        if run_id not in self._factual_results:
            raise KeyError(f"No factual result for run_id: {run_id}")
        meta = self.branches.get(run_id, "factual")
        return WorkflowRunResult(
            run_id=run_id,
            branch_id="factual",
            final_state=deepcopy(self._factual_results[run_id]),
            step_logs=list(self._factual_histories.get(run_id, [])),
            checkpoint_index=self._factual_indices.get(run_id),
            branch_metadata=meta,
        )

    def load_checkpoint(
        self,
        run_id: str,
        selector: BoundarySelector,
    ) -> CheckpointRecord:
        return self.get_checkpoint_index(run_id).resolve(selector)

    def get_checkpoint_snapshot(
        self,
        run_id: str,
        selector: BoundarySelector,
    ) -> Any:
        record = self.load_checkpoint(run_id, selector)
        if self.compiled_graph is not None and hasattr(self.compiled_graph, "get_state"):
            if not record.raw_config:
                raise KeyError(f"No raw_config for checkpoint {record.alias}")
            snap = self.compiled_graph.get_state(record.raw_config)
            snap_id = extract_checkpoint_id(snap)
            if snap_id != record.checkpoint_id:
                raise ValueError(
                    f"Retrieved checkpoint_id {snap_id!r} != indexed "
                    f"{record.checkpoint_id!r} for {record.alias}"
                )
            tid = extract_thread_id(snap, run_id)
            if tid != run_id:
                raise ValueError(
                    f"Retrieved thread_id {tid!r} != run_id {run_id!r}"
                )
            return snap
        for snap in self._factual_histories.get(run_id, []):
            try:
                if extract_checkpoint_id(snap) == record.checkpoint_id:
                    return snap
            except ValueError:
                continue
        raise KeyError(f"Snapshot not found for {record.alias} / {record.checkpoint_id}")

    def _validate_intervention_checkpoint(
        self,
        run_id: str,
        checkpoint: CheckpointRecord,
    ) -> CheckpointRecord:
        index = self.get_checkpoint_index(run_id)
        if checkpoint.thread_id and checkpoint.thread_id != run_id:
            raise ValueError(
                f"Intervention checkpoint thread_id {checkpoint.thread_id!r} "
                f"!= run_id {run_id!r}"
            )
        try:
            indexed = index.get_alias(checkpoint.alias)
        except KeyError as exc:
            raise ValueError(
                f"Checkpoint alias {checkpoint.alias!r} not in factual index"
            ) from exc
        if indexed.checkpoint_id != checkpoint.checkpoint_id:
            raise ValueError(
                f"Checkpoint id mismatch for {checkpoint.alias}: "
                f"{checkpoint.checkpoint_id!r} vs indexed {indexed.checkpoint_id!r}"
            )
        return indexed

    def _require_runtime(self, run_id: str) -> Any:
        runtime = self._runtimes.get(run_id)
        if runtime is None:
            raise RuntimeError(
                f"No WorkflowRuntime stored for factual run {run_id!r}; "
                "pass runtime= to run_factual before replay"
            )
        return runtime

    def _build_fork_values(
        self,
        *,
        origin_values: MappingLike,
        meta: BranchMetadata,
        blocked: Optional[Set[ArtifactRef]] = None,
        additional_envelopes: Optional[List[Any]] = None,
        additional_source_meta: Optional[List[Dict[str, Any]]] = None,
        explicit_available_refs: Optional[Set[ArtifactRef]] = None,
        substitutions: Optional[Dict[ArtifactRef, ArtifactRef]] = None,
        intervention_events: Optional[List[InterventionEvent]] = None,
    ) -> Dict[str, Any]:
        with _replay_span("artifact_registration"):
            channels = rebase_artifact_channels_for_branch(
                origin_values,
                run_id=meta.run_id,
                new_branch_id=meta.branch_id,
                blocked=blocked,
                additional_envelopes=additional_envelopes,
                additional_source_meta=additional_source_meta,
                explicit_available_refs=explicit_available_refs,
                substitutions=substitutions,
            )
            values: Dict[str, Any] = {
                "run_id": meta.run_id,
                "branch_id": meta.branch_id,
                "branch_type": meta.branch_type,
                "parent_branch_id": meta.parent_branch_id,
                "origin_checkpoint_alias": meta.origin_checkpoint_alias,
                "intervention_id": meta.intervention_id,
                "provenance_node_counts": {},
                "provenance_execution_index": 0,
                "intervention_events": list(intervention_events or []),
                **channels,
            }
        # Keep inherited provenance from the origin checkpoint values
        with _replay_span("provenance_recording"):
            for key in (
                "occurrence_records",
                "produced_by",
                "run_provenance_relations",
                "provenance_violations",
                "provenance_unresolved_producers",
            ):
                if key in origin_values and key not in values:
                    values[key] = deepcopy(origin_values.get(key))
        return values

    def _prepare_intervention_fork(
        self,
        run_id: str,
        intervention: ArtifactIntervention,
        *,
        artifact_type: str = "",
        intervention_registry: Optional[ArtifactRegistry] = None,
    ) -> Tuple[BranchMetadata, Dict[str, Any], Any, CheckpointRecord]:
        """Validate intervention and build fork values (shared ADD/REPLACE/REMOVE)."""
        op = intervention.operation
        with _replay_span("intervention_prepare"):
            record = self._validate_intervention_checkpoint(run_id, intervention.checkpoint)
        with _replay_span("checkpoint_restore"):
            origin_snap = self.get_checkpoint_snapshot(
                run_id, BoundarySelector.by_alias(record.alias)
            )
            origin_values = dict(getattr(origin_snap, "values", None) or {})
        t_prepare = time.perf_counter()
        factual_final = self._factual_results.get(run_id) or {}

        blocked: Optional[Set[ArtifactRef]] = None
        additional_envelopes: List[Any] = []
        additional_meta: List[Dict[str, Any]] = []
        explicit_available: Set[ArtifactRef] = set()
        substitutions: Dict[ArtifactRef, ArtifactRef] = {}
        artifact_refs: Tuple[ArtifactRef, ...] = ()
        supplied_refs: List[ArtifactRef] = []
        source_kinds: List[str] = []

        if op == "REMOVE":
            assert isinstance(intervention.artifact, ArtifactRef)
            ref = intervention.artifact
            if not artifact_exists_at_checkpoint(origin_values, ref):
                raise ValueError(
                    f"Artifact {ref} does not exist at origin checkpoint {record.alias}"
                )
            derived_type = envelope_type_at_checkpoint(origin_values, ref) or artifact_type
            if not derived_type:
                raise ValueError(f"Cannot derive artifact type for {ref}")
            if artifact_type and artifact_type != derived_type:
                raise ValueError(
                    f"artifact_type={artifact_type!r} does not match envelope "
                    f"type {derived_type!r}"
                )
            self.contracts.validate(intervention, derived_type)
            blocked = {ref}
            artifact_refs = (ref,)

        elif op == "ADD":
            assert isinstance(intervention.artifact, ArtifactRef)
            ref = intervention.artifact
            source = resolve_intervention_source_envelope(
                ref=ref,
                origin_values=origin_values,
                factual_final_state=factual_final,
                intervention_registry=intervention_registry,
            )
            derived_type = str(source.envelope.artifact_type or artifact_type)
            if not derived_type:
                raise ValueError(f"Cannot derive artifact type for {ref}")
            if artifact_type and artifact_type != derived_type:
                raise ValueError(
                    f"artifact_type={artifact_type!r} does not match envelope "
                    f"type {derived_type!r}"
                )
            self.contracts.validate(intervention, derived_type)
            contract = self.contracts.require(derived_type)
            if (
                source.source_kind == SOURCE_INTERVENTION
                and not contract.allow_external_add
            ):
                raise ValueError(
                    f"External ADD not allowed for {derived_type} "
                    f"(allow_external_add=false)"
                )
            already_at_origin = artifact_exists_at_checkpoint(origin_values, ref)
            if not already_at_origin:
                additional_envelopes.append(source.envelope)
                additional_meta.append(
                    {
                        "intervention_source_kind": source.source_kind,
                        "source_instance_id": source.source_instance_id,
                        "source_branch_id": source.source_branch,
                        "intervention_id": intervention.intervention_id,
                        "supplied_by_intervention": True,
                    }
                )
                supplied_refs.append(ref)
                source_kinds.append(source.source_kind)
            else:
                source_kinds.append(SOURCE_ORIGIN)
            explicit_available.add(ref)
            artifact_refs = (ref,)

        elif op == "REPLACE":
            old_ref, new_ref = intervention.artifact  # type: ignore[misc]
            if old_ref == new_ref:
                raise ValueError(f"Self-replacement is not allowed: {old_ref}")
            if not artifact_exists_at_checkpoint(origin_values, old_ref):
                raise ValueError(
                    f"REPLACE old artifact {old_ref} does not exist at origin "
                    f"checkpoint {record.alias}"
                )
            old_type = envelope_type_at_checkpoint(origin_values, old_ref) or ""
            if not old_type:
                raise ValueError(f"Cannot derive type for old artifact {old_ref}")
            if artifact_type and artifact_type != old_type:
                raise ValueError(
                    f"artifact_type={artifact_type!r} does not match old envelope "
                    f"type {old_type!r}"
                )
            self.contracts.validate(intervention, old_type)
            contract = self.contracts.require(old_type)
            new_source = resolve_intervention_source_envelope(
                ref=new_ref,
                origin_values=origin_values,
                factual_final_state=factual_final,
                intervention_registry=intervention_registry,
            )
            new_type = str(new_source.envelope.artifact_type or "")
            if not new_type:
                raise ValueError(f"Cannot derive type for new artifact {new_ref}")
            if not contract.replacement_ok(new_type):
                raise ValueError(
                    f"Incompatible REPLACE: {old_type} → {new_type} not allowed"
                )

            trial_subs = dict(
                (origin_values.get("artifact_substitutions") or {})
            )
            trial_subs[old_ref] = new_ref
            validate_no_substitution_cycle(trial_subs)

            new_at_origin = artifact_exists_at_checkpoint(origin_values, new_ref)
            if not new_at_origin:
                additional_envelopes.append(new_source.envelope)
                additional_meta.append(
                    {
                        "intervention_source_kind": new_source.source_kind,
                        "source_instance_id": new_source.source_instance_id,
                        "source_branch_id": new_source.source_branch,
                        "intervention_id": intervention.intervention_id,
                        "supplied_by_intervention": True,
                    }
                )
                supplied_refs.append(new_ref)
                source_kinds.append(new_source.source_kind)
            substitutions[old_ref] = new_ref
            explicit_available.add(new_ref)
            artifact_refs = (old_ref, new_ref)

        else:
            raise ValueError(f"Unsupported intervention operation: {op}")

        parent = self.branches.get(run_id, "factual")
        meta = self.branches.fork(
            parent=parent,
            branch_type="counterfactual",
            origin_checkpoint_alias=record.alias,
            origin_checkpoint_id=record.checkpoint_id,
            intervention_id=intervention.intervention_id,
        )
        event = InterventionEvent(
            intervention_id=intervention.intervention_id,
            run_id=run_id,
            branch_id=meta.branch_id,
            operation=op,
            origin_checkpoint_alias=record.alias,
            origin_checkpoint_id=record.checkpoint_id,
            artifact_refs=artifact_refs,
            supplied_refs=tuple(supplied_refs),
            source_kinds=tuple(source_kinds),
            description=intervention.description or "",
        )
        try:
            from ..experiment.timing import add_replay_span

            add_replay_span("intervention_prepare", time.perf_counter() - t_prepare)
        except Exception:
            pass
        with _replay_span("intervention_application"):
            fork_values = self._build_fork_values(
                origin_values=origin_values,
                meta=meta,
                blocked=blocked,
                additional_envelopes=additional_envelopes or None,
                additional_source_meta=additional_meta or None,
                explicit_available_refs=explicit_available or None,
                substitutions=substitutions or None,
                intervention_events=[event],
            )
        return meta, fork_values, origin_snap, record

    def _fork_update_config(self, run_id: str, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """Attach WorkflowRuntime so conditional routers can resolve during update_state."""
        config = deepcopy(dict(raw_config or {}))
        configurable = dict(config.get("configurable") or {})
        workflow_rt = self._runtimes.get(run_id)
        if workflow_rt is not None:
            try:
                from langgraph.config import CONFIG_KEY_RUNTIME
                from langgraph.runtime import Runtime

                configurable[CONFIG_KEY_RUNTIME] = Runtime(context=workflow_rt)
            except Exception:
                pass
        config["configurable"] = configurable
        return config

    def _create_langgraph_fork(
        self,
        *,
        run_id: str,
        record: CheckpointRecord,
        origin_snap: Any,
        fork_values: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Any, str]:
        """Call update_state, validate ``next``, return (fork_config, fork_snap, fork_id)."""
        update_config = self._fork_update_config(run_id, record.raw_config)
        with _replay_span("fork_creation"):
            fork_config = self.compiled_graph.update_state(update_config, fork_values)
            fork_snap = self.compiled_graph.get_state(fork_config)
        origin_next = tuple(getattr(origin_snap, "next", None) or ())
        fork_next = tuple(getattr(fork_snap, "next", None) or ())
        if origin_next != fork_next:
            raise RuntimeError(
                f"Fork continuation next={fork_next!r} != origin next={origin_next!r}"
            )
        fork_id = extract_checkpoint_id(fork_snap)
        return fork_config, fork_snap, fork_id

    def fork_noop(
        self,
        run_id: str,
        selector: BoundarySelector,
        *,
        explained_case: Any = None,
        stop_when_observable: bool = False,
        stop_predicate: Optional[Callable[..., Any]] = None,
    ) -> WorkflowRunResult:
        """Real NOOP: fork from factual checkpoint and replay downstream."""
        if self.compiled_graph is None:
            return self._fixture_fork_noop(run_id, selector)

        with _replay_span("checkpoint_restore"):
            record = self.load_checkpoint(run_id, selector)
            origin_snap = self.get_checkpoint_snapshot(run_id, selector)
            origin_values = dict(getattr(origin_snap, "values", None) or {})
        parent = self.branches.get(run_id, "factual")
        meta = self.branches.fork(
            parent=parent,
            branch_type="noop",
            origin_checkpoint_alias=record.alias,
            origin_checkpoint_id=record.checkpoint_id,
        )
        with _replay_span("intervention_application"):
            fork_values = self._build_fork_values(
                origin_values=origin_values, meta=meta, blocked=None
            )
        fork_config, _fork_snap, fork_id = self._create_langgraph_fork(
            run_id=run_id,
            record=record,
            origin_snap=origin_snap,
            fork_values=fork_values,
        )
        meta = replace(meta, fork_checkpoint_id=fork_id)
        self.branches.update(meta)
        self._branch_fork_configs[_branch_key(run_id, meta.branch_id)] = fork_config
        return self.replay_downstream(
            run_id,
            meta.branch_id,
            explained_case=explained_case,
            stop_when_observable=stop_when_observable,
            stop_predicate=stop_predicate,
        )

    def fork_with_intervention(
        self,
        run_id: str,
        intervention: ArtifactIntervention,
        *,
        artifact_type: str = "",
        registry: Optional[ArtifactRegistry] = None,
        intervention_registry: Optional[ArtifactRegistry] = None,
        explained_case: Any = None,
        stop_when_observable: bool = False,
        stop_predicate: Optional[Callable[..., Any]] = None,
    ) -> WorkflowRunResult:
        """Real REMOVE / ADD / REPLACE fork-and-replay (shared LangGraph path).

        ``registry`` / ``intervention_registry`` is an optional controlled
        intervention artifact catalog for external ADD/REPLACE targets.
        """
        op = intervention.operation
        controlled = intervention_registry if intervention_registry is not None else registry
        if self.compiled_graph is None:
            return self._fixture_fork_intervention(
                run_id,
                intervention,
                artifact_type=artifact_type,
                registry=controlled,
            )
        if op == "NOOP":
            return self.fork_noop(
                run_id,
                BoundarySelector.by_alias(intervention.checkpoint.alias),
                explained_case=explained_case,
                stop_when_observable=stop_when_observable,
                stop_predicate=stop_predicate,
            )
        if op not in {"REMOVE", "ADD", "REPLACE"}:
            raise ValueError(f"Unsupported intervention operation: {op}")

        meta, fork_values, origin_snap, record = self._prepare_intervention_fork(
            run_id,
            intervention,
            artifact_type=artifact_type,
            intervention_registry=controlled,
        )
        fork_config, _fork_snap, fork_id = self._create_langgraph_fork(
            run_id=run_id,
            record=record,
            origin_snap=origin_snap,
            fork_values=fork_values,
        )
        meta = replace(meta, fork_checkpoint_id=fork_id)
        self.branches.update(meta)
        self._branch_fork_configs[_branch_key(run_id, meta.branch_id)] = fork_config

        from ..artifacts.channel_state import load_artifact_channels

        with _replay_span("fork_resolver"):
            _reg, resolver = load_artifact_channels(fork_values)
            self._resolvers[_branch_key(run_id, meta.branch_id)] = resolver

        return self.replay_downstream(
            run_id,
            meta.branch_id,
            explained_case=explained_case,
            stop_when_observable=stop_when_observable,
            stop_predicate=stop_predicate,
        )

    def replay_until_case_observable(
        self,
        run_id: str,
        branch_id: str,
        *,
        explained_case: Any,
        stop_predicate: Optional[Callable[..., Any]] = None,
    ) -> WorkflowRunResult:
        """Bounded replay: stop once the explained case outcome is observable."""
        return self.replay_downstream(
            run_id,
            branch_id,
            explained_case=explained_case,
            stop_when_observable=True,
            stop_predicate=stop_predicate,
        )

    def replay_downstream(
        self,
        run_id: str,
        branch_id: str = "",
        *,
        from_checkpoint: Optional[CheckpointRecord] = None,
        explained_case: Any = None,
        stop_when_observable: bool = False,
        stop_predicate: Optional[Callable[..., Any]] = None,
    ) -> WorkflowRunResult:
        """Continue a forked branch via real LangGraph ``invoke`` / bounded stream.

        When ``stop_when_observable`` is True and ``explained_case`` is set, the
        branch is streamed node-by-node and may stop once the case-level
        observability predicate reports COMPLETE. Full ``invoke``-to-END remains
        the default and the fallback oracle.
        """
        # Backward-compat: old API used branch_id as first positional alone
        if branch_id == "" and run_id in self._branch_fork_configs:
            # mis-ordered legacy call — not supported
            pass
        # Support old signature replay_downstream(branch_id) used in placeholders
        if (
            branch_id == ""
            and from_checkpoint is None
            and ":" not in run_id
            and self.compiled_graph is None
        ):
            # legacy placeholder path keyed by branch_id only
            key_branch = run_id
            for (rid, bid), state in self._branch_results.items():
                if bid == key_branch:
                    return WorkflowRunResult(
                        run_id=rid,
                        branch_id=bid,
                        final_state=deepcopy(state),
                    )
            if key_branch in self._factual_results:
                return WorkflowRunResult(
                    run_id=key_branch,
                    branch_id=key_branch,
                    final_state=deepcopy(self._factual_results[key_branch]),
                )
            raise KeyError(f"Unknown branch: {key_branch}")

        if self.compiled_graph is None:
            raise RuntimeError("replay_downstream requires a compiled LangGraph")

        meta = self.branches.get(run_id, branch_id)
        fork_config = self._branch_fork_configs.get(_branch_key(run_id, branch_id))
        if fork_config is None:
            raise KeyError(f"No fork config for run={run_id!r} branch={branch_id!r}")
        runtime = self._require_runtime(run_id)

        with _replay_span("replay_setup"):
            self._ensure_workflow_llm_wrapped(runtime)
            runtime = self._ensure_cross_cove_verifier_llm(runtime)
            if runtime is not None:
                self._runtimes[run_id] = runtime
            ledger = self.get_or_create_llm_ledger(run_id)
            self._bind_ledger_models(ledger, runtime)

            # Exact match only — never use `"factual" in "counterfactual"`.
            branch_type = _normalize_workflow_branch_type(
                getattr(meta, "branch_type", None),
                branch_id=branch_id,
            )
            ledger.begin_branch(branch_type)

        error = None
        raw = None
        completion_mode = "graph_end"
        stopped_after_node: Optional[str] = None
        executed_nodes: List[str] = []
        target_observable: Optional[bool] = None
        skipped_downstream: Optional[int] = None
        node_timings: List[Dict[str, Any]] = []
        self._last_node_timings = []
        self._last_node_unattributed = {}

        try:
            with workflow_llm_execution_context(
                run_id=run_id,
                branch_id=branch_id,
                branch_type=branch_type,
                ledger=ledger,
            ):
                use_bounded = bool(stop_when_observable and explained_case is not None)
                with _replay_span("graph_execution"):
                    if use_bounded:
                        (
                            raw,
                            final,
                            completion_mode,
                            stopped_after_node,
                            executed_nodes,
                            target_observable,
                            skipped_downstream,
                        ) = self._stream_until_observable(
                            fork_config=fork_config,
                            runtime=runtime,
                            branch_id=branch_id,
                            explained_case=explained_case,
                            stop_predicate=stop_predicate,
                            ledger=ledger,
                        )
                        node_timings = list(getattr(self, "_last_node_timings", None) or [])
                    else:
                        raw = self.compiled_graph.invoke(
                            None, config=fork_config, context=runtime
                        )
                        final = dict(raw) if isinstance(raw, dict) else {"result": raw}
                        completion_mode = "graph_end"
                        node_timings = []
        except WorkflowLLMReplayMissError as exc:
            error = str(exc)
            final = {}
            completion_mode = "graph_end"
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            final = {}
            completion_mode = "graph_end"

        branch_llm_diag = ledger.end_branch(branch_type)
        if isinstance(branch_llm_diag, dict):
            branch_llm_diag = dict(branch_llm_diag)
            branch_llm_diag["replay_completion_mode"] = completion_mode
            if stopped_after_node is not None:
                branch_llm_diag["stopped_after_node"] = stopped_after_node
            if executed_nodes:
                branch_llm_diag["executed_node_count"] = len(executed_nodes)

        if final and final.get("branch_id") not in (None, branch_id):
            if final.get("branch_id") != branch_id:
                raise RuntimeError(
                    f"Replay final branch_id={final.get('branch_id')!r} "
                    f"!= expected {branch_id!r}"
                )

        unattributed = dict(getattr(self, "_last_node_unattributed", None) or {})

        # Collect this branch's checkpoints by walking parent lineage from the
        # *latest* thread snapshot. Do not call get_state_history(thread_id):
        # that lists factual + every sibling fork (O(total thread history)).
        # Do not start from the pinned fork_config checkpoint_id (stale origin).
        branch_history: List[Any] = []
        snapshots_scanned = 0
        newest_first: List[Any] = []
        try:
            with _replay_span("branch_history_fetch"):
                newest_first, snapshots_scanned = self._walk_branch_parent_lineage(
                    fork_config=fork_config,
                    branch_id=branch_id,
                )
            with _replay_span("branch_history_filter"):
                branch_history = list(reversed(newest_first))
        except Exception:  # noqa: BLE001
            branch_history = []
            newest_first = []
        try:
            from ..experiment.timing import set_replay_profile_meta

            set_replay_profile_meta("history_snapshots_scanned", snapshots_scanned)
            # Unknown without a full-thread scan; do not pretend we counted it.
            set_replay_profile_meta("total_thread_history_size", 0)
            set_replay_profile_meta("branch_snapshots_retained", len(branch_history))
            set_replay_profile_meta("history_collection_mode", "parent_lineage")
        except Exception:
            pass

        final_ckpt = None
        with _replay_span("branch_metadata_update"):
            if branch_history:
                try:
                    final_ckpt = extract_checkpoint_id(branch_history[-1])
                except ValueError:
                    final_ckpt = None
            meta = replace(meta, final_checkpoint_id=final_ckpt)
            self.branches.update(meta)

        with _replay_span("final_state_store_copy"):
            self._branch_results[_branch_key(run_id, branch_id)] = deepcopy(final)
            self._branch_histories[_branch_key(run_id, branch_id)] = branch_history

        from ..artifacts.channel_state import load_artifact_channels

        with _replay_span("artifact_resolver_rebuild"):
            try:
                _reg, resolver = load_artifact_channels(final)
                self._resolvers[_branch_key(run_id, branch_id)] = resolver
            except Exception:
                pass

        origin = None
        with _replay_span("origin_checkpoint_lookup"):
            try:
                origin = self.get_checkpoint_index(run_id).get_alias(
                    meta.origin_checkpoint_alias
                )
            except Exception:
                origin = None

        with _replay_span("workflow_result_materialization"):
            try:
                from ..experiment.timing import current_replay_profile

                span_prof = dict(current_replay_profile() or {})
            except Exception:
                span_prof = {}
            span_prof.setdefault("history_snapshots_scanned", snapshots_scanned)
            span_prof.setdefault("total_thread_history_size", 0)
            span_prof.setdefault("branch_snapshots_retained", len(branch_history))
            span_prof.setdefault("history_collection_mode", "parent_lineage")
            result = WorkflowRunResult(
                run_id=run_id,
                branch_id=branch_id,
                final_state=final,
                step_logs=list(final.get("step_logs") or []),
                checkpoint_index=self._factual_indices.get(run_id),
                error=error,
                raw_result=raw,
                branch_metadata=meta,
                branch_history=branch_history,
                origin_checkpoint=origin,
                llm_replay_diagnostics=dict(branch_llm_diag or {}),
                replay_completion_mode=completion_mode,
                stopped_after_node=stopped_after_node,
                skipped_downstream_steps=skipped_downstream,
                target_observable=target_observable,
                executed_nodes=list(executed_nodes),
                node_timings=list(node_timings),
                node_profile_unattributed=unattributed,
                replay_span_profile=span_prof,
            )
        return result

    def _resolve_stop_predicate(
        self,
        stop_predicate: Optional[Callable[..., Any]],
        *,
        runtime: Any = None,
    ) -> Callable[..., Any]:
        if stop_predicate is not None:
            return stop_predicate
        case_def = getattr(runtime, "case_definition", None) if runtime is not None else None
        if case_def is not None and hasattr(case_def, "is_replay_observable"):
            return case_def.is_replay_observable
        from ..cases.mcq.replay_observable import default_mcq_stop_predicate

        return default_mcq_stop_predicate

    @staticmethod
    def _thread_config(fork_config: Dict[str, Any]) -> Dict[str, Any]:
        """Config that resolves the *latest* checkpoint for a thread.

        ``update_state`` returns a config with a pinned ``checkpoint_id`` for the
        fork origin. ``get_state(that_config)`` re-reads that stale snapshot —
        never the post-node replay state. Drop ``checkpoint_id`` so LangGraph
        returns the newest checkpoint on the thread.
        """
        cfg = deepcopy(dict(fork_config or {}))
        configurable = dict(cfg.get("configurable") or {})
        configurable.pop("checkpoint_id", None)
        cfg["configurable"] = configurable
        return cfg

    def _walk_branch_parent_lineage(
        self,
        *,
        fork_config: Dict[str, Any],
        branch_id: str,
        max_walk: int = 2000,
    ) -> Tuple[List[Any], int]:
        """Return (newest-first branch snapshots, snapshots inspected).

        Starts at the latest thread checkpoint (no pinned origin ``checkpoint_id``),
        then follows ``parent_config``. Stops at the first snapshot whose
        ``values.branch_id`` is not this branch (the factual parent). Work is
        O(current branch length), not O(accumulated sibling history).
        """
        latest_cfg = self._thread_config(fork_config)
        snap = self.compiled_graph.get_state(latest_cfg)
        scanned = 0
        newest_first: List[Any] = []
        seen: Set[str] = set()
        while snap is not None and scanned < max_walk:
            scanned += 1
            try:
                cid = extract_checkpoint_id(snap)
            except ValueError:
                cid = f"missing-id:{scanned}"
            if cid in seen:
                break
            seen.add(cid)
            vals = getattr(snap, "values", None) or {}
            if vals.get("branch_id") == branch_id:
                newest_first.append(snap)
            else:
                break
            parent_cfg = getattr(snap, "parent_config", None)
            if not parent_cfg:
                break
            try:
                snap = self.compiled_graph.get_state(parent_cfg)
            except Exception:
                break
        return newest_first, scanned

    def _invoke_from_latest(
        self,
        *,
        fork_config: Dict[str, Any],
        runtime: Any,
    ) -> Any:
        """Continue invoke from the latest thread checkpoint (not the pinned fork)."""
        latest_cfg = self._thread_config(fork_config)
        try:
            return self.compiled_graph.invoke(
                None, config=latest_cfg, context=runtime
            )
        except TypeError:
            return self.compiled_graph.invoke(None, config=latest_cfg)

    def _stream_until_observable(
        self,
        *,
        fork_config: Dict[str, Any],
        runtime: Any,
        branch_id: str,
        explained_case: Any,
        stop_predicate: Optional[Callable[..., Any]],
        ledger: Any = None,
    ) -> Tuple[Any, Dict[str, Any], str, Optional[str], List[str], Optional[bool], Optional[int]]:
        """Stream graph values; stop on COMPLETE or fall back to invoke-to-END.

        Uses ``stream_mode=["updates", "values"]`` so observability checks see the
        actual post-node state. Never call ``get_state(fork_config)`` for this —
        the fork config pins the origin checkpoint_id.
        """
        from ..cases.mcq.replay_observable import ReplayObservableStatus

        predicate = self._resolve_stop_predicate(stop_predicate, runtime=runtime)
        executed_nodes: List[str] = []
        stopped_after: Optional[str] = None
        last_values: Dict[str, Any] = {}
        completion_mode = "graph_end"
        target_observable: Optional[bool] = None
        saw_update = False
        node_timings: List[Dict[str, Any]] = []
        unattributed: Dict[str, Any] = {
            "attribution_scope": "unattributed",
            "logical_llm_requests": 0,
            "factual_ledger_hits": 0,
            "cf_memo_hits": 0,
            "provider_calls": 0,
            "provider_time_s": 0.0,
        }
        node_t0 = time.perf_counter()
        snap_before: Optional[Dict[str, int]] = None
        provider_before = 0.0
        try:
            from ..experiment.timing import current_timing as _current_timing
        except Exception:
            _current_timing = lambda: None  # type: ignore[assignment]

        def _llm_snap() -> Dict[str, int]:
            if ledger is None or not hasattr(ledger, "_counter_snapshot"):
                return {}
            return dict(ledger._counter_snapshot())

        def _provider_time() -> float:
            timing = _current_timing()
            if timing is None:
                return 0.0
            return float((timing.workflow_llm or {}).get("provider_time_s") or 0.0)

        def _llm_delta(after: Dict[str, int], before: Dict[str, int]) -> Dict[str, int]:
            return {
                "logical_llm_requests": int(after.get("counterfactual_llm_request_count") or 0)
                - int(before.get("counterfactual_llm_request_count") or 0),
                "factual_ledger_hits": int(
                    after.get("counterfactual_factual_response_hits") or 0
                )
                - int(before.get("counterfactual_factual_response_hits") or 0),
                "cf_memo_hits": int(after.get("counterfactual_memo_hits") or 0)
                - int(before.get("counterfactual_memo_hits") or 0),
                "provider_calls": int(after.get("counterfactual_live_provider_calls") or 0)
                - int(before.get("counterfactual_live_provider_calls") or 0),
            }

        def _flush_completed(names: List[str]) -> None:
            """Assign LLM/time delta to the node(s) that just completed.

            LangGraph ``updates`` events are post-node: the payload keys are the
            nodes that finished, and their LLM work is already in the counters.
            """
            nonlocal node_t0, snap_before, provider_before
            if not names:
                return
            elapsed = time.perf_counter() - node_t0
            after = _llm_snap()
            before = snap_before or {}
            provider_after = _provider_time()
            counts = _llm_delta(after, before)
            provider_s = max(0.0, provider_after - provider_before)
            sequential = len(names) == 1
            entry = {
                "node": names[0] if sequential else "+".join(names),
                "attribution_scope": "node" if sequential else "superstep",
                "nodes": list(names),
                "elapsed_s": float(max(0.0, elapsed)),
                "logical_llm_requests": int(counts["logical_llm_requests"]),
                "factual_ledger_hits": int(counts["factual_ledger_hits"]),
                "cf_memo_hits": int(counts["cf_memo_hits"]),
                "provider_calls": int(counts["provider_calls"]),
                "provider_time_s": float(provider_s),
            }
            node_timings.append(entry)
            node_t0 = time.perf_counter()
            snap_before = after
            provider_before = provider_after

        def _absorb_unattributed() -> None:
            nonlocal node_t0, snap_before, provider_before
            after = _llm_snap()
            before = snap_before or {}
            provider_after = _provider_time()
            counts = _llm_delta(after, before)
            unattributed["logical_llm_requests"] += int(counts["logical_llm_requests"])
            unattributed["factual_ledger_hits"] += int(counts["factual_ledger_hits"])
            unattributed["cf_memo_hits"] += int(counts["cf_memo_hits"])
            unattributed["provider_calls"] += int(counts["provider_calls"])
            unattributed["provider_time_s"] += max(0.0, provider_after - provider_before)
            snap_before = after
            provider_before = provider_after
            node_t0 = time.perf_counter()

        snap_before = _llm_snap()
        provider_before = _provider_time()
        self._last_node_timings = node_timings
        self._last_node_unattributed = unattributed

        def _open_stream():
            try:
                return self.compiled_graph.stream(
                    None,
                    config=fork_config,
                    context=runtime,
                    stream_mode=["updates", "values"],
                )
            except TypeError:
                try:
                    return self.compiled_graph.stream(
                        None,
                        config=fork_config,
                        stream_mode=["updates", "values"],
                    )
                except TypeError:
                    # Very old API: values-only stream (no node-name diagnostics).
                    try:
                        return self.compiled_graph.stream(
                            None,
                            config=fork_config,
                            context=runtime,
                            stream_mode="values",
                        )
                    except TypeError:
                        return self.compiled_graph.stream(
                            None,
                            config=fork_config,
                            stream_mode="values",
                        )

        fallback = False
        try:
            stream_iter = _open_stream()
        except Exception:
            fallback = True
            stream_iter = iter(())

        for item in stream_iter:
            mode: Optional[str] = None
            payload: Any = item
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                mode, payload = item[0], item[1]
            elif isinstance(item, dict) and (
                "updates" in item or "values" in item
            ):
                # Some adapters yield a mapping of mode -> chunk.
                if "updates" in item and item["updates"] is not None:
                    mode, payload = "updates", item["updates"]
                elif "values" in item:
                    mode, payload = "values", item["values"]
            else:
                # values-only stream yields state dicts directly.
                mode, payload = "values", item

            if mode == "updates":
                if isinstance(payload, dict):
                    names = [str(node_name) for node_name in payload.keys()]
                    # ``updates`` is a post-node event: these nodes just finished,
                    # and their LLM counters are already updated.
                    _flush_completed(names)
                    executed_nodes.extend(names)
                    if names:
                        stopped_after = names[-1]
                    saw_update = True
                continue

            if mode != "values" or not isinstance(payload, dict):
                continue

            values = dict(payload)
            last_values = values
            # Skip the initial fork values emission before any node runs.
            if not saw_update and not executed_nodes:
                continue

            try:
                with _replay_span("observability_check"):
                    status = predicate(values, explained_case, branch_id=branch_id)
            except TypeError:
                try:
                    status = predicate(values, explained_case)
                except Exception:
                    fallback = True
                    break
            except Exception:
                fallback = True
                break

            # Observability is nested inside graph_execution; do not attribute
            # that gap to the next node's elapsed_s.
            node_t0 = time.perf_counter()

            if status == ReplayObservableStatus.FALLBACK_FULL or status == "fallback_full":
                fallback = True
                break
            if status == ReplayObservableStatus.COMPLETE or status is True or status == "complete":
                completion_mode = "target_observable"
                target_observable = True
                break

        self._last_node_timings = node_timings
        self._last_node_unattributed = unattributed

        raw: Any
        if fallback:
            try:
                # If the stream already advanced the branch, continue from latest.
                # If it failed before any progress, latest still equals the fork.
                raw = self._invoke_from_latest(
                    fork_config=fork_config, runtime=runtime
                )
            except Exception:
                raw = self.compiled_graph.invoke(
                    None, config=fork_config, context=runtime
                ) if runtime is not None else self.compiled_graph.invoke(
                    None, config=fork_config
                )
            _absorb_unattributed()
            self._last_node_unattributed = unattributed
            final = dict(raw) if isinstance(raw, dict) else {"result": raw}
            completion_mode = "fallback_full"
            target_observable = False
            return (
                raw,
                final,
                completion_mode,
                stopped_after,
                executed_nodes,
                target_observable,
                None,
            )

        if completion_mode == "target_observable":
            final = dict(last_values)
            raw = final
            skipped = None
            cs = final.get("cooperative_state")
            if cs is not None:
                try:
                    from ..domains.quiz.state_integration import ensure_step2_mcq_state

                    ensure_step2_mcq_state(cs)
                    skipped = len(cs.planned_blueprints())
                except Exception:
                    skipped = None
            return (
                raw,
                final,
                completion_mode,
                stopped_after,
                executed_nodes,
                True,
                skipped,
            )

        # Stream exhausted without early stop — last values chunk is the true END state.
        final = dict(last_values) if last_values else {}
        if not final:
            try:
                snap = self.compiled_graph.get_state(self._thread_config(fork_config))
                final = dict(getattr(snap, "values", None) or {})
            except Exception:
                final = {}
        raw = final
        return (
            raw,
            final,
            "graph_end",
            stopped_after,
            executed_nodes,
            False,
            None,
        )

    def get_resolver(self, branch_id: str, run_id: Optional[str] = None) -> ArtifactResolver:
        """Resolve convenience ArtifactResolver.

        Prefer ``get_resolver(branch_id, run_id=...)``. Legacy calls with only
        ``branch_id`` search all runs; ``run_id`` as sole key still maps factual.
        """
        if run_id is not None:
            key = _branch_key(run_id, branch_id)
            if key not in self._resolvers:
                raise KeyError(f"No resolver for {key}")
            return self._resolvers[key]
        # factual run_id keyed historically
        if branch_id in self._factual_results:
            key = _branch_key(branch_id, "factual")
            if key in self._resolvers:
                return self._resolvers[key]
        for (rid, bid), resolver in self._resolvers.items():
            if bid == branch_id:
                return resolver
        # legacy: key was run_id for factual resolver
        key = _branch_key(branch_id, "factual")
        if key in self._resolvers:
            return self._resolvers[key]
        raise KeyError(f"No resolver for branch {branch_id}")

    # --- Fixture / resolver-only path (compiled_graph is None) -----------------

    def _fixture_fork_noop(
        self, run_id: str, selector: BoundarySelector
    ) -> WorkflowRunResult:
        ckpt = self.load_checkpoint(run_id, selector)
        parent = self.branches.get(run_id, "factual")
        meta = self.branches.fork(
            parent=parent,
            branch_type="noop",
            origin_checkpoint_alias=ckpt.alias,
            origin_checkpoint_id=ckpt.checkpoint_id,
        )
        factual_state = deepcopy(self._factual_results[run_id])
        factual_state["branch_id"] = meta.branch_id
        factual_state["branch_type"] = "noop"
        self._branch_results[_branch_key(run_id, meta.branch_id)] = factual_state
        base = self._resolvers.get(_branch_key(run_id, "factual"))
        self._resolvers[_branch_key(run_id, meta.branch_id)] = deepcopy(
            base or ArtifactResolver(registry=ArtifactRegistry())
        )
        return WorkflowRunResult(
            run_id=run_id,
            branch_id=meta.branch_id,
            final_state=factual_state,
            checkpoint_index=self._factual_indices[run_id],
            branch_metadata=meta,
            origin_checkpoint=ckpt,
        )

    def _fixture_fork_intervention(
        self,
        run_id: str,
        intervention: ArtifactIntervention,
        *,
        artifact_type: str = "",
        registry: Optional[ArtifactRegistry] = None,
    ) -> WorkflowRunResult:
        """Legacy resolver-only REMOVE/ADD/REPLACE for unit fixtures without a graph."""
        ckpt = intervention.checkpoint
        try:
            ckpt = self._validate_intervention_checkpoint(run_id, ckpt)
        except Exception:
            # fixture tests may use synthetic records matching alias only
            ckpt = self.load_checkpoint(
                run_id, BoundarySelector.by_alias(intervention.checkpoint.alias)
            )
        parent = self.branches.get(run_id, "factual")
        meta = self.branches.fork(
            parent=parent,
            branch_type="counterfactual",
            origin_checkpoint_alias=ckpt.alias,
            origin_checkpoint_id=ckpt.checkpoint_id,
            intervention_id=intervention.intervention_id,
        )
        base_registry = registry or ArtifactRegistry()
        # Fixture catalogs use factual branch envelopes; keep lookup branch factual.
        resolver = ArtifactResolver(registry=base_registry, branch_id="factual")
        for ref in base_registry.all_refs():
            resolver.make_available(ref)
        op = intervention.operation
        if artifact_type and op != "NOOP":
            self.contracts.validate(intervention, artifact_type)
        if op == "REMOVE":
            assert isinstance(intervention.artifact, ArtifactRef)
            resolver.block(intervention.artifact)
        elif op == "ADD":
            assert isinstance(intervention.artifact, ArtifactRef)
            if base_registry.get_by_ref(intervention.artifact) is None:
                raise ValueError(
                    f"Cannot ADD unknown artifact {intervention.artifact}; "
                    "must exist in catalog or fixture"
                )
            resolver.make_available(intervention.artifact, override_block=True)
        elif op == "REPLACE":
            old, new = intervention.artifact  # type: ignore[misc]
            if base_registry.get_by_ref(new) is None:
                raise ValueError(f"REPLACE target not in catalog: {new}")
            resolver.substitute(old, new)
        factual_state = deepcopy(self._factual_results[run_id])
        factual_state["branch_id"] = meta.branch_id
        factual_state["branch_type"] = "counterfactual"
        self._branch_results[_branch_key(run_id, meta.branch_id)] = factual_state
        self._resolvers[_branch_key(run_id, meta.branch_id)] = resolver
        return WorkflowRunResult(
            run_id=run_id,
            branch_id=meta.branch_id,
            final_state=factual_state,
            checkpoint_index=self._factual_indices.get(run_id),
            branch_metadata=meta,
            origin_checkpoint=ckpt,
        )


# typing helper for Mapping-like origin values
MappingLike = Dict[str, Any]


@dataclass
class _SyntheticSnapshot:
    """Minimal stand-in when LangGraph history is unavailable (unit tests only)."""

    index: int
    thread_id: str
    values: Dict[str, Any]
    parent_index: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    next: tuple = ()

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {"step": self.index * 10}

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "configurable": {
                "thread_id": self.thread_id,
                "checkpoint_id": f"synth_{self.thread_id}_{self.index}",
                "checkpoint_ns": "",
            }
        }

    @property
    def parent_config(self) -> Optional[Dict[str, Any]]:
        if self.parent_index is None:
            return None
        return {
            "configurable": {
                "thread_id": self.thread_id,
                "checkpoint_id": f"synth_{self.thread_id}_{self.parent_index}",
                "checkpoint_ns": "",
            }
        }

    @property
    def id(self) -> str:
        return f"synth_{self.thread_id}_{self.index}"
