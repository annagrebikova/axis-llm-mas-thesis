"""Central LangGraph node instrumentation for occurrence-level provenance."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Set, runtime_checkable

from langgraph.runtime import Runtime

from ..artifacts.channel_state import load_artifact_channels
from ..artifacts.models import ArtifactRef
from ..artifacts.registry import ArtifactRegistry
from ..artifacts.resolver import ArtifactResolver
from ..core.runtime import WorkflowRuntime
from ..provenance.channel_state import dump_provenance_channels, load_provenance_recorder
from ..provenance.models import ProvenanceSchema


@runtime_checkable
class ProvenanceInstrumentationAdapter(Protocol):
    """Case-specific bindings and compact state summaries for provenance."""

    def input_bindings_for_node(
        self,
        node_id: str,
        state: Mapping[str, Any],
        registry: ArtifactRegistry,
        *,
        branch_id: str,
        resolver: ArtifactResolver,
    ) -> Sequence[Any]: ...

    def snapshot_state_summary(
        self,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def compact_state_delta(
        self,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        *,
        new_artifact_refs: Sequence[ArtifactRef],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]: ...


def _catalog_ref_keys(catalog: Dict[str, Any], branch_id: str) -> Set[str]:
    keys: Set[str] = set()
    for env in (catalog or {}).values():
        if getattr(env, "branch_id", branch_id) != branch_id:
            continue
        ref = getattr(env, "ref", None)
        if ref is None:
            continue
        keys.add(f"{ref.logical_id}:v{ref.version}")
    return keys


def _new_refs_from_catalog_delta(
    before: Dict[str, Any],
    after: Dict[str, Any],
    branch_id: str,
) -> list[ArtifactRef]:
    before_keys = _catalog_ref_keys(before, branch_id)
    produced: list[ArtifactRef] = []
    seen: Set[str] = set()
    for env in (after or {}).values():
        if getattr(env, "branch_id", branch_id) != branch_id:
            continue
        ref = getattr(env, "ref", None)
        if ref is None:
            continue
        key = f"{ref.logical_id}:v{ref.version}"
        if key in before_keys or key in seen:
            continue
        seen.add(key)
        produced.append(ref)
    return produced


def _load_schema(runtime: Runtime[WorkflowRuntime]) -> ProvenanceSchema:
    case = getattr(runtime.context, "case_definition", None)
    if case is None:
        raise RuntimeError(
            "No case_definition configured on the workflow runtime; "
            "provenance instrumentation requires an active case"
        )
    schema = case.load_provenance_schema()
    if schema is None or not getattr(schema, "relations", None):
        raise RuntimeError(
            "Active case provenance schema is missing or empty; "
            "refusing permissive empty schema on the factual path"
        )
    return schema


def _load_instrumentation_adapter(runtime: Runtime[WorkflowRuntime]) -> Any:
    case = getattr(runtime.context, "case_definition", None)
    if case is None:
        raise RuntimeError(
            "No case_definition configured on the workflow runtime; "
            "provenance instrumentation requires an active case"
        )
    if not hasattr(case, "load_provenance_instrumentation_adapter"):
        raise RuntimeError(
            "Active case does not provide load_provenance_instrumentation_adapter()"
        )
    return case.load_provenance_instrumentation_adapter()


def instrument_node(
    node_id: str,
    node_fn: Callable[..., Dict[str, Any]],
) -> Callable[..., Dict[str, Any]]:
    """Wrap a LangGraph node to record occurrence-level provenance.

    Does not change domain behavior — only appends provenance channel updates.
    """

    def wrapped(
        state: Dict[str, Any],
        runtime: Runtime[WorkflowRuntime],
    ) -> Dict[str, Any]:
        adapter = _load_instrumentation_adapter(runtime)
        branch_id = str(state.get("branch_id") or "factual")
        run_id = str(state.get("run_id") or "")
        schema = _load_schema(runtime)
        registry, resolver = load_artifact_channels(state)
        recorder = load_provenance_recorder(
            state, schema=schema, run_id=run_id, branch_id=branch_id
        )
        from ..replay.branch_artifacts import intervention_supplied_refs

        recorder.intervention_supplied_refs = intervention_supplied_refs(state)

        catalog_before = dict(state.get("artifact_catalog") or {})
        counts_before = dict(adapter.snapshot_state_summary(state) or {})
        bindings = adapter.input_bindings_for_node(
            node_id, state, registry, branch_id=branch_id, resolver=resolver
        )
        partial = recorder.begin_occurrence(node_id, input_bindings=bindings)

        error: Optional[str] = None
        update: Dict[str, Any] = {}
        raised: Optional[BaseException] = None
        try:
            update = node_fn(state, runtime) or {}
        except BaseException as exc:
            raised = exc
            error = str(exc)
            update = {"error": error}

        after_state = dict(state)
        after_state.update(update)
        catalog_after = update.get("artifact_catalog", state.get("artifact_catalog") or {})
        counts_after = dict(adapter.snapshot_state_summary(after_state) or {})

        produced = _new_refs_from_catalog_delta(catalog_before, catalog_after, branch_id)
        type_map: Dict[ArtifactRef, str] = {}
        for env in (catalog_after or {}).values():
            ref = getattr(env, "ref", None)
            atype = getattr(env, "artifact_type", "") or ""
            if ref is not None and atype:
                type_map[ref] = atype
                recorder.register_artifact_type(ref, atype)

        delta = adapter.compact_state_delta(
            counts_before,
            counts_after,
            new_artifact_refs=produced,
            extra={"error": error} if error else None,
        )
        recorder.complete_occurrence(
            partial,
            produced_artifacts=produced,
            state_delta=delta,
            error=error or (update.get("error") if isinstance(update, dict) else None),
            producer_artifact_types=type_map,
        )
        update = dict(update)
        update.update(dump_provenance_channels(recorder))
        if raised is not None:
            raise raised
        return update

    wrapped.__name__ = getattr(node_fn, "__name__", node_id)
    wrapped.__qualname__ = getattr(node_fn, "__qualname__", node_id)
    return wrapped
