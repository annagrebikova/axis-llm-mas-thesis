"""Copy-safe workflow state transitions for checkpoint replay (Prompt 4)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, MutableMapping, Tuple

from ..artifacts.channel_state import clone_artifact_channels_dict
from ..provenance.channel_state import clone_provenance_channels_dict


_REBIND_HELPER_ATTRS = (
    "snapshot_counts",
    "update_section_evidence_status",
    "get_sections_without_evidence",
    "get_sections_with_sufficient_evidence",
    "store_mcq_blueprints",
    "planned_blueprints",
    "store_candidate_mcq",
    "store_candidate_mcqs",
    "store_evidence_report",
    "store_evidence_reports",
    "store_supervisor_decision",
    "store_outcome_assessment",
    "store_mcq_coverage_summary",
    "store_cross_cove_report",
    "store_cross_cove_reports",
    "add_revision_record",
    "get_revision_count",
)


def _rebind_cloned_instance_helpers(cloned: Any) -> None:
    """Deepcopy copies patched closures that still point at the source instance.

    RAG ``CooperativeState`` attaches ``snapshot_counts`` / Step-2 helpers as
    instance attributes. After clone those functions would report the *original*
    blackboard, which is how CLI summaries showed outline=10 while retrieval
    saw an empty clone.

    Checkpoint deserialize can also rebuild a clean RAG blackboard without any
    instance helpers; re-attach in that case so LangGraph nodes can continue.
    """
    inst = getattr(cloned, "__dict__", None)
    if not isinstance(inst, dict):
        return
    has_instance_snapshot = "snapshot_counts" in inst
    snapshot_ok = callable(getattr(cloned, "snapshot_counts", None))
    if not has_instance_snapshot and snapshot_ok:
        # Pure class-based blackboard (e.g. cases.mcq.CooperativeState).
        return
    for name in _REBIND_HELPER_ATTRS:
        if name in inst:
            try:
                delattr(cloned, name)
            except AttributeError:
                pass
    from ..integrations.rag_agents_bridge import attach_snapshot_counts

    attach_snapshot_counts(cloned)


def clone_cooperative_state(state: Any) -> Any:
    """Return a structurally independent copy of a cooperative blackboard."""
    cloned = deepcopy(state)
    _rebind_cloned_instance_helpers(cloned)
    return cloned


def clone_workflow_state(state: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Clone checkpoint-relevant workflow fields (not runtime services)."""
    artifact_keys = {
        "artifact_catalog",
        "available_artifact_refs",
        "blocked_artifact_refs",
        "artifact_substitutions",
        "artifact_derivations",
    }
    provenance_keys = {
        "occurrence_records",
        "produced_by",
        "run_provenance_relations",
        "provenance_violations",
        "provenance_unresolved_producers",
        "provenance_node_counts",
        "provenance_execution_index",
    }
    skip = artifact_keys | provenance_keys
    cloned: Dict[str, Any] = {}
    for key, value in state.items():
        if key in skip:
            continue
        if key == "cooperative_state" and value is not None:
            cloned[key] = clone_cooperative_state(value)
        elif key == "searched_queries" and value is not None:
            cloned[key] = set(value)
        elif key == "step_logs":
            cloned[key] = list(value or [])
        else:
            cloned[key] = deepcopy(value)
    cloned.update(clone_artifact_channels_dict(state))
    cloned.update(clone_provenance_channels_dict(state))
    return cloned


def prepare_state_transition(
    state: MutableMapping[str, Any],
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Clone cooperative state for a mutating node; capture before-summary."""
    input_cs = state["cooperative_state"]
    if hasattr(input_cs, "snapshot_counts"):
        state_before = deepcopy(input_cs.snapshot_counts())
    else:
        state_before = {}
    working_cs = clone_cooperative_state(input_cs)
    return input_cs, working_cs, state_before
