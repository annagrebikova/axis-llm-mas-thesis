"""Checkpointed provenance channel helpers — branch-local, copy-safe."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, MutableMapping, Optional

from ..artifacts.models import ArtifactRef
from .models import ProvenanceSchema
from .recorder import ProvenanceRecorder


def empty_provenance_channels() -> Dict[str, Any]:
    """Default checkpointed provenance fields for a new workflow branch."""
    return {
        "occurrence_records": [],
        "produced_by": [],
        "run_provenance_relations": [],
        "provenance_violations": [],
        "provenance_unresolved_producers": [],
        "provenance_node_counts": {},
        "provenance_execution_index": 0,
    }


def ensure_provenance_channels(state: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Ensure graph/checkpoint state contains provenance channel defaults."""
    merged = dict(state)
    defaults = empty_provenance_channels()
    for key, default in defaults.items():
        merged.setdefault(key, deepcopy(default))
    return merged


def clone_provenance_channels_dict(state: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Deep-copy provenance channels from graph state for a node transition."""
    return {
        "occurrence_records": list(state.get("occurrence_records") or []),
        "produced_by": list(state.get("produced_by") or []),
        "run_provenance_relations": list(state.get("run_provenance_relations") or []),
        "provenance_violations": list(state.get("provenance_violations") or []),
        "provenance_unresolved_producers": list(
            state.get("provenance_unresolved_producers") or []
        ),
        "provenance_node_counts": dict(state.get("provenance_node_counts") or {}),
        "provenance_execution_index": int(state.get("provenance_execution_index") or 0),
    }


def _artifact_types_from_catalog(state: MutableMapping[str, Any]) -> Dict[ArtifactRef, str]:
    types: Dict[ArtifactRef, str] = {}
    catalog = state.get("artifact_catalog") or {}
    branch_id = str(state.get("branch_id") or "factual")
    for env in catalog.values():
        if getattr(env, "branch_id", branch_id) != branch_id:
            continue
        ref = getattr(env, "ref", None)
        atype = getattr(env, "artifact_type", "") or ""
        if ref is not None and atype:
            types[ref] = atype
    return types


def load_provenance_recorder(
    state: MutableMapping[str, Any],
    *,
    schema: Optional[ProvenanceSchema] = None,
    run_id: Optional[str] = None,
    branch_id: Optional[str] = None,
) -> ProvenanceRecorder:
    """Build a ProvenanceRecorder view over checkpointed provenance channels."""
    channels = clone_provenance_channels_dict(state)
    return ProvenanceRecorder.from_state(
        occurrence_records=channels["occurrence_records"],
        produced_by=channels["produced_by"],
        run_provenance_relations=channels["run_provenance_relations"],
        violations=channels["provenance_violations"],
        unresolved_producers=channels["provenance_unresolved_producers"],
        node_counts=channels["provenance_node_counts"],
        execution_index=channels["provenance_execution_index"],
        artifact_types=_artifact_types_from_catalog(state),
        run_id=str(run_id if run_id is not None else state.get("run_id") or ""),
        branch_id=str(branch_id if branch_id is not None else state.get("branch_id") or "factual"),
        schema=schema or ProvenanceSchema(),
    )


def dump_provenance_channels(recorder: ProvenanceRecorder) -> Dict[str, Any]:
    """Serialize recorder back into checkpointed graph fields."""
    return recorder.to_state()
