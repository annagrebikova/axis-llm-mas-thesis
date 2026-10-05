"""Wire checkpointed artifact channels into live LangGraph node transitions."""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

from ..artifacts.channel_state import dump_artifact_channels, load_artifact_channels
from ..cases.mcq.artifact_registration import register_mcq_artifacts_for_stage
from ..cases.mcq.cooperative_state import CooperativeState


def commit_artifact_channels(
    input_state: MutableMapping[str, Any],
    working_cs: CooperativeState,
    artifact_stage: Optional[str] = None,
) -> Dict[str, Any]:
    """Clone artifact channels, optionally register new MCQ snapshots, return graph update."""
    registry, resolver = load_artifact_channels(input_state)
    if artifact_stage:
        register_mcq_artifacts_for_stage(
            registry,
            resolver,
            working_cs,
            stage=artifact_stage,
            run_id=str(input_state.get("run_id") or ""),
            branch_id=str(input_state.get("branch_id") or "factual"),
        )
    return dump_artifact_channels(
        registry,
        resolver,
        intervention_events=list(input_state.get("intervention_events") or []),
    )
