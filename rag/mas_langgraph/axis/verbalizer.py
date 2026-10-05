"""Generic verbalization interfaces for AXIS explanation context.

The concrete MCQ verbalizer lives in ``mas_langgraph.cases.mcq.verbalizer``.
Generic AXIS code must not import case-specific verbalizers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..artifacts.models import ArtifactEnvelope
from ..provenance.models import OccurrenceRecord
from ..replay.checkpoints import CheckpointRecord
from .models import ExplainedCase, FactualQueryRecord, InterventionQueryRecord
from .slicing import CaseSlice


@runtime_checkable
class AxisVerbalizer(Protocol):
    """Case-agnostic verbalization surface."""

    def verbalize_artifact(self, envelope: ArtifactEnvelope) -> str: ...

    def verbalize_occurrence(self, occurrence: OccurrenceRecord) -> str: ...

    def verbalize_checkpoint(self, checkpoint: CheckpointRecord) -> str: ...

    def verbalize_slice(self, case_slice: CaseSlice) -> str: ...

    def verbalize_decision(self, case: ExplainedCase) -> str: ...

    def verbalize_factual_query(self, record: FactualQueryRecord) -> str: ...

    def verbalize_intervention(self, record: InterventionQueryRecord) -> str: ...


class _GenericVerbalizer:
    """Minimal case-neutral fallback verbalizer (no MCQ terminology)."""

    def verbalize_artifact(self, envelope: ArtifactEnvelope) -> str:
        return f"{envelope.artifact_type} {envelope.ref} (branch={envelope.branch_id})"

    def verbalize_occurrence(self, occurrence: OccurrenceRecord) -> str:
        return (
            f"Occurrence {occurrence.labeled_id} "
            f"(id={occurrence.occurrence_id}, branch={occurrence.branch_id})"
        )

    def verbalize_checkpoint(self, checkpoint: CheckpointRecord) -> str:
        return f"Checkpoint {checkpoint.alias} (id={checkpoint.checkpoint_id})"

    def verbalize_slice(self, case_slice: CaseSlice) -> str:
        refs = ", ".join(str(r) for r in case_slice.relevant_artifact_refs[:12])
        return f"Slice for {case_slice.explained_case.artifact_ref}: [{refs}]"

    def verbalize_decision(self, case: ExplainedCase) -> str:
        return f"Decision {case.decision_ref}: {case.outcome}"

    def verbalize_factual_query(self, record: FactualQueryRecord) -> str:
        if record.error:
            return f"Factual query failed: {record.error}"
        return f"WHAT result for {record.resolved.target_ref}: {record.result!r}"

    def verbalize_intervention(self, record: InterventionQueryRecord) -> str:
        if record.error:
            return f"Intervention failed: {record.error}"
        return (
            f"Intervention {record.resolved.operation} "
            f"on {record.resolved.target_ref} at {record.resolved.checkpoint_alias}"
        )


GenericVerbalizer = _GenericVerbalizer
