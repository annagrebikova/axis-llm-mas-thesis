"""Provenance data models — case-agnostic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..artifacts.models import ArtifactRef


@dataclass(frozen=True)
class ArtifactBinding:
    """How an occurrence consumed an artifact under a named semantic relation."""

    artifact_ref: ArtifactRef
    relation: str


@dataclass(frozen=True)
class OccurrenceRecord:
    """One execution of a node (occurrence), distinct from node identity.

    Repeated executions of the same node become ``node_id#1``, ``node_id#2``, …
    ``execution_index`` is the global order within the factual branch run.
    ``checkpoint_before`` / ``checkpoint_after`` / ``superstep`` are derived
    after the factual run by ``CheckpointIndex`` (Prompt 9) and are not
    fabricated inside node wrappers.
    """

    occurrence_id: str
    node_id: str
    occurrence_index: int
    branch_id: str
    run_id: str = ""
    execution_index: int = 0
    checkpoint_before: Optional[str] = None
    checkpoint_after: Optional[str] = None
    superstep: Optional[int] = None
    input_bindings: Tuple[ArtifactBinding, ...] = ()
    produced_artifacts: Tuple[ArtifactRef, ...] = ()
    state_delta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def labeled_id(self) -> str:
        """Human-readable occurrence label such as ``verify_...#1``."""
        return f"{self.node_id}#{self.occurrence_index}"


@dataclass(frozen=True)
class SchemaRelation:
    """One permitted producer–artifact–relation–consumer tuple."""

    producer: str
    artifact_type: str
    relation: str
    consumer: str


@dataclass
class ProvenanceSchema:
    """Schema-level allowed provenance relations (D_schema)."""

    relations: List[SchemaRelation] = field(default_factory=list)

    def allows(
        self,
        producer: str,
        artifact_type: str,
        relation: str,
        consumer: str,
    ) -> bool:
        for r in self.relations:
            if (
                r.producer == producer
                and r.artifact_type == artifact_type
                and r.relation == relation
                and r.consumer == consumer
            ):
                return True
        return False

    def validate_runtime(
        self,
        producer: str,
        artifact_type: str,
        relation: str,
        consumer: str,
    ) -> Optional[str]:
        """Return a violation message, or None if valid.

        Empty schema accepts all relations (permissive for generic unit fixtures).
        Live MCQ execution must load a non-empty case schema.
        """
        if not self.relations:
            return None
        if self.allows(producer, artifact_type, relation, consumer):
            return None
        return (
            f"Schema violation: ({producer}, {artifact_type}, {relation}, {consumer}) "
            f"not in provenance schema"
        )


@dataclass(frozen=True)
class RunProvenanceRelation:
    """One concrete run-level provenance edge (D_run)."""

    producer_occurrence_id: str
    artifact_ref: ArtifactRef
    relation: str
    consumer_occurrence_id: str
    artifact_type: str = ""


def make_occurrence_id(
    *,
    run_id: str,
    branch_id: str,
    node_id: str,
    occurrence_index: int,
) -> str:
    """Deterministic occurrence identity within one run and branch."""
    return f"{run_id}:{branch_id}:{node_id}#{occurrence_index}"
