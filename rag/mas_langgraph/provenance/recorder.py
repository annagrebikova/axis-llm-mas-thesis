"""Record occurrence-level provenance during workflow execution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..artifacts.models import ArtifactRef, ProducedBy
from .models import (
    ArtifactBinding,
    OccurrenceRecord,
    ProvenanceSchema,
    RunProvenanceRelation,
    make_occurrence_id,
)


@dataclass
class ProvenanceRecorder:
    """Instruments node occurrences and builds run-level provenance relations.

    Authoritative data lives in checkpointed workflow channels; this object is a
    helper view reconstructed via :meth:`from_state` / :meth:`to_state`.
    """

    schema: ProvenanceSchema = field(default_factory=ProvenanceSchema)
    branch_id: str = "factual"
    run_id: str = ""
    occurrences: List[OccurrenceRecord] = field(default_factory=list)
    produced_by: List[ProducedBy] = field(default_factory=list)
    run_provenance_relations: List[RunProvenanceRelation] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    unresolved_producers: List[str] = field(default_factory=list)
    _node_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _execution_index: int = 0
    # ArtifactRef -> artifact_type from envelopes / producer registration
    _artifact_types: Dict[ArtifactRef, str] = field(default_factory=dict)
    # Refs supplied by InterventionEvent (not workflow-produced)
    intervention_supplied_refs: Set[ArtifactRef] = field(default_factory=set)
    intervention_supplied_bindings: List[str] = field(default_factory=list)

    @classmethod
    def from_state(
        cls,
        *,
        occurrence_records: Optional[Sequence[OccurrenceRecord]] = None,
        produced_by: Optional[Sequence[ProducedBy]] = None,
        run_provenance_relations: Optional[Sequence[RunProvenanceRelation]] = None,
        violations: Optional[Sequence[str]] = None,
        unresolved_producers: Optional[Sequence[str]] = None,
        node_counts: Optional[Dict[str, int]] = None,
        execution_index: int = 0,
        artifact_types: Optional[Dict[ArtifactRef, str]] = None,
        run_id: str = "",
        branch_id: str = "factual",
        schema: Optional[ProvenanceSchema] = None,
    ) -> "ProvenanceRecorder":
        """Rebuild a recorder from checkpointed provenance channels."""
        counts: Dict[str, int] = defaultdict(int)
        if node_counts:
            for k, v in node_counts.items():
                counts[str(k)] = int(v)
        else:
            for occ in occurrence_records or ():
                counts[occ.node_id] = max(counts[occ.node_id], occ.occurrence_index)
        rec = cls(
            schema=schema or ProvenanceSchema(),
            branch_id=branch_id,
            run_id=run_id,
            occurrences=list(occurrence_records or []),
            produced_by=list(produced_by or []),
            run_provenance_relations=list(run_provenance_relations or []),
            violations=list(violations or []),
            unresolved_producers=list(unresolved_producers or []),
            _node_counts=counts,
            _execution_index=int(execution_index),
            _artifact_types=dict(artifact_types or {}),
        )
        return rec

    def to_state(self) -> Dict[str, Any]:
        """Serialize into checkpointed provenance channel fields."""
        return {
            "occurrence_records": list(self.occurrences),
            "produced_by": list(self.produced_by),
            "run_provenance_relations": list(self.run_provenance_relations),
            "provenance_violations": list(self.violations),
            "provenance_unresolved_producers": list(self.unresolved_producers),
            "provenance_node_counts": dict(self._node_counts),
            "provenance_execution_index": int(self._execution_index),
        }

    def register_artifact_type(self, ref: ArtifactRef, artifact_type: str) -> None:
        if artifact_type:
            self._artifact_types[ref] = artifact_type

    def begin_occurrence(
        self,
        node_id: str,
        *,
        input_bindings: Sequence[ArtifactBinding] = (),
        checkpoint_before: Optional[str] = None,
        superstep: Optional[int] = None,
    ) -> OccurrenceRecord:
        """Allocate the next occurrence index for ``node_id`` (incomplete record)."""
        self._node_counts[node_id] += 1
        idx = self._node_counts[node_id]
        self._execution_index += 1
        occ_id = make_occurrence_id(
            run_id=self.run_id or "run",
            branch_id=self.branch_id,
            node_id=node_id,
            occurrence_index=idx,
        )
        return OccurrenceRecord(
            occurrence_id=occ_id,
            node_id=node_id,
            occurrence_index=idx,
            branch_id=self.branch_id,
            run_id=self.run_id,
            execution_index=self._execution_index,
            checkpoint_before=checkpoint_before,
            superstep=superstep,
            input_bindings=tuple(input_bindings),
        )

    def complete_occurrence(
        self,
        partial: OccurrenceRecord,
        *,
        produced_artifacts: Sequence[ArtifactRef] = (),
        state_delta: Optional[Dict[str, Any]] = None,
        checkpoint_after: Optional[str] = None,
        error: Optional[str] = None,
        producer_artifact_types: Optional[Dict[ArtifactRef, str]] = None,
    ) -> OccurrenceRecord:
        """Finalize an occurrence and record ProducedBy for new artifacts.

        Schema validation happens when reconstructing ``D_run`` from consumer
        bindings — not via synthetic ``produced`` relations.
        """
        record = OccurrenceRecord(
            occurrence_id=partial.occurrence_id,
            node_id=partial.node_id,
            occurrence_index=partial.occurrence_index,
            branch_id=partial.branch_id,
            run_id=partial.run_id or self.run_id,
            execution_index=partial.execution_index,
            checkpoint_before=partial.checkpoint_before,
            checkpoint_after=checkpoint_after,
            superstep=partial.superstep,
            input_bindings=partial.input_bindings,
            produced_artifacts=tuple(produced_artifacts),
            state_delta=dict(state_delta or {}),
            error=error,
        )
        self.occurrences.append(record)
        type_map = producer_artifact_types or {}
        for ref in produced_artifacts:
            atype = type_map.get(ref) or self._artifact_types.get(ref, "")
            if atype:
                self._artifact_types[ref] = atype
            # Idempotent ProducedBy for the same (ref, occurrence)
            already = any(
                pb.artifact_ref == ref and pb.producer_occurrence_id == record.occurrence_id
                for pb in self.produced_by
            )
            if not already:
                self.produced_by.append(
                    ProducedBy(artifact_ref=ref, producer_occurrence_id=record.occurrence_id)
                )
        # Rebuild D_run incrementally for this consumer
        self._append_run_relations_for_occurrence(record)
        return record

    def _append_run_relations_for_occurrence(self, occ: OccurrenceRecord) -> None:
        producer_of: Dict[ArtifactRef, str] = {
            pb.artifact_ref: pb.producer_occurrence_id for pb in self.produced_by
        }
        producer_node_of: Dict[str, str] = {
            o.occurrence_id: o.node_id for o in self.occurrences
        }
        for binding in occ.input_bindings:
            prod_occ = producer_of.get(binding.artifact_ref)
            if prod_occ is None:
                if binding.artifact_ref in self.intervention_supplied_refs:
                    note = (
                        f"intervention_supplied: artifact={binding.artifact_ref} "
                        f"consumer={occ.occurrence_id} relation={binding.relation}"
                    )
                    if note not in self.intervention_supplied_bindings:
                        self.intervention_supplied_bindings.append(note)
                    continue
                msg = (
                    f"unresolved_producer: artifact={binding.artifact_ref} "
                    f"consumer={occ.occurrence_id} relation={binding.relation}"
                )
                if msg not in self.unresolved_producers:
                    self.unresolved_producers.append(msg)
                continue
            atype = self._artifact_types.get(binding.artifact_ref, "")
            rel = RunProvenanceRelation(
                producer_occurrence_id=prod_occ,
                artifact_ref=binding.artifact_ref,
                relation=binding.relation,
                consumer_occurrence_id=occ.occurrence_id,
                artifact_type=atype,
            )
            if any(
                r.producer_occurrence_id == rel.producer_occurrence_id
                and r.artifact_ref == rel.artifact_ref
                and r.relation == rel.relation
                and r.consumer_occurrence_id == rel.consumer_occurrence_id
                for r in self.run_provenance_relations
            ):
                continue
            self.run_provenance_relations.append(rel)
            producer_node = producer_node_of.get(prod_occ, "unknown")
            msg = self.schema.validate_runtime(
                producer_node,
                atype or "unknown",
                binding.relation,
                occ.node_id,
            )
            if msg and msg not in self.violations:
                self.violations.append(msg)

    def build_run_relations(
        self,
        *,
        artifact_type_lookup: Optional[Dict[ArtifactRef, str]] = None,
    ) -> List[RunProvenanceRelation]:
        """Deterministically reconstruct D_run from ProducedBy + consumer bindings."""
        if artifact_type_lookup:
            self._artifact_types.update(artifact_type_lookup)
        self.run_provenance_relations = []
        self.violations = []
        self.unresolved_producers = []
        for occ in self.occurrences:
            self._append_run_relations_for_occurrence(occ)
        return list(self.run_provenance_relations)

    def occurrence_by_label(self, labeled_id: str) -> Optional[OccurrenceRecord]:
        """Look up ``node_id#n``."""
        for occ in self.occurrences:
            if occ.labeled_id == labeled_id:
                return occ
        return None

    def producer_of(self, ref: ArtifactRef) -> Optional[str]:
        for pb in self.produced_by:
            if pb.artifact_ref == ref:
                return pb.producer_occurrence_id
        return None
