"""Deterministic backward provenance slicing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Set, Tuple, Any

from ..artifacts.models import ArtifactRef, ProducedBy, DerivedFrom
from ..artifacts.registry import ArtifactRegistry
from ..core.outcomes import CriterionValue
from ..provenance.models import (
    ArtifactBinding,
    OccurrenceRecord,
    RunProvenanceRelation,
)
from ..replay.events import InterventionEvent
from .models import ExplainedCase


@dataclass
class CaseSlice:
    """Deterministic provenance slice (provenance relevance only)."""

    explained_case: ExplainedCase

    relevant_artifact_refs: Tuple[ArtifactRef, ...] = ()
    relevant_occurrences: Tuple[OccurrenceRecord, ...] = ()
    relevant_relations: Tuple[RunProvenanceRelation, ...] = ()
    relevant_derivations: Tuple[DerivedFrom, ...] = ()

    # Terminal MCQ context
    criterion_values: Tuple[CriterionValue, ...] = ()
    decision: str = ""
    decision_reasons: Tuple[str, ...] = ()
    required_fixes: Tuple[str, ...] = ()
    assessment_ref: Optional[ArtifactRef] = None

    # Execution projection (super-step grouping)
    execution_projection: Tuple["SliceSuperstep", ...] = ()
    relevant_node_ids: Tuple[str, ...] = ()

    source_boundaries: Tuple["SliceBoundary", ...] = ()
    warnings: Tuple[str, ...] = ()

    # Backward compatibility (deprecated; do not interpret causally)
    factual_decision: str = ""
    provenance_relevant: bool = True
    counterfactual_relevance: bool = False

    @property
    def relevant_criteria(self) -> List[str]:
        """Compatibility alias: list of criterion names."""
        return [cv.name for cv in self.criterion_values]


class SliceValidationError(ValueError):
    """Raised when strict slicing invariants are violated."""


@dataclass(frozen=True)
class SliceBoundary:
    artifact_ref: ArtifactRef
    artifact_type: str
    kind: str
    note: str = ""


@dataclass(frozen=True)
class SliceSuperstep:
    superstep: Optional[int]
    checkpoint_before: Optional[str]
    checkpoint_after: Optional[str]
    occurrence_ids: Tuple[str, ...]


class ProvenanceSliceAdapter(Protocol):
    """Case adapter: select which producer input bindings matter for one output."""

    def relevant_bindings_for_output(
        self,
        *,
        output_ref: ArtifactRef,
        producer_occurrence: OccurrenceRecord,
        candidate_bindings: Sequence[ArtifactBinding],
        registry: ArtifactRegistry,
        branch_id: str,
    ) -> Sequence[ArtifactBinding]: ...


def _artifact_type_from_registry(
    registry: Optional[ArtifactRegistry],
    ref: ArtifactRef,
    *,
    branch_id: str,
) -> str:
    if registry is None:
        return ""
    env = registry.get_by_ref(ref, branch_id=branch_id) or registry.get_by_ref(ref)
    return getattr(env, "artifact_type", "") or ""


def _resolve_producer_map(
    *,
    produced_by: Sequence[ProducedBy],
    occurrence_records: Sequence[OccurrenceRecord],
    strict: bool,
) -> Dict[ArtifactRef, List[str]]:
    if produced_by:
        by_artifact: Dict[ArtifactRef, List[str]] = {}
        for pb in produced_by:
            by_artifact.setdefault(pb.artifact_ref, []).append(
                pb.producer_occurrence_id
            )
        if strict:
            for ref, occ_ids in by_artifact.items():
                if len(set(occ_ids)) != 1:
                    raise SliceValidationError(
                        f"Ambiguous producer for {ref}: {sorted(set(occ_ids))!r}"
                    )
        return by_artifact

    # Compatibility fallback: reconstruct from occurrence.produced_artifacts
    by_artifact = {}
    for occ in occurrence_records:
        for ref in occ.produced_artifacts or ():
            by_artifact.setdefault(ref, []).append(occ.occurrence_id)
    if strict:
        for ref, occ_ids in by_artifact.items():
            if len(set(occ_ids)) != 1:
                raise SliceValidationError(
                    f"Ambiguous producer for {ref}: {sorted(set(occ_ids))!r}"
                )
    return by_artifact


class ProvenanceSlicer:
    """Deterministic backward provenance slicer (Prompt 12)."""

    def backward_slice(
        self,
        explained_case: ExplainedCase,
        occurrence_records: Sequence[OccurrenceRecord],
        provenance_relations: Sequence[RunProvenanceRelation],
        *,
        registry: Optional[ArtifactRegistry] = None,
        produced_by: Sequence[ProducedBy] = (),
        artifact_derivations: Sequence[DerivedFrom] = (),
        intervention_events: Sequence[InterventionEvent] = (),
        adapter: Optional[ProvenanceSliceAdapter] = None,
        stop_artifact_types: Optional[Set[str]] = None,
        require_resolved_boundaries: bool = False,
        include_unrelated: bool = False,
        strict: bool = False,
    ) -> CaseSlice:
        """BackwardSlice(ζ, τ, D_run^τ) + factual terminal context."""
        stop_types = stop_artifact_types or set()
        branch_id = explained_case.branch_id

        # Filter to the explained run/branch when provenance records carry it.
        occ_filtered = []
        for o in occurrence_records:
            if o.occurrence_id is None:
                continue
            if o.branch_id and o.branch_id != branch_id:
                continue
            if explained_case.run_id and o.run_id and o.run_id != explained_case.run_id:
                continue
            occ_filtered.append(o)

        occ_by_id: Dict[str, OccurrenceRecord] = {o.occurrence_id: o for o in occ_filtered}

        if produced_by:
            # Strict validation: ProducedBy must align with occurrence.produced_artifacts.
            if strict:
                for pb in produced_by:
                    occ = occ_by_id.get(pb.producer_occurrence_id)
                    if occ is None:
                        raise SliceValidationError(
                            f"ProducedBy producer occurrence missing: {pb.producer_occurrence_id}"
                        )
                    if pb.artifact_ref not in (occ.produced_artifacts or ()):
                        raise SliceValidationError(
                            f"ProducedBy inconsistency: {pb.artifact_ref} not in "
                            f"{pb.producer_occurrence_id}.produced_artifacts"
                        )

        # Strict validation: decision/assessment must match the explained candidate version.
        if strict and registry is not None:
            dec_env = registry.get_by_ref(
                explained_case.decision_ref, branch_id=branch_id
            ) or registry.get_by_ref(explained_case.decision_ref)
            if dec_env is not None:
                payload = dec_env.payload
                q_id = getattr(payload, "q_id", None)
                cand_v = getattr(payload, "candidate_version", None)
                if q_id is not None and q_id != explained_case.artifact_ref.logical_id:
                    raise SliceValidationError("decision payload q_id mismatch")
                if cand_v is not None and int(cand_v) != explained_case.artifact_ref.version:
                    raise SliceValidationError("decision payload candidate_version mismatch")
                persisted_ass = getattr(payload, "assessment_ref", None)
                if explained_case.assessment_ref is not None and persisted_ass != explained_case.assessment_ref:
                    raise SliceValidationError("decision payload assessment_ref mismatch")

            if explained_case.assessment_ref is not None:
                ass_env = registry.get_by_ref(
                    explained_case.assessment_ref, branch_id=branch_id
                ) or registry.get_by_ref(explained_case.assessment_ref)
                if ass_env is not None:
                    payload = ass_env.payload
                    target_ref = getattr(payload, "target_ref", None)
                    if target_ref is not None and target_ref != explained_case.artifact_ref:
                        raise SliceValidationError("assessment payload target_ref mismatch")

        produced_by_map = _resolve_producer_map(
            produced_by=produced_by,
            occurrence_records=occurrence_records,
            strict=strict,
        )

        # Match D_run by (consumer_occurrence_id, artifact_ref, relation).
        rels_by_key: Dict[
            Tuple[str, ArtifactRef, str], List[RunProvenanceRelation]
        ] = {}
        for rel in provenance_relations:
            # Keep only relations whose consumer occurrence is on the explained branch.
            cons = occ_by_id.get(rel.consumer_occurrence_id)
            prod = occ_by_id.get(rel.producer_occurrence_id)
            if cons is None or prod is None:
                continue
            if cons.branch_id and cons.branch_id != branch_id:
                continue
            if explained_case.run_id and cons.run_id and cons.run_id != explained_case.run_id:
                continue
            key = (rel.consumer_occurrence_id, rel.artifact_ref, rel.relation)
            rels_by_key.setdefault(key, []).append(rel)

        intervention_supplied: Set[ArtifactRef] = set()
        for ev in intervention_events or ():
            for ref in getattr(ev, "supplied_refs", ()) or ():
                intervention_supplied.add(ref)

        def default_adapter(
            *,
            output_ref: ArtifactRef,
            producer_occurrence: OccurrenceRecord,
            candidate_bindings: Sequence[ArtifactBinding],
            registry: ArtifactRegistry,
            branch_id: str,
        ) -> Sequence[ArtifactBinding]:
            return [b for b in candidate_bindings if b.artifact_ref == output_ref]

        if adapter is None:
            adapter = default_adapter  # type: ignore[assignment]

        # Seed from terminal decision (and companion assessment if present).
        frontier: List[ArtifactRef] = [explained_case.decision_ref]
        if explained_case.assessment_ref is not None:
            frontier.append(explained_case.assessment_ref)
        frontier.append(explained_case.artifact_ref)
        frontier = sorted(set(frontier), key=lambda r: (r.logical_id, r.version))

        visited: Set[ArtifactRef] = set()
        relevant_artifacts: Set[ArtifactRef] = set(frontier)
        relevant_occ_ids: Set[str] = set()
        relevant_rels_dedup: Set[Tuple[str, str, ArtifactRef, str]] = set()
        relevant_rels: List[RunProvenanceRelation] = []
        boundaries: List[SliceBoundary] = []

        while frontier:
            alpha = frontier.pop(0)
            if alpha in visited:
                continue
            visited.add(alpha)

            if alpha in intervention_supplied:
                boundaries.append(
                    SliceBoundary(
                        artifact_ref=alpha,
                        artifact_type=_artifact_type_from_registry(
                            registry, alpha, branch_id=branch_id
                        ),
                        kind="intervention_supplied",
                        note="Supplied by intervention; stop backward producer traversal.",
                    )
                )
                continue

            if registry is not None:
                a_type = _artifact_type_from_registry(
                    registry, alpha, branch_id=branch_id
                )
                if a_type in stop_types:
                    boundaries.append(
                        SliceBoundary(
                            artifact_ref=alpha,
                            artifact_type=a_type,
                            kind="configured_stop",
                            note=f"Stopped by stop_artifact_types={stop_types!r}.",
                        )
                    )
                    continue

            producer_occ_ids = produced_by_map.get(alpha) or []
            if not producer_occ_ids:
                boundaries.append(
                    SliceBoundary(
                        artifact_ref=alpha,
                        artifact_type=_artifact_type_from_registry(
                            registry, alpha, branch_id=branch_id
                        ),
                        kind="unresolved_producer",
                        note="No ProducedBy record for this artifact.",
                    )
                )
                if strict:
                    raise SliceValidationError(f"No producer for {alpha}")
                continue
            if strict and len(set(producer_occ_ids)) != 1:
                raise SliceValidationError(
                    f"Ambiguous producer for {alpha}: {sorted(set(producer_occ_ids))!r}"
                )

            omega_prod_id = sorted(set(producer_occ_ids))[0]
            omega_prod = occ_by_id.get(omega_prod_id)
            if omega_prod is None:
                boundaries.append(
                    SliceBoundary(
                        artifact_ref=alpha,
                        artifact_type=_artifact_type_from_registry(
                            registry, alpha, branch_id=branch_id
                        ),
                        kind="unresolved_producer",
                        note=f"Producer occurrence {omega_prod_id!r} missing from occurrence_records.",
                    )
                )
                if strict:
                    raise SliceValidationError(
                        f"Missing producer occurrence {omega_prod_id!r}"
                    )
                continue

            relevant_occ_ids.add(omega_prod_id)

            if hasattr(adapter, "relevant_bindings_for_output"):
                relevant_bindings = adapter.relevant_bindings_for_output(  # type: ignore[union-attr]
                    output_ref=alpha,
                    producer_occurrence=omega_prod,
                    candidate_bindings=omega_prod.input_bindings,
                    registry=registry,  # type: ignore[arg-type]
                    branch_id=branch_id,
                )
            else:
                # Backward-compat: allow a plain function adapter.
                relevant_bindings = adapter(  # type: ignore[operator]
                    output_ref=alpha,
                    producer_occurrence=omega_prod,
                    candidate_bindings=omega_prod.input_bindings,
                    registry=registry,  # type: ignore[arg-type]
                    branch_id=branch_id,
                )

            for binding in relevant_bindings:
                beta = binding.artifact_ref
                matching = rels_by_key.get((omega_prod_id, beta, binding.relation)) or []

                if not matching:
                    boundaries.append(
                        SliceBoundary(
                            artifact_ref=beta,
                            artifact_type=_artifact_type_from_registry(
                                registry, beta, branch_id=branch_id
                            ),
                            kind="attribution_boundary",
                            note=(
                                "Missing D_run edge "
                                f"(consumer={omega_prod_id}, artifact={beta}, relation={binding.relation})."
                            ),
                        )
                    )
                    if strict or require_resolved_boundaries:
                        raise SliceValidationError(
                            f"Missing D_run for consumer={omega_prod_id} artifact={beta} relation={binding.relation}"
                        )
                    continue

                for rel in matching:
                    rel_key = (
                        rel.producer_occurrence_id,
                        rel.consumer_occurrence_id,
                        rel.artifact_ref,
                        rel.relation,
                    )
                    if rel_key in relevant_rels_dedup:
                        continue
                    relevant_rels_dedup.add(rel_key)
                    relevant_rels.append(rel)
                    relevant_occ_ids.add(rel.producer_occurrence_id)
                    relevant_occ_ids.add(rel.consumer_occurrence_id)
                    relevant_artifacts.add(beta)
                    if beta not in visited:
                        frontier.append(beta)

        relevant_occurrences: List[OccurrenceRecord] = [
            occ_by_id[oid] for oid in relevant_occ_ids if oid in occ_by_id
        ]
        relevant_occurrences.sort(key=lambda o: (o.execution_index, o.occurrence_id))

        if not include_unrelated:
            relevant_refs = set(relevant_artifacts)
            filtered: List[OccurrenceRecord] = []
            for occ in relevant_occurrences:
                touches = set(occ.produced_artifacts) | {
                    b.artifact_ref for b in (occ.input_bindings or ())
                }
                if touches & relevant_refs:
                    filtered.append(occ)
            relevant_occurrences = filtered

        # Deterministic artifact ordering: earliest producer execution index.
        occ_exec_by_artifact: Dict[ArtifactRef, int] = {}
        for occ in relevant_occurrences:
            for ref in occ.produced_artifacts or ():
                occ_exec_by_artifact[ref] = min(
                    occ_exec_by_artifact.get(ref, 10**12), occ.execution_index
                )

        def art_sort_key(r: ArtifactRef) -> Tuple[int, str, int]:
            return (
                occ_exec_by_artifact.get(r, 10**12),
                r.logical_id,
                r.version,
            )

        relevant_artifact_refs = sorted(relevant_artifacts, key=art_sort_key)

        relevant_artifact_set = set(relevant_artifact_refs)
        relevant_derivations = [
            d
            for d in (artifact_derivations or ())
            if d.new_artifact_ref in relevant_artifact_set
        ]
        relevant_derivations.sort(
            key=lambda d: (
                art_sort_key(d.new_artifact_ref),
                art_sort_key(d.previous_artifact_ref),
            )
        )

        # Terminal extraction (criteria + decision details) from persisted artifacts.
        criterion_values: Tuple[CriterionValue, ...] = ()
        decision_str: str = explained_case.outcome
        decision_reasons: Tuple[str, ...] = ()
        required_fixes: Tuple[str, ...] = ()
        assessment_ref: Optional[ArtifactRef] = explained_case.assessment_ref

        if registry is not None:
            dec_env = registry.get_by_ref(
                explained_case.decision_ref, branch_id=branch_id
            ) or registry.get_by_ref(explained_case.decision_ref)
            if dec_env is not None:
                payload = dec_env.payload
                decision_str = str(getattr(payload, "decision", decision_str) or decision_str).lower().strip()
                mapping = {
                    "accepted": "accept",
                    "accept": "accept",
                    "needs_revision": "revise",
                    "revise": "revise",
                    "rejected": "reject",
                    "reject": "reject",
                }
                decision_str = mapping.get(decision_str, decision_str)
                decision_reasons = tuple(getattr(payload, "reasons", ()) or ())
                required_fixes = tuple(getattr(payload, "required_fixes", ()) or ())
                persisted_ass = getattr(payload, "assessment_ref", None)
                if persisted_ass is not None:
                    assessment_ref = persisted_ass

            if assessment_ref is not None:
                ass_env = registry.get_by_ref(assessment_ref, branch_id=branch_id) or registry.get_by_ref(
                    assessment_ref
                )
                if ass_env is not None:
                    criterion_values = tuple(
                        getattr(ass_env.payload, "criteria", ()) or ()
                    )

        # Super-step projection.
        superstep_groups: Dict[
            Tuple[Optional[int], Optional[str], Optional[str]], List[str]
        ] = {}
        for occ in relevant_occurrences:
            key = (occ.superstep, occ.checkpoint_before, occ.checkpoint_after)
            superstep_groups.setdefault(key, []).append(occ.occurrence_id)

        projection: List[SliceSuperstep] = []

        def group_sort_key(
            k: Tuple[Optional[int], Optional[str], Optional[str]],
        ) -> Tuple[int, str]:
            ss = k[0]
            return (ss if ss is not None else 10**12, f"{k[1]}|{k[2]}")

        for k in sorted(superstep_groups.keys(), key=group_sort_key):
            ss, cb, ca = k
            projection.append(
                SliceSuperstep(
                    superstep=ss,
                    checkpoint_before=cb,
                    checkpoint_after=ca,
                    occurrence_ids=tuple(superstep_groups[k]),
                )
            )

        # Deterministic relation ordering (consumer then producer then relation then artifact).
        relevant_rels.sort(
            key=lambda r: (
                occ_by_id.get(r.consumer_occurrence_id).execution_index
                if occ_by_id.get(r.consumer_occurrence_id) is not None
                else 10**12,
                occ_by_id.get(r.producer_occurrence_id).execution_index
                if occ_by_id.get(r.producer_occurrence_id) is not None
                else 10**12,
                r.relation,
                r.artifact_ref.logical_id,
                r.artifact_ref.version,
            )
        )

        return CaseSlice(
            explained_case=explained_case,
            relevant_artifact_refs=relevant_artifact_refs,
            relevant_occurrences=tuple(relevant_occurrences),
            relevant_relations=tuple(relevant_rels),
            relevant_derivations=tuple(relevant_derivations),
            criterion_values=criterion_values,
            decision=decision_str,
            decision_reasons=decision_reasons,
            required_fixes=required_fixes,
            assessment_ref=assessment_ref,
            execution_projection=tuple(projection),
            relevant_node_ids=tuple(sorted({o.node_id for o in relevant_occurrences})),
            source_boundaries=tuple(boundaries),
            warnings=(),
            factual_decision=decision_str,
            provenance_relevant=True,
            counterfactual_relevance=False,
        )

    def slice_run_result(
        self,
        explained_case: ExplainedCase,
        run_result: Any,
        *,
        registry: Optional[ArtifactRegistry] = None,
        adapter: Optional[ProvenanceSliceAdapter] = None,
        strict: bool = False,
    ) -> CaseSlice:
        """Convenience slicing entrypoint from a `WorkflowRunResult`-like object."""
        state = getattr(run_result, "final_state", None) or {}
        occs = state.get("occurrence_records") or getattr(run_result, "occurrences", []) or []
        rels = state.get("run_provenance_relations") or []
        produced_by = state.get("produced_by") or []
        derivations = state.get("artifact_derivations") or []
        intervention_events = state.get("intervention_events") or []

        if registry is None:
            try:
                from ..artifacts.channel_state import load_artifact_channels

                registry, _resolver = load_artifact_channels(state)
            except Exception:
                registry = None

        return self.backward_slice(
            explained_case,
            occs,
            rels,
            registry=registry,
            produced_by=produced_by,
            artifact_derivations=derivations,
            intervention_events=intervention_events,
            adapter=adapter,
            strict=strict,
        )
