"""MCQ-specific deterministic provenance slicing adapters (Prompt 12)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ...artifacts.models import ArtifactRef
from ...artifacts.registry import ArtifactRegistry
from ...provenance.models import ArtifactBinding, OccurrenceRecord
from ...axis.models import ExplainedCase
from ...axis.slicing import ProvenanceSliceAdapter
from ...provenance.models import RunProvenanceRelation
from ...core.entities import KnowledgeAtom
from ...domains.quiz.entities import MCQBlueprint, CandidateMCQ
from ...core.entities import EvidenceChunk
from ...cases.mcq.artifacts import candidate_artifact_version


def _get_env(
    registry: ArtifactRegistry,
    ref: ArtifactRef,
    *,
    branch_id: str,
):
    return registry.get_by_ref(ref, branch_id=branch_id) or registry.get_by_ref(ref)


def resolve_mcq_explained_case(
    *,
    run_id: str,
    branch_id: str,
    artifact_ref: ArtifactRef,
    registry: ArtifactRegistry,
) -> ExplainedCase:
    """Resolve decision + assessment refs for one exact CandidateMCQ version."""
    q_id = artifact_ref.logical_id
    decision_ref = ArtifactRef(logical_id=f"decision:{q_id}", version=artifact_ref.version)
    assessment_ref = ArtifactRef(
        logical_id=f"assessment:{q_id}", version=artifact_ref.version
    )

    decision_env = _get_env(registry, decision_ref, branch_id=branch_id)
    assessment_env = _get_env(registry, assessment_ref, branch_id=branch_id)
    if decision_env is None:
        raise ValueError(f"Missing decision envelope: {decision_ref}")
    if assessment_env is None:
        raise ValueError(f"Missing assessment envelope: {assessment_ref}")

    decision_payload = decision_env.payload
    assessment_payload = assessment_env.payload

    # Validate: exact candidate version & cross-ref identity
    if getattr(decision_payload, "q_id", None) != q_id:
        raise ValueError("decision.q_id mismatch")
    if int(getattr(decision_payload, "candidate_version", -1) or -1) != artifact_ref.version:
        raise ValueError("decision.candidate_version mismatch")
    if getattr(decision_payload, "assessment_ref", None) != assessment_ref:
        raise ValueError("decision.assessment_ref mismatch")

    if getattr(assessment_payload, "assessment_ref", None) != assessment_ref:
        raise ValueError("assessment.assessment_ref mismatch")
    if getattr(assessment_payload, "target_ref", None) != artifact_ref:
        raise ValueError("assessment.target_ref mismatch")

    outcome = str(getattr(decision_payload, "decision", "") or "")
    cand_env = _get_env(registry, artifact_ref, branch_id=branch_id)
    correspondence_key = None
    if cand_env is not None:
        correspondence_key = getattr(getattr(cand_env, "payload", None), "blueprint_id", None)
        if correspondence_key is not None:
            correspondence_key = str(correspondence_key)
    return ExplainedCase(
        run_id=run_id,
        artifact_ref=artifact_ref,
        branch_id=branch_id,
        decision_ref=decision_ref,
        outcome=outcome,
        assessment_ref=assessment_ref,
        correspondence_key=correspondence_key,
    )


@dataclass(frozen=True)
class MCQProvenanceSliceAdapter(ProvenanceSliceAdapter):
    """Filter batch-node input bindings so one output candidate stays scoped."""

    def relevant_bindings_for_output(
        self,
        *,
        output_ref: ArtifactRef,
        producer_occurrence: OccurrenceRecord,
        candidate_bindings: Sequence[ArtifactBinding],
        registry: ArtifactRegistry,
        branch_id: str,
    ) -> Sequence[ArtifactBinding]:
        node_id = producer_occurrence.node_id

        # apply_acceptance_gate produces SupervisorDecision + OutcomeAssessment
        if node_id == "apply_acceptance_gate":
            if output_ref.logical_id.startswith("decision:"):
                q_id = output_ref.logical_id.split("decision:", 1)[1]
            elif output_ref.logical_id.startswith("assessment:"):
                q_id = output_ref.logical_id.split("assessment:", 1)[1]
            else:
                q_id = output_ref.logical_id
            candidate_ref = ArtifactRef(q_id, output_ref.version)

            cand_env = _get_env(registry, candidate_ref, branch_id=branch_id)
            if cand_env is None:
                return ()
            blueprint_id = getattr(cand_env.payload, "blueprint_id", None)
            if not blueprint_id:
                return ()
            blueprint_ref = ArtifactRef(str(blueprint_id), 1)

            result = []
            for b in candidate_bindings:
                if b.relation == "candidate_question" and b.artifact_ref == candidate_ref:
                    result.append(b)
                elif (
                    b.relation == "evidence_report"
                    and b.artifact_ref.logical_id == f"evidence-report:{q_id}"
                    and b.artifact_ref.version == output_ref.version
                ):
                    result.append(b)
                elif (
                    b.relation == "cross_cove_report"
                    and b.artifact_ref.logical_id == f"cross-cove-report:{q_id}"
                    and b.artifact_ref.version == output_ref.version
                ):
                    result.append(b)
                elif (
                    b.relation == "blueprint_constraints"
                    and b.artifact_ref.logical_id == str(blueprint_id)
                    and b.artifact_ref.version == 1
                ):
                    result.append(b)
            return tuple(result)

        # revise_candidate_mcqs produces a revised CandidateMCQ version
        if node_id == "revise_candidate_mcqs":
            q_id = output_ref.logical_id
            new_v = output_ref.version
            old_v = new_v - 1
            old_candidate_ref = ArtifactRef(q_id, old_v)

            old_env = _get_env(registry, old_candidate_ref, branch_id=branch_id)
            if old_env is None:
                return ()
            blueprint_id = getattr(old_env.payload, "blueprint_id", None)
            if not blueprint_id:
                return ()
            blueprint_ref = ArtifactRef(str(blueprint_id), 1)
            bp_env = _get_env(registry, blueprint_ref, branch_id=branch_id)
            if bp_env is None:
                return ()
            bp_payload = bp_env.payload

            cited_atom_ids = tuple(getattr(bp_payload, "target_atom_ids", ()) or ())
            evidence_chunk_ids = tuple(getattr(bp_payload, "evidence_chunk_ids", ()) or ())

            result = []
            for b in candidate_bindings:
                if b.relation == "candidate_to_revise" and b.artifact_ref == old_candidate_ref:
                    result.append(b)
                elif (
                    b.relation == "revision_blueprint"
                    and b.artifact_ref == blueprint_ref
                ):
                    result.append(b)
                elif (
                    b.relation == "knowledge_atom"
                    and b.artifact_ref.logical_id in cited_atom_ids
                    and b.artifact_ref.version == 1
                ):
                    result.append(b)
                elif (
                    b.relation == "supporting_evidence"
                    and b.artifact_ref.logical_id in evidence_chunk_ids
                    and b.artifact_ref.version == 1
                ):
                    result.append(b)
                elif (
                    b.relation == "evidence_feedback"
                    and b.artifact_ref.logical_id == f"evidence-report:{q_id}"
                    and b.artifact_ref.version == old_v
                ):
                    result.append(b)
                elif (
                    b.relation == "cross_cove_feedback"
                    and b.artifact_ref.logical_id == f"cross-cove-report:{q_id}"
                    and b.artifact_ref.version == old_v
                ):
                    result.append(b)
                elif (
                    b.relation == "revision_decision"
                    and b.artifact_ref.logical_id == f"decision:{q_id}"
                    and b.artifact_ref.version == old_v
                ):
                    result.append(b)
            return tuple(result)

        # generate_candidate_mcqs_from_blueprints produces a CandidateMCQ version
        if node_id == "generate_candidate_mcqs_from_blueprints":
            q_id = output_ref.logical_id
            candidate_ref = ArtifactRef(q_id, output_ref.version)
            cand_env = _get_env(registry, candidate_ref, branch_id=branch_id)
            if cand_env is None:
                return ()
            blueprint_id = getattr(cand_env.payload, "blueprint_id", None)
            if not blueprint_id:
                return ()
            blueprint_ref = ArtifactRef(str(blueprint_id), 1)
            bp_env = _get_env(registry, blueprint_ref, branch_id=branch_id)
            if bp_env is None:
                return ()
            bp_payload = bp_env.payload
            atom_ids = tuple(getattr(bp_payload, "target_atom_ids", ()) or ())
            evidence_chunk_ids = tuple(getattr(bp_payload, "evidence_chunk_ids", ()) or ())

            result = []
            for b in candidate_bindings:
                if b.relation == "generation_blueprint" and b.artifact_ref == blueprint_ref:
                    result.append(b)
                elif (
                    b.relation == "knowledge_atom"
                    and b.artifact_ref.logical_id in atom_ids
                    and b.artifact_ref.version == 1
                ):
                    result.append(b)
                elif (
                    b.relation == "supporting_evidence"
                    and b.artifact_ref.logical_id in evidence_chunk_ids
                    and b.artifact_ref.version == 1
                ):
                    result.append(b)
            return tuple(result)

        # verify_candidate_mcqs_against_evidence produces EvidenceCheckReport version v
        if node_id == "verify_candidate_mcqs_against_evidence":
            if not output_ref.logical_id.startswith("evidence-report:"):
                return ()
            q_id = output_ref.logical_id.split("evidence-report:", 1)[1]
            v = output_ref.version
            candidate_ref = ArtifactRef(q_id, v)
            cand_env = _get_env(registry, candidate_ref, branch_id=branch_id)
            if cand_env is None:
                return ()
            blueprint_id = getattr(cand_env.payload, "blueprint_id", None)
            if not blueprint_id:
                return ()
            blueprint_ref = ArtifactRef(str(blueprint_id), 1)
            bp_env = _get_env(registry, blueprint_ref, branch_id=branch_id)
            if bp_env is None:
                return ()
            bp_payload = bp_env.payload
            atom_ids = tuple(getattr(bp_payload, "target_atom_ids", ()) or ())
            evidence_chunk_ids = tuple(getattr(bp_payload, "evidence_chunk_ids", ()) or ())

            result = []
            for b in candidate_bindings:
                if b.relation == "candidate_question" and b.artifact_ref == candidate_ref:
                    result.append(b)
                elif (
                    b.relation == "verification_blueprint"
                    and b.artifact_ref == blueprint_ref
                ):
                    result.append(b)
                elif (
                    b.relation == "grounding_atom"
                    and b.artifact_ref.logical_id in atom_ids
                    and b.artifact_ref.version == 1
                ):
                    result.append(b)
                elif (
                    b.relation == "supporting_evidence"
                    and b.artifact_ref.logical_id in evidence_chunk_ids
                    and b.artifact_ref.version == 1
                ):
                    result.append(b)
            return tuple(result)

        # cross_cove_verify_candidate_mcqs produces CrossCoVeReport version v
        if node_id == "cross_cove_verify_candidate_mcqs":
            if not output_ref.logical_id.startswith("cross-cove-report:"):
                return ()
            q_id = output_ref.logical_id.split("cross-cove-report:", 1)[1]
            v = output_ref.version
            candidate_ref = ArtifactRef(q_id, v)
            evidence_gate_ref = ArtifactRef(f"evidence-report:{q_id}", v)

            result = []
            for b in candidate_bindings:
                if b.relation == "candidate_question" and b.artifact_ref == candidate_ref:
                    result.append(b)
                elif (
                    b.relation == "evidence_verification_gate"
                    and b.artifact_ref == evidence_gate_ref
                ):
                    result.append(b)
            return tuple(result)

        # plan_mcq_blueprints produces a single MCQBlueprint (version 1)
        if node_id == "plan_mcq_blueprints":
            # output_ref.logical_id is the blueprint_id
            q_id = output_ref.logical_id
            blueprint_ref = ArtifactRef(q_id, output_ref.version)
            bp_env = _get_env(registry, blueprint_ref, branch_id=branch_id)
            if bp_env is None:
                return ()
            bp_payload = bp_env.payload
            section_id = getattr(bp_payload, "section_id", None)
            lo_id = getattr(bp_payload, "lo_id", None)
            atom_ids = tuple(getattr(bp_payload, "target_atom_ids", ()) or ())
            evidence_chunk_ids = tuple(getattr(bp_payload, "evidence_chunk_ids", ()) or ())

            result = []
            for b in candidate_bindings:
                if b.relation == "document_section" and b.artifact_ref.logical_id == str(section_id):
                    result.append(b)
                elif b.relation == "learning_objective" and b.artifact_ref.logical_id == str(lo_id):
                    result.append(b)
                elif b.relation == "planning_atom" and b.artifact_ref.logical_id in atom_ids:
                    result.append(b)
                elif b.relation == "planning_evidence" and b.artifact_ref.logical_id in evidence_chunk_ids:
                    result.append(b)
            return tuple(result)

        # atomize_evidence / atomize_new_evidence produce KnowledgeAtom(s)
        if node_id in {"atomize_evidence", "atomize_new_evidence"}:
            if output_ref.logical_id is None:
                return ()
            atom_env = _get_env(registry, output_ref, branch_id=branch_id)
            if atom_env is None:
                return ()
            payload = atom_env.payload
            citations = getattr(payload, "citations", ()) or ()
            chunk_ids = {
                c.get("chunk_id")
                for c in citations
                if isinstance(c, dict) and c.get("chunk_id")
            }
            result = []
            for b in candidate_bindings:
                if (
                    b.relation == "evidence_to_atomize"
                    and b.artifact_ref.logical_id in chunk_ids
                    and b.artifact_ref.version == 1
                ):
                    result.append(b)
            return tuple(result)

        # Default: only direct artifact bindings for this output.
        return tuple(b for b in candidate_bindings if b.artifact_ref == output_ref)

