"""Wrap existing MCQ dataclasses as ArtifactEnvelope payloads."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

from ...artifacts.models import ArtifactEnvelope, ArtifactRef, DerivedFrom
from ...artifacts.registry import ArtifactRegistry
from ...artifacts.utils import payloads_equivalent


def candidate_artifact_version(candidate: Any) -> int:
    """Map domain revision_count to 1-based artifact version."""
    return int(getattr(candidate, "revision_count", 0)) + 1


def report_artifact_version(state: Any, q_id: str) -> int:
    """Temporary report version aligned with candidate revision_count + 1."""
    candidate = getattr(state, "candidate_mcqs", {}).get(q_id)
    if candidate is not None:
        return candidate_artifact_version(candidate)
    return 1


def wrap_payload(
    payload: Any,
    *,
    artifact_type: str,
    logical_id: str,
    branch_id: str = "factual",
    created_in_run: str = "",
    version: int = 1,
    registry: Optional[ArtifactRegistry] = None,
) -> ArtifactEnvelope:
    """Wrap a payload into an immutable envelope and optionally store it."""
    env = ArtifactEnvelope.create(
        logical_id=logical_id,
        artifact_type=artifact_type,
        payload=payload,
        branch_id=branch_id,
        created_in_run=created_in_run,
        version=version,
    )
    if registry is not None:
        return registry.store(env)
    return env


def wrap_doc_section(
    section: Any,
    *,
    branch_id: str = "factual",
    created_in_run: str = "",
    registry: Optional[ArtifactRegistry] = None,
) -> ArtifactEnvelope:
    sid = getattr(section, "section_id", None) or getattr(section, "id", "section")
    return wrap_payload(
        section,
        artifact_type="DocSection",
        logical_id=str(sid),
        branch_id=branch_id,
        created_in_run=created_in_run,
        registry=registry,
    )


def wrap_learning_objective(
    lo: Any,
    *,
    branch_id: str = "factual",
    created_in_run: str = "",
    registry: Optional[ArtifactRegistry] = None,
) -> ArtifactEnvelope:
    lid = getattr(lo, "lo_id", None) or getattr(lo, "id", "lo")
    return wrap_payload(
        lo,
        artifact_type="LearningObjective",
        logical_id=str(lid),
        branch_id=branch_id,
        created_in_run=created_in_run,
        registry=registry,
    )


def wrap_evidence_chunk(
    chunk: Any,
    *,
    branch_id: str = "factual",
    created_in_run: str = "",
    registry: Optional[ArtifactRegistry] = None,
) -> ArtifactEnvelope:
    cid = getattr(chunk, "chunk_id", None) or getattr(chunk, "id", "chunk")
    return wrap_payload(
        chunk,
        artifact_type="EvidenceChunk",
        logical_id=str(cid),
        branch_id=branch_id,
        created_in_run=created_in_run,
        registry=registry,
    )


def wrap_knowledge_atom(
    atom: Any,
    *,
    branch_id: str = "factual",
    created_in_run: str = "",
    registry: Optional[ArtifactRegistry] = None,
) -> ArtifactEnvelope:
    aid = getattr(atom, "atom_id", None) or getattr(atom, "id", "atom")
    return wrap_payload(
        atom,
        artifact_type="KnowledgeAtom",
        logical_id=str(aid),
        branch_id=branch_id,
        created_in_run=created_in_run,
        registry=registry,
    )


def wrap_blueprint(
    blueprint: Any,
    *,
    branch_id: str = "factual",
    created_in_run: str = "",
    registry: Optional[ArtifactRegistry] = None,
) -> ArtifactEnvelope:
    bid = getattr(blueprint, "blueprint_id", "bp")
    return wrap_payload(
        blueprint,
        artifact_type="MCQBlueprint",
        logical_id=str(bid),
        branch_id=branch_id,
        created_in_run=created_in_run,
        registry=registry,
    )


def wrap_candidate_mcq(
    candidate: Any,
    *,
    version: Optional[int] = None,
    branch_id: str = "factual",
    created_in_run: str = "",
    registry: Optional[ArtifactRegistry] = None,
) -> ArtifactEnvelope:
    qid = getattr(candidate, "q_id", "q")
    resolved_version = version if version is not None else candidate_artifact_version(candidate)
    return wrap_payload(
        candidate,
        artifact_type="CandidateMCQ",
        logical_id=str(qid),
        version=resolved_version,
        branch_id=branch_id,
        created_in_run=created_in_run,
        registry=registry,
    )


def wrap_evidence_report(
    report: Any,
    *,
    version: int = 1,
    branch_id: str = "factual",
    created_in_run: str = "",
    registry: Optional[ArtifactRegistry] = None,
) -> ArtifactEnvelope:
    qid = getattr(report, "q_id", "q")
    return wrap_payload(
        report,
        artifact_type="EvidenceCheckReport",
        logical_id=f"evidence-report:{qid}",
        version=version,
        branch_id=branch_id,
        created_in_run=created_in_run,
        registry=registry,
    )


def wrap_cross_cove_report(
    report: Any,
    *,
    version: int = 1,
    branch_id: str = "factual",
    created_in_run: str = "",
    registry: Optional[ArtifactRegistry] = None,
) -> ArtifactEnvelope:
    qid = getattr(report, "q_id", "q")
    return wrap_payload(
        report,
        artifact_type="CrossCoVeReport",
        logical_id=f"cross-cove-report:{qid}",
        version=version,
        branch_id=branch_id,
        created_in_run=created_in_run,
        registry=registry,
    )


def wrap_supervisor_decision(
    decision: Any,
    *,
    version: int = 1,
    branch_id: str = "factual",
    created_in_run: str = "",
    registry: Optional[ArtifactRegistry] = None,
) -> ArtifactEnvelope:
    qid = getattr(decision, "q_id", "q")
    return wrap_payload(
        decision,
        artifact_type="SupervisorDecision",
        logical_id=f"decision:{qid}",
        version=version,
        branch_id=branch_id,
        created_in_run=created_in_run,
        registry=registry,
    )


def wrap_outcome_assessment(
    assessment: Any,
    *,
    version: Optional[int] = None,
    branch_id: str = "factual",
    created_in_run: str = "",
    registry: Optional[ArtifactRegistry] = None,
) -> ArtifactEnvelope:
    assessment_ref = getattr(assessment, "assessment_ref", None)
    target_ref = getattr(assessment, "target_ref", None)
    if assessment_ref is not None:
        logical_id = assessment_ref.logical_id
        resolved_version = version if version is not None else assessment_ref.version
    elif target_ref is not None:
        logical_id = f"assessment:{target_ref.logical_id}"
        resolved_version = version if version is not None else target_ref.version
    else:
        logical_id = "assessment:unknown"
        resolved_version = version if version is not None else 1
    return wrap_payload(
        assessment,
        artifact_type="OutcomeAssessment",
        logical_id=str(logical_id),
        version=int(resolved_version),
        branch_id=branch_id,
        created_in_run=created_in_run,
        registry=registry,
    )


def wrap_quiz_pack(
    pack: Any,
    *,
    branch_id: str = "factual",
    created_in_run: str = "",
    registry: Optional[ArtifactRegistry] = None,
) -> ArtifactEnvelope:
    pid = getattr(pack, "pack_id", "pack")
    return wrap_payload(
        pack,
        artifact_type="QuizPack",
        logical_id=str(pid),
        branch_id=branch_id,
        created_in_run=created_in_run,
        registry=registry,
    )


def validate_artifact_snapshot(
    *,
    domain_payload: Any,
    envelope: ArtifactEnvelope,
) -> bool:
    """Return True when a registered envelope still matches the current domain payload."""
    return payloads_equivalent(domain_payload, envelope.payload)


def revise_candidate_version(
    registry: ArtifactRegistry,
    previous: ArtifactEnvelope,
    new_payload: Any,
    *,
    branch_id: Optional[str] = None,
    created_in_run: str = "",
) -> ArtifactEnvelope:
    """Create Q:vN+1 without mutating Q:vN; record DerivedFrom."""
    if previous.artifact_type != "CandidateMCQ":
        raise TypeError("revise_candidate_version expects CandidateMCQ envelope")
    new_ref = previous.ref.next_version()
    env = ArtifactEnvelope.create(
        logical_id=new_ref.logical_id,
        artifact_type="CandidateMCQ",
        payload=new_payload,
        branch_id=branch_id or previous.branch_id,
        created_in_run=created_in_run or previous.created_in_run,
        version=new_ref.version,
    )
    registry.store(env)
    registry.mark_superseded(previous.ref, env.branch_id)
    registry.derived_from.append(
        DerivedFrom(new_artifact_ref=env.ref, previous_artifact_ref=previous.ref)
    )
    return env


def unwrap(envelope: ArtifactEnvelope) -> Any:
    """Return the case-specific payload."""
    return envelope.payload
