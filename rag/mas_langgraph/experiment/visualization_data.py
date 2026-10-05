"""Researcher-only visualization payload for target-relevant CaseSlice data.

This module must not be imported by participant synthesis, verbalization,
faithfulness, or AXIS interrogation. It serializes structured occurrence /
artifact / provenance data for a later graph UI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..artifacts.models import ArtifactEnvelope, ArtifactRef
from ..artifacts.registry import ArtifactRegistry
from ..axis.models import ExplainedCase
from ..axis.slicing import CaseSlice
from ..core.entities import DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
from ..core.outcomes import CriterionValue, OutcomeAssessment
from ..domains.quiz.entities import (
    CandidateMCQ,
    CrossCoVeReport,
    EvidenceCheckReport,
    MCQBlueprint,
    SourceRef,
    SupervisorDecision,
)
from ..provenance.models import ArtifactBinding, OccurrenceRecord, RunProvenanceRelation

VISUALIZATION_SCHEMA_VERSION = 1


def serialize_ref(ref: Any) -> Optional[Dict[str, Any]]:
    """Stable ArtifactRef JSON: logical_id, version, key."""
    if ref is None:
        return None
    logical_id = str(getattr(ref, "logical_id", "") or "")
    version = int(getattr(ref, "version", 1) or 1)
    key = str(ref) if not isinstance(ref, str) else ref
    if not logical_id and ":" in key:
        # Fallback only when the object already stringifies as id:vN.
        pass
    return {
        "logical_id": logical_id,
        "version": version,
        "key": f"{logical_id}:v{version}" if logical_id else key,
    }


def json_safe(value: Any) -> Any:
    """Convert a value to JSON-compatible types without ``default=str``."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, ArtifactRef):
        return serialize_ref(value)
    if isinstance(value, SourceRef):
        return serialize_source_ref(value)
    if isinstance(value, ArtifactBinding):
        return serialize_binding(value)
    if isinstance(value, CriterionValue):
        return serialize_criterion(value)
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        return {str(k): json_safe(v) for k, v in items}
    raise TypeError(
        f"visualization_data cannot serialize {type(value).__name__} without an explicit projector"
    )


def serialize_source_ref(src: Any) -> Dict[str, Any]:
    return {
        "doc_id": str(getattr(src, "doc_id", "") or ""),
        "page": getattr(src, "page", None),
        "chunk_id": getattr(src, "chunk_id", None),
        "atom_id": getattr(src, "atom_id", None),
        "quote": getattr(src, "quote", None),
    }


def serialize_binding(binding: Any) -> Dict[str, Any]:
    return {
        "relation": str(getattr(binding, "relation", "") or ""),
        "artifact_ref": serialize_ref(getattr(binding, "artifact_ref", None)),
    }


def serialize_criterion(cv: Any) -> Dict[str, Any]:
    value = getattr(cv, "value", None)
    if value is not None and not isinstance(value, (str, int, float, bool)):
        try:
            value = json_safe(value)
        except TypeError:
            value = None
    return {
        "name": str(getattr(cv, "name", "") or ""),
        "value": value,
        "value_type": str(getattr(cv, "value_type", "") or getattr(cv, "expected_type", "") or ""),
        "label": str(getattr(cv, "label", "") or ""),
        "explanation": str(getattr(cv, "explanation", "") or ""),
        "decision_relevant": bool(getattr(cv, "decision_relevant", False)),
    }


def serialize_occurrence(occ: Any) -> Dict[str, Any]:
    produced = [
        serialize_ref(r)
        for r in (getattr(occ, "produced_artifacts", ()) or ())
        if r is not None
    ]
    bindings = [
        serialize_binding(b)
        for b in (getattr(occ, "input_bindings", ()) or ())
        if b is not None
    ]
    bindings.sort(
        key=lambda b: (
            str(b.get("relation") or ""),
            str((b.get("artifact_ref") or {}).get("logical_id") or ""),
            int((b.get("artifact_ref") or {}).get("version") or 0),
        )
    )
    produced.sort(
        key=lambda r: (str(r.get("logical_id") or ""), int(r.get("version") or 0))
    )
    delta = getattr(occ, "state_delta", None) or {}
    superstep = getattr(occ, "superstep", None)
    return {
        "occurrence_id": str(getattr(occ, "occurrence_id", "") or ""),
        "node_id": str(getattr(occ, "node_id", "") or ""),
        "occurrence_index": int(getattr(occ, "occurrence_index", 0) or 0),
        "execution_index": int(getattr(occ, "execution_index", 0) or 0),
        "superstep": None if superstep is None else int(superstep),
        "branch_id": str(getattr(occ, "branch_id", "") or ""),
        "run_id": str(getattr(occ, "run_id", "") or ""),
        "checkpoint_before": getattr(occ, "checkpoint_before", None),
        "checkpoint_after": getattr(occ, "checkpoint_after", None),
        "error": getattr(occ, "error", None),
        "produced_artifacts": produced,
        "input_bindings": bindings,
        "has_state_delta": bool(delta),
    }


def serialize_relation(rel: Any) -> Dict[str, Any]:
    return {
        "relation": str(getattr(rel, "relation", "") or ""),
        "artifact_ref": serialize_ref(getattr(rel, "artifact_ref", None)),
        "producer_occurrence_id": str(getattr(rel, "producer_occurrence_id", "") or ""),
        "consumer_occurrence_id": str(getattr(rel, "consumer_occurrence_id", "") or ""),
        "artifact_type": str(getattr(rel, "artifact_type", "") or ""),
    }


def _option_content(opt: Any) -> Dict[str, Any]:
    refs = [
        serialize_source_ref(s)
        for s in (getattr(opt, "source_refs", None) or [])
    ]
    return {
        "option_id": str(getattr(opt, "option_id", "") or ""),
        "text": str(getattr(opt, "text", "") or ""),
        "is_correct": bool(getattr(opt, "is_correct", False)),
        "explanation": getattr(opt, "explanation", None),
        "misconception": getattr(opt, "misconception", None),
        "source_refs": refs,
    }


def _project_doc_section(payload: Any) -> Dict[str, Any]:
    slide = getattr(payload, "slide_range", None)
    if isinstance(slide, tuple):
        slide = list(slide)
    return {
        "section_id": str(getattr(payload, "section_id", "") or ""),
        "title": str(getattr(payload, "title", "") or ""),
        "slide_range": json_safe(slide) if slide is not None else None,
        "importance_weight": float(getattr(payload, "importance_weight", 0.0) or 0.0),
        "content_summary": getattr(payload, "content_summary", None),
    }


def _project_learning_objective(payload: Any) -> Dict[str, Any]:
    return {
        "lo_id": str(getattr(payload, "lo_id", "") or ""),
        "section_id": str(getattr(payload, "section_id", "") or ""),
        "objective_text": str(getattr(payload, "objective_text", "") or ""),
        "difficulty_level": str(getattr(payload, "difficulty_level", "") or ""),
    }


def _project_evidence_chunk(payload: Any) -> Dict[str, Any]:
    metadata = getattr(payload, "metadata", None) or {}
    return {
        "chunk_id": str(getattr(payload, "chunk_id", "") or ""),
        "content": str(getattr(payload, "content", "") or ""),
        "metadata": json_safe(dict(metadata)) if isinstance(metadata, Mapping) else {},
        "relevance_score": float(getattr(payload, "relevance_score", 0.0) or 0.0),
        "quality_score": float(getattr(payload, "quality_score", 0.0) or 0.0),
        "relevance_tags": list(getattr(payload, "relevance_tags", None) or []),
        "used_in_atoms": list(getattr(payload, "used_in_atoms", None) or []),
        "duplicate_of": getattr(payload, "duplicate_of", None),
        "flags": list(getattr(payload, "flags", None) or []),
    }


def _project_knowledge_atom(payload: Any) -> Dict[str, Any]:
    citations = getattr(payload, "citations", None) or []
    return {
        "atom_id": str(getattr(payload, "atom_id", "") or ""),
        "content": str(getattr(payload, "content", "") or ""),
        "atom_type": str(getattr(payload, "atom_type", "") or ""),
        "citations": json_safe(list(citations)),
        "section_id": getattr(payload, "section_id", None),
        "lo_id": getattr(payload, "lo_id", None),
    }


def _project_blueprint(payload: Any) -> Dict[str, Any]:
    return {
        "blueprint_id": str(getattr(payload, "blueprint_id", "") or ""),
        "section_id": str(getattr(payload, "section_id", "") or ""),
        "lo_id": str(getattr(payload, "lo_id", "") or ""),
        "target_atom_ids": list(getattr(payload, "target_atom_ids", None) or []),
        "evidence_chunk_ids": list(getattr(payload, "evidence_chunk_ids", None) or []),
        "question_type": str(getattr(payload, "question_type", "") or ""),
        "difficulty": str(getattr(payload, "difficulty", "") or ""),
        "bloom_level": str(getattr(payload, "bloom_level", "") or ""),
        "generation_goal": str(getattr(payload, "generation_goal", "") or ""),
        "status": str(getattr(payload, "status", "") or ""),
    }


def _project_candidate_mcq(payload: Any) -> Dict[str, Any]:
    options = [_option_content(o) for o in (getattr(payload, "options", None) or [])]
    return {
        "q_id": str(getattr(payload, "q_id", "") or ""),
        "blueprint_id": str(getattr(payload, "blueprint_id", "") or ""),
        "stem": str(getattr(payload, "stem", "") or ""),
        "options": options,
        "correct_option_id": str(getattr(payload, "correct_option_id", "") or ""),
        "answer_explanation": str(getattr(payload, "answer_explanation", "") or ""),
        "source_refs": [
            serialize_source_ref(s)
            for s in (getattr(payload, "source_refs", None) or [])
        ],
        "status": str(getattr(payload, "status", "") or ""),
        "revision_count": int(getattr(payload, "revision_count", 0) or 0),
        "flags": list(getattr(payload, "flags", None) or []),
    }


def _project_evidence_report(payload: Any) -> Dict[str, Any]:
    return {
        "q_id": str(getattr(payload, "q_id", "") or ""),
        "candidate_version": int(getattr(payload, "candidate_version", 1) or 1),
        "supported_by_pdf": bool(getattr(payload, "supported_by_pdf", False)),
        "confidence": float(getattr(payload, "confidence", 0.0) or 0.0),
        "stem_supported": bool(getattr(payload, "stem_supported", False)),
        "answer_supported": bool(getattr(payload, "answer_supported", False)),
        "distractors_valid": bool(getattr(payload, "distractors_valid", False)),
        "explanation_supported": bool(getattr(payload, "explanation_supported", False)),
        "unsupported_claims": list(getattr(payload, "unsupported_claims", None) or []),
        "required_fixes": list(getattr(payload, "required_fixes", None) or []),
        "notes": list(getattr(payload, "notes", None) or []),
        "source_refs": [
            serialize_source_ref(s)
            for s in (getattr(payload, "source_refs", None) or [])
        ],
    }


def _project_cross_cove(payload: Any) -> Dict[str, Any]:
    return {
        "q_id": str(getattr(payload, "q_id", "") or ""),
        "candidate_version": int(getattr(payload, "candidate_version", 1) or 1),
        "generator_answer": str(getattr(payload, "generator_answer", "") or ""),
        "verifier_answers": list(getattr(payload, "verifier_answers", None) or []),
        "majority_answer": str(getattr(payload, "majority_answer", "") or ""),
        "passed": bool(getattr(payload, "passed", False)),
        "confidence": float(getattr(payload, "confidence", 0.0) or 0.0),
        "verifier_model": getattr(payload, "verifier_model", None),
        "notes": list(getattr(payload, "notes", None) or []),
    }


def _project_decision(payload: Any) -> Dict[str, Any]:
    return {
        "q_id": str(getattr(payload, "q_id", "") or ""),
        "candidate_version": int(getattr(payload, "candidate_version", 1) or 1),
        "decision": str(getattr(payload, "decision", "") or ""),
        "reasons": list(getattr(payload, "reasons", None) or []),
        "required_fixes": list(getattr(payload, "required_fixes", None) or []),
        "assessment_ref": serialize_ref(getattr(payload, "assessment_ref", None)),
    }


def _project_outcome_assessment(payload: Any) -> Dict[str, Any]:
    criteria = [
        serialize_criterion(c)
        for c in (getattr(payload, "criteria", ()) or ())
    ]
    criteria.sort(key=lambda c: str(c.get("name") or ""))
    notes = getattr(payload, "notes", ()) or ()
    return {
        "assessment_ref": serialize_ref(getattr(payload, "assessment_ref", None)),
        "target_ref": serialize_ref(getattr(payload, "target_ref", None)),
        "criteria": criteria,
        "notes": list(notes),
    }


_PROJECTORS = {
    "DocSection": _project_doc_section,
    "LearningObjective": _project_learning_objective,
    "EvidenceChunk": _project_evidence_chunk,
    "KnowledgeAtom": _project_knowledge_atom,
    "MCQBlueprint": _project_blueprint,
    "CandidateMCQ": _project_candidate_mcq,
    "EvidenceCheckReport": _project_evidence_report,
    "CrossCoVeReport": _project_cross_cove,
    "SupervisorDecision": _project_decision,
    "OutcomeAssessment": _project_outcome_assessment,
}

_PAYLOAD_TYPE_FALLBACK = (
    (LearningObjective, "LearningObjective"),
    (DocSection, "DocSection"),
    (EvidenceChunk, "EvidenceChunk"),
    (KnowledgeAtom, "KnowledgeAtom"),
    (MCQBlueprint, "MCQBlueprint"),
    (CandidateMCQ, "CandidateMCQ"),
    (EvidenceCheckReport, "EvidenceCheckReport"),
    (CrossCoVeReport, "CrossCoVeReport"),
    (SupervisorDecision, "SupervisorDecision"),
    (OutcomeAssessment, "OutcomeAssessment"),
)


def _resolve_artifact_type(env: ArtifactEnvelope) -> str:
    declared = str(getattr(env, "artifact_type", "") or "")
    if declared:
        return declared
    payload = getattr(env, "payload", None)
    for cls, name in _PAYLOAD_TYPE_FALLBACK:
        if isinstance(payload, cls):
            return name
    return type(payload).__name__ if payload is not None else ""


def _producer_map(
    occurrences: Sequence[Any],
    produced_by: Optional[Sequence[Any]] = None,
) -> Dict[Tuple[str, int], str]:
    """Recorded producers only. Omit when more than one producer is recorded."""
    counts: Dict[Tuple[str, int], Dict[str, int]] = {}

    def _add(ref: Any, occ_id: str) -> None:
        if ref is None or not occ_id:
            return
        key = (str(getattr(ref, "logical_id", "") or ""), int(getattr(ref, "version", 1) or 1))
        if not key[0]:
            return
        bucket = counts.setdefault(key, {})
        bucket[str(occ_id)] = bucket.get(str(occ_id), 0) + 1

    for occ in occurrences or ():
        oid = str(getattr(occ, "occurrence_id", "") or "")
        for ref in getattr(occ, "produced_artifacts", ()) or ():
            _add(ref, oid)
    for pb in produced_by or ():
        _add(getattr(pb, "artifact_ref", None), getattr(pb, "producer_occurrence_id", "") or "")

    out: Dict[Tuple[str, int], str] = {}
    for key, bucket in counts.items():
        if len(bucket) == 1:
            out[key] = next(iter(bucket))
    return out


def serialize_visualization_artifact(
    env: ArtifactEnvelope,
    *,
    produced_by_occurrence_id: Optional[str] = None,
) -> Dict[str, Any]:
    artifact_type = _resolve_artifact_type(env)
    projector = _PROJECTORS.get(artifact_type)
    content: Dict[str, Any] = projector(env.payload) if projector is not None else {}
    row: Dict[str, Any] = {
        "ref": serialize_ref(env.ref),
        "artifact_type": artifact_type,
        "branch_id": str(getattr(env, "branch_id", "") or ""),
        "content": content,
    }
    if produced_by_occurrence_id:
        row["produced_by_occurrence_id"] = str(produced_by_occurrence_id)
    return row


def _lookup_envelope(
    registry: Optional[ArtifactRegistry],
    ref: ArtifactRef,
    branch_id: str,
) -> Optional[ArtifactEnvelope]:
    if registry is None:
        return None
    return registry.get_by_ref(ref, branch_id) or registry.get_by_ref(ref)


def build_case_visualization_data(
    *,
    explained_case: ExplainedCase,
    case_slice: Optional[CaseSlice],
    registry: Optional[ArtifactRegistry],
    produced_by: Optional[Sequence[Any]] = None,
    factual_decision: str = "",
) -> Dict[str, Any]:
    """Build a target-relevant, JSON-safe visualization payload from CaseSlice."""
    occs = list(getattr(case_slice, "relevant_occurrences", ()) or ()) if case_slice else []
    rels = list(getattr(case_slice, "relevant_relations", ()) or ()) if case_slice else []
    refs = list(getattr(case_slice, "relevant_artifact_refs", ()) or ()) if case_slice else []
    branch_id = str(getattr(explained_case, "branch_id", "") or "factual")

    occs.sort(
        key=lambda o: (
            int(getattr(o, "execution_index", 0) or 0),
            str(getattr(o, "occurrence_id", "") or ""),
        )
    )
    serialized_occs = [serialize_occurrence(o) for o in occs]

    producers = _producer_map(occs, produced_by=produced_by)
    artifacts: List[Dict[str, Any]] = []
    for ref in refs:
        env = _lookup_envelope(registry, ref, branch_id)
        if env is None:
            artifacts.append(
                {
                    "ref": serialize_ref(ref),
                    "artifact_type": "",
                    "branch_id": branch_id,
                    "content": {},
                }
            )
            continue
        key = (str(env.ref.logical_id), int(env.ref.version))
        artifacts.append(
            serialize_visualization_artifact(
                env,
                produced_by_occurrence_id=producers.get(key),
            )
        )
    artifacts.sort(
        key=lambda a: (
            str(a.get("artifact_type") or ""),
            str((a.get("ref") or {}).get("logical_id") or ""),
            int((a.get("ref") or {}).get("version") or 0),
        )
    )

    serialized_rels = [serialize_relation(r) for r in rels]
    serialized_rels.sort(
        key=lambda r: (
            str(r.get("producer_occurrence_id") or ""),
            str(r.get("consumer_occurrence_id") or ""),
            str(r.get("relation") or ""),
            str((r.get("artifact_ref") or {}).get("logical_id") or ""),
            int((r.get("artifact_ref") or {}).get("version") or 0),
        )
    )

    criteria = [
        serialize_criterion(cv)
        for cv in (getattr(case_slice, "criterion_values", ()) or ())
    ] if case_slice is not None else []
    criteria.sort(key=lambda c: str(c.get("name") or ""))

    decision = (
        factual_decision
        or str(getattr(explained_case, "outcome", "") or "")
        or str(getattr(case_slice, "decision", "") or "")
    )
    return {
        "schema_version": VISUALIZATION_SCHEMA_VERSION,
        "target": {
            "candidate_ref": serialize_ref(getattr(explained_case, "artifact_ref", None)),
            "assessment_ref": serialize_ref(getattr(explained_case, "assessment_ref", None)),
            "decision_ref": serialize_ref(getattr(explained_case, "decision_ref", None)),
            "factual_decision": str(decision or ""),
        },
        "occurrences": serialized_occs,
        "artifacts": artifacts,
        "provenance_relations": serialized_rels,
        "criteria": criteria,
    }
