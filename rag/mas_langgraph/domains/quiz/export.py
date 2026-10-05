"""Export MCQs for external IWF evaluation (outside the MAS)."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .entities import AnswerOption, CrossCoVeReport, EvidenceCheckReport, RevisionRecord, SourceRef, SupervisorDecision

if TYPE_CHECKING:
    from ...core.state import CooperativeState


STRUCTURAL_FAILURE_FLAGS = frozenset({
    "empty_stem",
    "too_few_options",
    "no_correct_option",
    "multiple_correct_options",
    "correct_option_id_missing",
    "missing_answer_explanation",
    "missing_source_refs",
    "duplicate_options",
    "duplicate_option_ids",
    "empty_option_text",
    "insufficient_evidence",
    "insufficient_atoms",
    "llm_parse_error",
})


def _serialize_source_ref(ref: SourceRef) -> Dict[str, Any]:
    return asdict(ref)


def _serialize_option(opt: AnswerOption) -> Dict[str, Any]:
    payload = asdict(opt)
    payload["source_refs"] = [_serialize_source_ref(r) for r in opt.source_refs]
    return payload


def _serialize_evidence_report(report: Optional[EvidenceCheckReport]) -> Dict[str, Any]:
    if report is None:
        return {
            "supported_by_pdf": None,
            "confidence": None,
            "unsupported_claims": [],
            "required_fixes": [],
            "notes": [],
            "stem_supported": None,
            "answer_supported": None,
            "distractors_valid": None,
            "explanation_supported": None,
        }
    return {
        "supported_by_pdf": report.supported_by_pdf,
        "confidence": report.confidence,
        "unsupported_claims": list(report.unsupported_claims),
        "required_fixes": list(report.required_fixes),
        "notes": list(report.notes),
        "stem_supported": report.stem_supported,
        "answer_supported": report.answer_supported,
        "distractors_valid": report.distractors_valid,
        "explanation_supported": report.explanation_supported,
        # Legacy flat keys for downstream consumers
        "evidence_supported_by_pdf": report.supported_by_pdf,
        "evidence_confidence": report.confidence,
        "evidence_notes": list(report.notes),
    }


def _serialize_supervisor_decision(decision: Optional[SupervisorDecision]) -> Dict[str, Any]:
    if decision is None:
        return {}
    return asdict(decision)


def _serialize_cross_cove_report(report: Optional[CrossCoVeReport]) -> Dict[str, Any]:
    if report is None:
        return {}
    return {
        "passed": report.passed,
        "generator_answer": report.generator_answer,
        "verifier_answers": list(report.verifier_answers),
        "majority_answer": report.majority_answer,
        "confidence": report.confidence,
        "verifier_model": report.verifier_model,
        "notes": list(report.notes),
    }


def _serialize_revision_record(record: RevisionRecord) -> Dict[str, Any]:
    payload = asdict(record)
    payload["previous_options"] = [
        _serialize_option(o) if isinstance(o, AnswerOption) else o
        for o in record.previous_options
    ]
    return payload


def _revision_history_for_qid(state: "CooperativeState", q_id: str) -> List[Dict[str, Any]]:
    return [
        _serialize_revision_record(r)
        for r in getattr(state, "revision_history", [])
        if r.q_id == q_id
    ]


def _blueprint_lookup(state: "CooperativeState", blueprint_id: str):
    return state.mcq_blueprints.get(blueprint_id)


def export_mcqs_for_external_iwf_check(
    state: "CooperativeState",
    *,
    accepted_only: bool = True,
) -> List[Dict[str, Any]]:
    """Serialize CandidateMCQs with evidence and supervisor decisions for external IWF."""
    exported: List[Dict[str, Any]] = []

    for mcq in state.candidate_mcqs.values():
        if accepted_only and mcq.status != "accepted":
            continue
        if not accepted_only and mcq.status == "rejected":
            continue

        blueprint = _blueprint_lookup(state, mcq.blueprint_id)
        section_id = blueprint.section_id if blueprint else ""
        lo_id = blueprint.lo_id if blueprint else ""
        evidence_report = state.evidence_reports.get(mcq.q_id)
        supervisor_decision = state.supervisor_decisions.get(mcq.q_id)
        cross_cove_report = state.cross_cove_reports.get(mcq.q_id)

        distractor_explanations = [
            {"option_id": o.option_id, "explanation": o.explanation, "misconception": o.misconception}
            for o in mcq.options
            if not o.is_correct
        ]

        exported.append({
            "q_id": mcq.q_id,
            "blueprint_id": mcq.blueprint_id,
            "section_id": section_id,
            "lo_id": lo_id,
            "stem": mcq.stem,
            "options": [_serialize_option(o) for o in mcq.options],
            "correct_option_id": mcq.correct_option_id,
            "answer_explanation": mcq.answer_explanation,
            "distractor_explanations": distractor_explanations,
            "misconceptions": [
                o.misconception for o in mcq.options if o.misconception
            ],
            "source_refs": [_serialize_source_ref(r) for r in mcq.source_refs],
            "evidence_report": _serialize_evidence_report(evidence_report),
            "cross_cove_report": _serialize_cross_cove_report(cross_cove_report),
            "supervisor_decision": _serialize_supervisor_decision(supervisor_decision),
            "candidate_flags": list(mcq.flags),
            "flags": list(mcq.flags),
            "status": mcq.status,
            "revision_count": mcq.revision_count,
            "revision_history": _revision_history_for_qid(state, mcq.q_id),
            "question_type": getattr(blueprint, "question_type", None),
            "difficulty": getattr(blueprint, "difficulty", None),
            "bloom_level": getattr(blueprint, "bloom_level", None),
            **_serialize_evidence_report(evidence_report),
        })

    return exported
