"""Final quiz packaging from accepted CandidateMCQs only."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ...core.entities import QuizPack, ValidatedQuestion
from .coverage import compute_mcq_coverage, select_final_mcqs_for_quiz
from .generation import log_ok
from .section_evidence import update_section_evidence_status
from .versioning import candidate_version_number, get_operational_supervisor_decision


if TYPE_CHECKING:
    from ...core.state import CooperativeState


def _mcq_to_validated_question(mcq, state: "CooperativeState") -> ValidatedQuestion:
    blueprint = state.mcq_blueprints.get(mcq.blueprint_id)
    correct_text = ""
    for opt in mcq.options:
        if opt.option_id == mcq.correct_option_id:
            correct_text = opt.text
            break

    topic = ""
    qtype = "multiple_choice"
    difficulty = "intermediate"
    if blueprint is not None:
        topic = blueprint.section_id
        qtype = blueprint.question_type
        difficulty = blueprint.difficulty

    evidence_chunks = [ref.chunk_id for ref in mcq.source_refs if ref.chunk_id]
    related_atoms = [ref.atom_id for ref in mcq.source_refs if ref.atom_id]
    if blueprint is not None:
        related_atoms = list(dict.fromkeys(related_atoms + list(blueprint.target_atom_ids)))

    decision = get_operational_supervisor_decision(state, mcq)
    notes = "; ".join(decision.reasons) if decision else mcq.answer_explanation

    return ValidatedQuestion(
        q_id=mcq.q_id,
        question_text=mcq.stem,
        question_type=qtype,
        correct_answer=correct_text or mcq.correct_option_id,
        difficulty=difficulty,
        options=[o.text for o in mcq.options],
        topic_tags=[topic] if topic else [],
        related_atoms=related_atoms,
        evidence_chunks=evidence_chunks,
        validation_notes=notes,
        flags=list(mcq.flags),
    )


def build_coverage_summary(state: "CooperativeState") -> Dict[str, Any]:
    """Full-outline coverage summary for final quiz metadata."""
    update_section_evidence_status(state)
    coverage = compute_mcq_coverage(state)
    requirements = coverage.get("requirements", {})
    unsupported = [
        requirements["sections"][sid]["title"]
        for sid in coverage.get("unsupported_sections", [])
        if sid in requirements.get("sections", {})
    ]
    uncovered_supported = [
        requirements["sections"][sid]["title"]
        for sid in coverage.get("uncovered_supported_sections", [])
        if sid in requirements.get("sections", {})
    ]
    return {
        "outline_sections_total": coverage["coverage_sections_total"],
        "sections_with_evidence": coverage["sections_with_evidence"],
        "sections_covered_by_accepted_mcq": coverage["coverage_sections_covered"],
        "unsupported_sections": unsupported,
        "uncovered_supported_sections": uncovered_supported,
        "uncovered_due_to_missing_evidence": coverage.get(
            "uncovered_due_to_missing_evidence", []
        ),
        "learning_objectives_total": coverage["coverage_los_total"],
        "learning_objectives_with_evidence": coverage["learning_objectives_with_evidence"],
        "learning_objectives_covered": coverage["coverage_los_covered"],
        "total_candidates": coverage["total_candidates"],
        "accepted": coverage["accepted_count"],
        "needs_revision": coverage["needs_revision_count"],
        "rejected": coverage["rejected_count"],
        "per_section": coverage["per_section"],
        "per_lo": coverage["per_lo"],
    }


def build_pack_quality_metrics(state: "CooperativeState") -> Dict[str, Any]:
    coverage = compute_mcq_coverage(state)
    return {
        "total_candidates": coverage["total_candidates"],
        "accepted": coverage["accepted_count"],
        "needs_revision": coverage["needs_revision_count"],
        "rejected": coverage["rejected_count"],
        "coverage_sections_total": coverage["coverage_sections_total"],
        "coverage_sections_covered": coverage["coverage_sections_covered"],
        "coverage_los_total": coverage["coverage_los_total"],
        "coverage_los_covered": coverage["coverage_los_covered"],
        "avg_evidence_confidence": coverage["avg_evidence_confidence"],
        "hard_flag_count": coverage["hard_flag_count"],
    }


def package_final_quiz(
    state: "CooperativeState",
    title: str = "Generated Quiz",
    *,
    max_final_questions: Optional[int] = None,
) -> QuizPack:
    """Package only accepted CandidateMCQs into a QuizPack."""
    accepted_mcqs = select_final_mcqs_for_quiz(state, limit=max_final_questions)
    validated = [_mcq_to_validated_question(mcq, state) for mcq in accepted_mcqs]

    for vq in validated:
        state.validated_questions[vq.q_id] = vq

    coverage = compute_mcq_coverage(state)
    quality = build_pack_quality_metrics(state)
    coverage_summary = build_coverage_summary(state)
    authorizing_decision_refs = [
        f"decision:{mcq.q_id}:v{candidate_version_number(mcq)}"
        for mcq in accepted_mcqs
    ]
    pack = QuizPack(
        pack_id=state.generate_id("pack"),
        title=title,
        questions=validated,
        metadata={
            "quality_metrics": quality,
            "accepted_only": True,
            "authorizing_decision_refs": authorizing_decision_refs,
        },
        coverage_summary=coverage_summary,
    )
    state.append_quiz_pack(pack)
    state.quality_metrics.validated_questions = len(state.validated_questions)
    state.quality_metrics.total_questions = len(state.candidate_mcqs)
    if state.quality_metrics.total_questions > 0:
        state.quality_metrics.rejection_rate = (
            state.quality_metrics.rejected_questions / state.quality_metrics.total_questions
        )
    supported_total = coverage.get("coverage_sections_supported_total", 0)
    log_ok(
        f"Final coverage: {coverage['coverage_sections_covered']}/"
        f"{supported_total or coverage['coverage_sections_total']} supported sections"
    )
    log_ok(f"Packaged quiz {pack.pack_id} with {len(validated)} accepted MCQ(s)")
    return pack


def serialize_final_accepted_quiz(state: Optional["CooperativeState"]) -> Dict[str, Any]:
    """Serialize the last final QuizPack as the complete accepted factual quiz.

    Independent of AXIS explanation selection. Empty packs are an authoritative
    empty quiz, not missing data.
    """
    packs = list(getattr(state, "final_quiz_packs", None) or []) if state is not None else []
    if not packs:
        return {
            "kind": "final_accepted_factual_quiz",
            "pack_id": "",
            "accepted_count": 0,
            "questions": [],
        }
    pack = packs[-1]
    mcqs = getattr(state, "candidate_mcqs", None) or {}
    questions: List[Dict[str, Any]] = []
    for index, vq in enumerate(list(getattr(pack, "questions", None) or []), start=1):
        qid = str(getattr(vq, "q_id", "") or "")
        mcq = mcqs.get(qid) if isinstance(mcqs, dict) else None
        if mcq is not None:
            version = candidate_version_number(mcq)
            options = [
                {
                    "option_id": str(getattr(opt, "option_id", "") or ""),
                    "text": str(getattr(opt, "text", "") or ""),
                    "is_correct": bool(getattr(opt, "is_correct", False)),
                }
                for opt in list(getattr(mcq, "options", None) or [])
            ]
            stem = str(getattr(mcq, "stem", "") or "")
            correct_id = str(getattr(mcq, "correct_option_id", "") or "")
            correct_text = next(
                (
                    str(opt.get("text") or "")
                    for opt in options
                    if opt.get("is_correct") or opt.get("option_id") == correct_id
                ),
                "",
            )
        else:
            version = 1
            opt_texts = [str(t) for t in list(getattr(vq, "options", None) or [])]
            correct_text = str(getattr(vq, "correct_answer", "") or "")
            options = [
                {
                    "option_id": chr(ord("A") + i),
                    "text": text,
                    "is_correct": text == correct_text,
                }
                for i, text in enumerate(opt_texts)
            ]
            stem = str(getattr(vq, "question_text", "") or "")
            correct_id = next(
                (str(opt["option_id"]) for opt in options if opt.get("is_correct")),
                "",
            )
        questions.append(
            {
                "question_index": index,
                "q_id": qid,
                "candidate_ref": f"{qid}:v{int(version)}" if qid else "",
                "stem": stem,
                "options": options,
                "correct_option_id": correct_id,
                "correct_option": correct_text,
                "question_version": int(version),
            }
        )
    return {
        "kind": "final_accepted_factual_quiz",
        "pack_id": str(getattr(pack, "pack_id", "") or ""),
        "accepted_count": len(questions),
        "questions": questions,
    }
