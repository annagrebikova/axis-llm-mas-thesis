"""Adapters between legacy CandidateQuestion and CandidateMCQ."""

from __future__ import annotations

from typing import List, Optional

from ...core.entities import CandidateQuestion
from .entities import AnswerOption, CandidateMCQ, SourceRef


def _parse_legacy_options(
    options: Optional[List[str]],
    correct_answer: Optional[str],
) -> tuple[List[AnswerOption], str]:
    """Convert legacy string options into AnswerOption list."""
    if not options:
        return [], ""

    parsed: List[AnswerOption] = []
    correct_id = ""
    correct_norm = (correct_answer or "").strip().lower()

    for idx, raw in enumerate(options):
        option_id = f"opt_{idx}"
        text = raw.strip()
        letter = text[:1].lower() if text else ""
        is_correct = False
        if correct_norm:
            is_correct = (
                text.lower() == correct_norm
                or correct_norm == letter
                or text.lower().startswith(f"{correct_norm})")
                or (len(correct_norm) == 1 and letter == correct_norm)
            )
        parsed.append(
            AnswerOption(option_id=option_id, text=text, is_correct=is_correct)
        )
        if is_correct:
            correct_id = option_id

    if not correct_id and parsed:
        parsed[0].is_correct = True
        correct_id = parsed[0].option_id

    return parsed, correct_id


def candidate_question_to_candidate_mcq(
    question: CandidateQuestion,
    *,
    blueprint_id: str = "",
    section_id: str = "",
    lo_id: str = "",
) -> CandidateMCQ:
    """Map legacy CandidateQuestion to structured CandidateMCQ."""
    options, correct_option_id = _parse_legacy_options(
        question.options,
        question.correct_answer,
    )
    source_refs: List[SourceRef] = []
    for chunk_id in question.evidence_chunks:
        source_refs.append(
            SourceRef(doc_id=section_id or "document", page=None, chunk_id=chunk_id)
        )
    for atom_id in question.related_atoms:
        source_refs.append(
            SourceRef(doc_id=section_id or "document", page=None, chunk_id=None, atom_id=atom_id)
        )

    topic_tag = question.topic_tags[0] if question.topic_tags else section_id
    if topic_tag and topic_tag.lower() == "general":
        topic_tag = section_id or lo_id or "section"

    return CandidateMCQ(
        q_id=question.q_id,
        blueprint_id=blueprint_id,
        stem=question.question_text,
        options=options,
        correct_option_id=correct_option_id,
        answer_explanation=f"Derived from legacy candidate ({topic_tag}).",
        source_refs=source_refs,
        status="draft",
        flags=list(question.flags),
    )


def candidate_mcq_to_candidate_question(
    mcq: CandidateMCQ,
    *,
    section_id: str = "",
    lo_id: str = "",
    blueprint: Optional[object] = None,
) -> CandidateQuestion:
    """Map CandidateMCQ back to legacy CandidateQuestion for classic orchestrator paths."""
    correct_text = ""
    for opt in mcq.options:
        if opt.option_id == mcq.correct_option_id:
            correct_text = opt.text
            break

    topic = section_id or lo_id
    if blueprint is not None:
        topic = getattr(blueprint, "section_id", None) or getattr(blueprint, "lo_id", None) or topic

    qtype = "multiple_choice"
    difficulty = "intermediate"
    if blueprint is not None:
        qtype = getattr(blueprint, "question_type", qtype)
        difficulty = getattr(blueprint, "difficulty", difficulty)

    return CandidateQuestion(
        q_id=mcq.q_id,
        question_text=mcq.stem,
        question_type=qtype,
        options=[o.text for o in mcq.options],
        correct_answer=correct_text or mcq.correct_option_id,
        difficulty=difficulty,
        topic_tags=[topic] if topic else [],
        related_atoms=[
            ref.atom_id for ref in mcq.source_refs if ref.atom_id
        ],
        evidence_chunks=[
            ref.chunk_id for ref in mcq.source_refs if ref.chunk_id
        ],
        flags=list(mcq.flags),
    )
