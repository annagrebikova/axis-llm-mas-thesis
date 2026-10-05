"""MAS-internal revision for CandidateMCQs failing evidence, Cross-CoVe, or acceptance."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .entities import AnswerOption, CandidateMCQ, RevisionRecord
from .generation import (
    OPTION_LABELS,
    _normalize_title,
    _source_refs_from_context,
    build_generation_context_from_blueprint,
    generate_distractors_from_blueprint,
    generate_explanations_from_blueprint,
    log_ok,
    log_warn,
    validate_candidate_mcq_structure,
)
from .option_positioning import assign_deterministic_option_positions
from .versioning import (
    candidate_version_number,
    clear_operational_reports,
    get_operational_cross_cove_report,
    get_operational_evidence_report,
    get_operational_supervisor_decision,
)

if TYPE_CHECKING:
    from ...core.state import CooperativeState

REVISION_CLEARABLE_FLAGS = frozenset({
    "evidence_not_supported",
    "answer_not_supported",
    "distractors_not_valid",
    "explanation_not_supported",
    "cross_cove_failed",
    "cross_cove_no_majority",
    "missing_answer_explanation",
    "duplicate_options",
    "multiple_correct_options",
    "no_correct_option",
    "correct_option_id_missing",
    "empty_stem",
    "empty_option_text",
})


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _invoke_llm(llm: Any, prompt: str) -> str:
    from ...replay.llm_determinism import workflow_llm_context

    with workflow_llm_context(layer="native_mcq", component="mcq_revision"):
        response = llm.invoke(prompt)
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)


def collect_revision_feedback(
    candidate: CandidateMCQ,
    state: "CooperativeState",
) -> Dict[str, Any]:
    """Collect structured revision feedback from reports, flags, and blueprint context."""
    blueprint = state.mcq_blueprints.get(candidate.blueprint_id)
    evidence_report = get_operational_evidence_report(state, candidate)
    cross_cove_report = get_operational_cross_cove_report(state, candidate)
    supervisor = get_operational_supervisor_decision(state, candidate)

    section_title = ""
    learning_objective = ""
    evidence_chunks: List[Dict[str, Any]] = []
    atoms: List[Dict[str, Any]] = []

    if blueprint is not None:
        section = state.get_section(blueprint.section_id)
        section_title = _normalize_title(
            section.title if section else blueprint.section_id,
            blueprint.section_id,
        )
        for lo in state.learning_objectives:
            if lo.lo_id == blueprint.lo_id:
                learning_objective = lo.objective_text
                break
        context = build_generation_context_from_blueprint(blueprint, state)
        evidence_chunks = list(context.get("evidence_chunks") or [])
        atoms = list(context.get("atoms") or [])

    reasons_for_revision: List[str] = []
    required_fixes: List[str] = []
    unsupported_claims: List[str] = []

    if supervisor is not None:
        reasons_for_revision.extend(supervisor.reasons)
        required_fixes.extend(supervisor.required_fixes)

    for flag in candidate.flags:
        if flag == "insufficient_atoms":
            reasons_for_revision.append("Candidate has insufficient_atoms flag.")
        elif flag == "insufficient_evidence":
            reasons_for_revision.append("Candidate has insufficient_evidence flag.")
        elif flag == "cross_cove_failed":
            reasons_for_revision.append("Cross-CoVe failed.")
        elif flag == "evidence_not_supported":
            reasons_for_revision.append("The answer is not supported by PDF evidence.")

    if evidence_report is not None:
        unsupported_claims.extend(evidence_report.unsupported_claims)
        required_fixes.extend(evidence_report.required_fixes)
        if not evidence_report.supported_by_pdf:
            reasons_for_revision.append("The answer is not supported by PDF evidence.")
        if not evidence_report.answer_supported:
            reasons_for_revision.append("Evidence report indicates answer is not supported.")
        if not evidence_report.distractors_valid:
            reasons_for_revision.append(
                "Distractors are not valid because one distractor may also be correct."
            )
        if not evidence_report.explanation_supported:
            reasons_for_revision.append("Answer explanation is not supported by evidence.")

    cross_cove_majority = ""
    generator_answer = ""
    if cross_cove_report is not None:
        cross_cove_majority = cross_cove_report.majority_answer or ""
        generator_answer = cross_cove_report.generator_answer or ""
        if not cross_cove_report.passed:
            reasons_for_revision.append(
                f"Cross-CoVe selected option {cross_cove_majority} "
                f"while generator marked option {generator_answer}."
            )

    if not (candidate.answer_explanation or "").strip():
        reasons_for_revision.append("Answer explanation is missing.")

    return {
        "q_id": candidate.q_id,
        "reasons_for_revision": list(dict.fromkeys(reasons_for_revision)),
        "required_fixes": list(dict.fromkeys(required_fixes)),
        "unsupported_claims": unsupported_claims,
        "cross_cove_majority_answer": cross_cove_majority,
        "generator_answer": generator_answer,
        "candidate_flags": list(candidate.flags),
        "section_title": section_title,
        "learning_objective": learning_objective,
        "question_type": getattr(blueprint, "question_type", ""),
        "difficulty": getattr(blueprint, "difficulty", ""),
        "bloom_level": getattr(blueprint, "bloom_level", ""),
        "evidence_chunks": evidence_chunks,
        "knowledge_atoms": atoms,
        "blueprint": blueprint,
    }


def build_revision_prompt(
    candidate: CandidateMCQ,
    feedback: Dict[str, Any],
) -> str:
    """Build LLM prompt to revise MCQ using PDF evidence and revision feedback."""
    options_block = json.dumps(
        [
            {
                "option_id": o.option_id,
                "text": o.text,
                "is_correct": o.is_correct,
                "explanation": o.explanation,
                "misconception": o.misconception,
            }
            for o in candidate.options
        ],
        indent=2,
        ensure_ascii=False,
    )
    feedback_block = json.dumps(
        {
            "reasons_for_revision": feedback.get("reasons_for_revision"),
            "required_fixes": feedback.get("required_fixes"),
            "unsupported_claims": feedback.get("unsupported_claims"),
            "cross_cove_majority_answer": feedback.get("cross_cove_majority_answer"),
            "generator_answer": feedback.get("generator_answer"),
        },
        indent=2,
        ensure_ascii=False,
    )
    evidence_block = json.dumps(feedback.get("evidence_chunks") or [], indent=2, ensure_ascii=False)
    atoms_block = json.dumps(feedback.get("knowledge_atoms") or [], indent=2, ensure_ascii=False)

    return f"""You are revising a multiple-choice question using only the provided PDF evidence.
Do not use outside knowledge.
Do not change the target section or learning objective.
Fix only the problems listed in the revision feedback.
Preserve the intended question type and difficulty if possible.
Return JSON only.

Section title: {feedback.get("section_title")}
Learning objective: {feedback.get("learning_objective")}
Question type: {feedback.get("question_type")}
Difficulty: {feedback.get("difficulty")}
Bloom level: {feedback.get("bloom_level")}

Original stem:
{candidate.stem}

Original options:
{options_block}

Original correct option id: {candidate.correct_option_id}
Original answer explanation: {candidate.answer_explanation}

Evidence chunks:
{evidence_block}

Knowledge atoms:
{atoms_block}

Revision feedback:
{feedback_block}

Return JSON only:
{{
  "stem": "...",
  "options": [
    {{
      "option_id": "A",
      "text": "...",
      "is_correct": false,
      "explanation": "...",
      "misconception": "..."
    }},
    {{
      "option_id": "B",
      "text": "...",
      "is_correct": false,
      "explanation": "...",
      "misconception": "..."
    }},
    {{
      "option_id": "C",
      "text": "...",
      "is_correct": false,
      "explanation": "...",
      "misconception": "..."
    }},
    {{
      "option_id": "D",
      "text": "...",
      "is_correct": false,
      "explanation": "...",
      "misconception": "..."
    }}
  ],
  "correct_option_id": "<letter of the single correct option>",
  "answer_explanation": "...",
  "changed_fields": ["stem", "options", "answer_explanation"],
  "revision_notes": ["..."]
}}

Hard rules:
- exactly four options
- exactly one correct option
- exactly one option must have is_correct=true
- correct_option_id must match that option
- option display position will be normalized by the workflow after revision
- correct answer must be supported by evidence
- distractors must be plausible but clearly incorrect according to evidence
- do not create generic trivia
- do not use "general" as topic
- if the evidence is insufficient, return {{"error": "INSUFFICIENT_EVIDENCE"}}
"""


def reset_reports_after_revision(
    candidate: CandidateMCQ,
    state: "CooperativeState",
) -> None:
    """Clear latest operational report pointers after creating a new candidate version."""
    clear_operational_reports(state, candidate.q_id)


def _filtered_revision_flags(flags: List[str]) -> List[str]:
    return [f for f in flags if f not in REVISION_CLEARABLE_FLAGS]


def _apply_parsed_revision(
    candidate: CandidateMCQ,
    parsed: Dict[str, Any],
    blueprint,
    state: "CooperativeState",
) -> tuple[CandidateMCQ, List[str]]:
    """Build a new candidate from parsed revision JSON; leave input unchanged."""
    changed: List[str] = []
    stem = candidate.stem
    if parsed.get("stem") and parsed["stem"] != candidate.stem:
        stem = str(parsed["stem"]).strip()
        changed.append("stem")

    options = copy.deepcopy(candidate.options)
    if parsed.get("options"):
        new_options: List[AnswerOption] = []
        source_refs = (
            _source_refs_from_context(blueprint, state) if blueprint is not None else candidate.source_refs
        )
        for raw in parsed["options"]:
            new_options.append(
                AnswerOption(
                    option_id=str(raw.get("option_id") or ""),
                    text=str(raw.get("text") or "").strip(),
                    is_correct=bool(raw.get("is_correct")),
                    explanation=raw.get("explanation"),
                    misconception=raw.get("misconception"),
                    source_refs=source_refs[:1] if raw.get("is_correct") else [],
                )
            )
        if new_options != candidate.options:
            options = new_options
            changed.append("options")

    correct_option_id = candidate.correct_option_id
    if parsed.get("correct_option_id"):
        new_correct = str(parsed["correct_option_id"]).strip()
        if new_correct and new_correct != candidate.correct_option_id:
            correct_option_id = new_correct
            changed.append("correct_option_id")

    # Prefer unique is_correct when options were rewritten.
    marked = [o for o in options if o.is_correct]
    if len(marked) == 1:
        correct_option_id = marked[0].option_id
    elif len(marked) == 0 and correct_option_id:
        for o in options:
            if o.option_id == correct_option_id:
                o.is_correct = True
                break
    elif len(marked) == 0 and not correct_option_id:
        correct_option_id = ""

    answer_explanation = candidate.answer_explanation
    if parsed.get("answer_explanation"):
        new_expl = str(parsed["answer_explanation"]).strip()
        if new_expl != candidate.answer_explanation:
            answer_explanation = new_expl
            changed.append("answer_explanation")

    source_refs = list(candidate.source_refs)
    if blueprint is not None and not source_refs:
        source_refs = _source_refs_from_context(blueprint, state)

    options, correct_option_id = assign_deterministic_option_positions(
        options,
        stable_key=candidate.q_id,
        correct_option_id=correct_option_id or None,
    )

    revised = replace(
        candidate,
        stem=stem,
        options=options,
        correct_option_id=correct_option_id,
        answer_explanation=answer_explanation,
        source_refs=source_refs,
        revision_count=candidate.revision_count + 1,
        status="draft",
        flags=_filtered_revision_flags(list(candidate.flags)),
    )
    return revised, (changed or list(parsed.get("changed_fields") or ["stem", "options"]))


def _stub_revise_parsed(
    candidate: CandidateMCQ,
    feedback: Dict[str, Any],
) -> Dict[str, Any]:
    """Deterministic revision using evidence context when no LLM is available."""
    if not feedback.get("evidence_chunks") and not feedback.get("knowledge_atoms"):
        return {"error": "INSUFFICIENT_EVIDENCE"}

    chunks = feedback.get("evidence_chunks") or []
    atoms = feedback.get("knowledge_atoms") or []
    key_text = ""
    if atoms:
        key_text = atoms[0].get("content", "")[:160]
    elif chunks:
        key_text = chunks[0].get("content", "")[:160]

    section_title = feedback.get("section_title") or "section"
    lo = feedback.get("learning_objective") or section_title
    stem = (
        f"According to the PDF evidence about {section_title}, which statement is best supported?"
    )
    correct_text = key_text or f"Evidence-supported concept from {section_title}"

    context = {
        "section_title": section_title,
        "learning_objective": lo,
        "evidence_chunks": chunks,
        "atoms": atoms,
    }
    distractors = generate_distractors_from_blueprint(context, correct_text)
    explanations = generate_explanations_from_blueprint(context, correct_text=correct_text)

    options: List[Dict[str, Any]] = [
        {
            "option_id": "TMP_CORRECT",
            "text": correct_text,
            "is_correct": True,
            "explanation": explanations.get("correct_option_explanation"),
            "misconception": None,
        },
    ]
    for d in distractors[:3]:
        options.append(d)

    while len(options) < 4:
        label = OPTION_LABELS[len(options) % len(OPTION_LABELS)]
        options.append({
            "option_id": f"TMP_D{len(options)}",
            "text": f"Unsupported claim about {section_title}.",
            "is_correct": False,
            "explanation": "Not supported by cited evidence.",
            "misconception": "Unsupported interpretation.",
        })

    return {
        "stem": stem,
        "options": options,
        "correct_option_id": "TMP_CORRECT",
        "answer_explanation": explanations.get("answer_explanation", ""),
        "changed_fields": ["stem", "options", "answer_explanation"],
        "revision_notes": ["Stub revision grounded in PDF evidence."],
    }


def _record_insufficient_revision(
    candidate: CandidateMCQ,
    state: "CooperativeState",
    *,
    previous_stem: str,
    previous_options: List[AnswerOption],
    max_revisions: int,
) -> CandidateMCQ:
    new_count = candidate.revision_count + 1
    flags = list(candidate.flags)
    if new_count >= max_revisions:
        status = "rejected"
        if "max_revisions_reached" not in flags:
            flags.append("max_revisions_reached")
    else:
        status = "needs_revision"
    revised = replace(
        candidate,
        revision_count=new_count,
        status=status,
        flags=flags,
    )
    record = RevisionRecord(
        q_id=candidate.q_id,
        revision_number=new_count,
        previous_stem=previous_stem,
        previous_options=previous_options,
        reason="INSUFFICIENT_EVIDENCE for revision",
        changed_fields=[],
        notes=["Revision could not proceed due to insufficient evidence."],
    )
    state.add_revision_record(record)
    return revised


def revise_candidate_mcq(
    candidate: CandidateMCQ,
    state: "CooperativeState",
    llm: Optional[Any] = None,
    max_revisions: int = 2,
) -> CandidateMCQ:
    """Revise one candidate using feedback from evidence/Cross-CoVe/acceptance.

    Returns a new CandidateMCQ object; the input candidate is never mutated.
    """
    if candidate.revision_count >= max_revisions:
        flags = list(candidate.flags)
        if "max_revisions_reached" not in flags:
            flags.append("max_revisions_reached")
        return replace(candidate, status="rejected", flags=flags)

    if candidate.status == "rejected" or "max_revisions_reached" in candidate.flags:
        return candidate

    feedback = collect_revision_feedback(candidate, state)
    blueprint = feedback.get("blueprint")
    previous_stem = candidate.stem
    previous_options = copy.deepcopy(candidate.options)

    prompt = build_revision_prompt(candidate, feedback)
    if llm is not None:
        raw = _invoke_llm(llm, prompt)
        parsed = _extract_json(raw)
    else:
        parsed = _stub_revise_parsed(candidate, feedback)

    if parsed is None or parsed.get("error") == "INSUFFICIENT_EVIDENCE":
        revised = _record_insufficient_revision(
            candidate,
            state,
            previous_stem=previous_stem,
            previous_options=previous_options,
            max_revisions=max_revisions,
        )
        state.store_candidate_mcq(revised)
        return revised

    revised, changed_fields = _apply_parsed_revision(candidate, parsed, blueprint, state)

    source_report_ids: List[str] = []
    if get_operational_evidence_report(state, candidate) is not None:
        source_report_ids.append(f"evidence:{candidate.q_id}:v{candidate_version_number(candidate)}")
    if get_operational_cross_cove_report(state, candidate) is not None:
        source_report_ids.append(f"cross_cove:{candidate.q_id}:v{candidate_version_number(candidate)}")

    reset_reports_after_revision(revised, state)
    validate_candidate_mcq_structure(revised)

    reason = "; ".join(feedback.get("reasons_for_revision") or [])[:500]
    record = RevisionRecord(
        q_id=candidate.q_id,
        revision_number=revised.revision_count,
        previous_stem=previous_stem,
        previous_options=previous_options,
        reason=reason or "Revision after failed verification",
        changed_fields=changed_fields,
        source_report_ids=source_report_ids,
        notes=list(parsed.get("revision_notes") or []),
    )
    state.add_revision_record(record)
    state.store_candidate_mcq(revised)
    return revised


def revise_all_needed_candidates(
    state: "CooperativeState",
    llm: Optional[Any] = None,
    max_revisions: int = 2,
    max_candidates: Optional[int] = None,
) -> List[CandidateMCQ]:
    """Revise all candidates with status needs_revision."""
    revised: List[CandidateMCQ] = []
    candidates = [
        c for c in state.candidate_mcqs.values()
        if c.status == "needs_revision"
        and "max_revisions_reached" not in c.flags
    ]
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    for candidate in candidates:
        updated = revise_candidate_mcq(
            candidate, state, llm=llm, max_revisions=max_revisions,
        )
        revised.append(updated)

    return revised


def count_candidates_needing_revision(
    state: "CooperativeState",
    max_revisions: int = 2,
) -> int:
    """Count candidates eligible for another revision attempt."""
    return sum(
        1
        for c in state.candidate_mcqs.values()
        if c.status == "needs_revision"
        and "max_revisions_reached" not in c.flags
        and c.revision_count < max_revisions
    )
