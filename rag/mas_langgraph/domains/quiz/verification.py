"""PDF evidence grounding verification for CandidateMCQs (not IWF checking)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .entities import AnswerOption, CandidateMCQ, EvidenceCheckReport, SourceRef
from .generation import log_ok, log_warn
from .versioning import (
    candidate_version_number,
    get_operational_evidence_report,
    report_targets_candidate,
)

if TYPE_CHECKING:
    from ...core.state import CooperativeState


def _normalize_title(title: str, section_id: str) -> str:
    cleaned = (title or "").strip()
    if not cleaned or cleaned.lower() == "general":
        return section_id
    return cleaned


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

    with workflow_llm_context(layer="native_mcq", component="evidence_verification"):
        response = llm.invoke(prompt)
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)


def _option_payload(options: List[AnswerOption]) -> List[Dict[str, Any]]:
    return [
        {
            "option_id": o.option_id,
            "text": o.text,
            "is_correct": o.is_correct,
            "explanation": o.explanation,
            "misconception": o.misconception,
        }
        for o in options
    ]


def build_evidence_verification_context(
    candidate: CandidateMCQ,
    state: "CooperativeState",
) -> Dict[str, Any]:
    """Collect verification context from candidate + blueprint — no new retrieval."""
    blueprint = state.mcq_blueprints.get(candidate.blueprint_id)
    section_title = ""
    learning_objective = ""
    evidence_chunks: List[Dict[str, Any]] = []
    knowledge_atoms: List[Dict[str, Any]] = []

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
        for chunk_id in blueprint.evidence_chunk_ids:
            chunk = state.get_chunk(chunk_id)
            if chunk is None:
                continue
            meta = getattr(chunk, "metadata", {}) or {}
            evidence_chunks.append({
                "chunk_id": chunk_id,
                "content": chunk.content,
                "page_number": meta.get("page_number"),
                "source_file": meta.get("source_file"),
            })
        for atom_id in blueprint.target_atom_ids:
            atom = state.get_atom(atom_id)
            if atom is None:
                continue
            knowledge_atoms.append({
                "atom_id": atom_id,
                "content": atom.content,
                "atom_type": atom.atom_type,
            })

    correct_option = next(
        (o for o in candidate.options if o.option_id == candidate.correct_option_id),
        None,
    )
    distractor_explanations = [
        {
            "option_id": o.option_id,
            "text": o.text,
            "explanation": o.explanation,
            "misconception": o.misconception,
        }
        for o in candidate.options
        if not o.is_correct
    ]

    return {
        "q_id": candidate.q_id,
        "blueprint_id": candidate.blueprint_id,
        "section_title": section_title,
        "learning_objective": learning_objective,
        "stem": candidate.stem,
        "options": _option_payload(candidate.options),
        "correct_option_id": candidate.correct_option_id,
        "correct_option_text": correct_option.text if correct_option else "",
        "answer_explanation": candidate.answer_explanation,
        "distractor_explanations": distractor_explanations,
        "evidence_chunks": evidence_chunks,
        "knowledge_atoms": knowledge_atoms,
        "source_refs": [asdict(r) for r in candidate.source_refs],
        "blueprint_found": blueprint is not None,
        "blueprint_evidence_chunk_ids": list(blueprint.evidence_chunk_ids) if blueprint else [],
    }


def _run_prechecks(
    candidate: CandidateMCQ,
    context: Dict[str, Any],
) -> Optional[EvidenceCheckReport]:
    """Deterministic pre-checks — return report if verification cannot proceed."""
    required_fixes: List[str] = []
    notes: List[str] = []

    if not context.get("blueprint_found"):
        required_fixes.append("missing_blueprint")
    if not candidate.source_refs:
        required_fixes.append("missing_source_refs")
    if not context.get("evidence_chunks"):
        required_fixes.append("missing_evidence")
    if not context.get("blueprint_evidence_chunk_ids"):
        required_fixes.append("missing_evidence")

    correct_count = sum(1 for o in candidate.options if o.is_correct)
    if candidate.correct_option_id not in {o.option_id for o in candidate.options}:
        required_fixes.append("missing_correct_option_id")
    if correct_count != 1:
        required_fixes.append("invalid_correct_option_count")
    if not (candidate.answer_explanation or "").strip():
        required_fixes.append("missing_answer_explanation")

    if not required_fixes:
        return None

    notes.append("Precheck failed; LLM verification skipped.")
    return EvidenceCheckReport(
        q_id=candidate.q_id,
        supported_by_pdf=False,
        unsupported_claims=["Precheck failures prevent evidence grounding."],
        source_refs=list(candidate.source_refs),
        confidence=0.0,
        candidate_version=candidate_version_number(candidate),
        answer_supported=False,
        stem_supported=False,
        distractors_valid=False,
        explanation_supported=False,
        required_fixes=sorted(set(required_fixes)),
        notes=notes,
    )


def _build_verification_prompt(context: Dict[str, Any]) -> str:
    evidence_block = json.dumps(context.get("evidence_chunks") or [], indent=2, ensure_ascii=False)
    atoms_block = json.dumps(context.get("knowledge_atoms") or [], indent=2, ensure_ascii=False)
    options_block = json.dumps(context.get("options") or [], indent=2, ensure_ascii=False)
    return f"""You are verifying whether an MCQ is grounded in the provided PDF evidence.
Use only the provided evidence.
Do not use outside knowledge.
Do not judge item-writing quality.
Do not judge whether the question is pedagogically elegant.
Only judge whether the question, correct answer, distractors, and explanations are supported by or consistent with the evidence.

Section title: {context.get("section_title")}
Learning objective: {context.get("learning_objective")}

Stem:
{context.get("stem")}

Options:
{options_block}

Correct option id: {context.get("correct_option_id")}
Answer explanation: {context.get("answer_explanation")}

Distractor explanations:
{json.dumps(context.get("distractor_explanations") or [], indent=2, ensure_ascii=False)}

Evidence chunks:
{evidence_block}

Knowledge atoms:
{atoms_block}

Return JSON only:
{{
  "supported_by_pdf": true,
  "stem_supported": true,
  "answer_supported": true,
  "distractors_valid": true,
  "explanation_supported": true,
  "unsupported_claims": [],
  "required_fixes": [],
  "confidence": 0.0,
  "notes": []
}}

Definitions:
- stem_supported: the question asks about something present in the evidence.
- answer_supported: the marked correct answer is supported by the evidence.
- distractors_valid: the distractors are not also correct according to the evidence.
- explanation_supported: the answer explanation is based on the evidence.
- supported_by_pdf: true only if stem_supported, answer_supported, distractors_valid, and explanation_supported are all true.
"""


def _stub_verify(context: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic evidence check using keyword overlap with chunk/atom text."""
    corpus = " ".join(
        [c.get("content", "") for c in context.get("evidence_chunks") or []]
        + [a.get("content", "") for a in context.get("knowledge_atoms") or []]
    ).lower()
    if not corpus.strip():
        return {
            "supported_by_pdf": False,
            "stem_supported": False,
            "answer_supported": False,
            "distractors_valid": False,
            "explanation_supported": False,
            "unsupported_claims": ["No evidence text available for grounding."],
            "required_fixes": ["missing_evidence"],
            "confidence": 0.0,
            "notes": ["Stub verifier: no evidence corpus."],
        }

    stem = (context.get("stem") or "").lower()
    answer = (context.get("correct_option_text") or "").lower()
    explanation = (context.get("answer_explanation") or "").lower()

    def _supported(text: str) -> bool:
        if not text.strip():
            return False
        tokens = [t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 4]
        if not tokens:
            return bool(text.strip())
        hits = sum(1 for t in tokens if t in corpus)
        return hits >= max(1, len(tokens) // 4)

    stem_supported = _supported(stem)
    answer_supported = _supported(answer)
    explanation_supported = _supported(explanation) or answer_supported
    distractors_valid = True
    unsupported: List[str] = []
    if not stem_supported:
        unsupported.append("Stem not clearly grounded in provided evidence.")
    if not answer_supported:
        unsupported.append("Correct answer not clearly grounded in provided evidence.")
    if not explanation_supported:
        unsupported.append("Answer explanation not clearly grounded in provided evidence.")

    supported_by_pdf = (
        stem_supported and answer_supported and distractors_valid and explanation_supported
    )
    confidence = 0.75 if supported_by_pdf else 0.35
    return {
        "supported_by_pdf": supported_by_pdf,
        "stem_supported": stem_supported,
        "answer_supported": answer_supported,
        "distractors_valid": distractors_valid,
        "explanation_supported": explanation_supported,
        "unsupported_claims": unsupported,
        "required_fixes": [] if supported_by_pdf else ["evidence_grounding_failed"],
        "confidence": confidence,
        "notes": ["Stub keyword overlap verifier."],
    }


def validate_evidence_report_structure(report: EvidenceCheckReport) -> EvidenceCheckReport:
    """Ensure report fields are internally consistent."""
    all_true = (
        report.stem_supported
        and report.answer_supported
        and report.distractors_valid
        and report.explanation_supported
    )
    if report.supported_by_pdf and not all_true:
        report.supported_by_pdf = False
        if "inconsistent_support_flags" not in report.required_fixes:
            report.required_fixes.append("inconsistent_support_flags")
    if all_true and not report.supported_by_pdf:
        report.supported_by_pdf = True
    report.confidence = max(0.0, min(1.0, float(report.confidence or 0.0)))
    return report


def _report_from_parsed(
    candidate: CandidateMCQ,
    parsed: Dict[str, Any],
) -> EvidenceCheckReport:
    stem_supported = bool(parsed.get("stem_supported"))
    answer_supported = bool(parsed.get("answer_supported"))
    distractors_valid = bool(parsed.get("distractors_valid"))
    explanation_supported = bool(parsed.get("explanation_supported"))
    supported_by_pdf = bool(parsed.get("supported_by_pdf"))
    if supported_by_pdf:
        supported_by_pdf = (
            stem_supported and answer_supported and distractors_valid and explanation_supported
        )
    report = EvidenceCheckReport(
        q_id=candidate.q_id,
        supported_by_pdf=supported_by_pdf,
        unsupported_claims=list(parsed.get("unsupported_claims") or []),
        source_refs=list(candidate.source_refs),
        confidence=float(parsed.get("confidence") or 0.0),
        candidate_version=candidate_version_number(candidate),
        answer_supported=answer_supported,
        stem_supported=stem_supported,
        distractors_valid=distractors_valid,
        explanation_supported=explanation_supported,
        required_fixes=list(parsed.get("required_fixes") or []),
        notes=list(parsed.get("notes") or []),
    )
    return validate_evidence_report_structure(report)


def apply_candidate_status_from_report(
    candidate: CandidateMCQ,
    report: EvidenceCheckReport,
) -> None:
    """Update candidate status and flags from evidence report — no permanent rejection."""
    if report.supported_by_pdf:
        candidate.status = "evidence_verified"
        return

    candidate.status = "needs_revision"
    flag_map = {
        "evidence_not_supported": not report.supported_by_pdf,
        "unsupported_claims": bool(report.unsupported_claims),
        "missing_source_refs": "missing_source_refs" in report.required_fixes,
        "answer_not_supported": not report.answer_supported,
        "distractors_not_valid": not report.distractors_valid,
        "explanation_not_supported": not report.explanation_supported,
    }
    for flag, active in flag_map.items():
        if active and flag not in candidate.flags:
            candidate.flags.append(flag)


def verify_candidate_against_evidence(
    candidate: CandidateMCQ,
    state: "CooperativeState",
    llm: Optional[Any] = None,
) -> EvidenceCheckReport:
    """Verify one CandidateMCQ against attached PDF evidence only."""
    context = build_evidence_verification_context(candidate, state)
    precheck = _run_prechecks(candidate, context)
    if precheck is not None:
        apply_candidate_status_from_report(candidate, precheck)
        return validate_evidence_report_structure(precheck)

    if llm is not None:
        raw = _invoke_llm(llm, _build_verification_prompt(context))
        parsed = _extract_json(raw)
        if parsed is None:
            report = EvidenceCheckReport(
                q_id=candidate.q_id,
                supported_by_pdf=False,
                unsupported_claims=["LLM verification response could not be parsed."],
                source_refs=list(candidate.source_refs),
                confidence=0.0,
                candidate_version=candidate_version_number(candidate),
                required_fixes=["llm_parse_error"],
                notes=["Verifier LLM parse failed."],
            )
        else:
            report = _report_from_parsed(candidate, parsed)
    else:
        report = _report_from_parsed(candidate, _stub_verify(context))

    apply_candidate_status_from_report(candidate, report)
    return report


def verify_all_candidate_mcqs(
    state: "CooperativeState",
    llm: Optional[Any] = None,
    max_candidates: Optional[int] = None,
    *,
    force: bool = False,
) -> List[EvidenceCheckReport]:
    """Verify all candidate MCQs and store EvidenceCheckReports on state."""
    reports: List[EvidenceCheckReport] = []
    candidates = list(state.candidate_mcqs.values())
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    for candidate in candidates:
        existing = state.evidence_reports.get(candidate.q_id)
        if not force and existing is not None and report_targets_candidate(existing, candidate):
            reports.append(existing)
            continue
        report = verify_candidate_against_evidence(candidate, state, llm=llm)
        state.store_evidence_report(report)
        state.store_candidate_mcq(candidate)
        reports.append(report)

    return reports
