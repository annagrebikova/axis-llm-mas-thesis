"""Deterministic human-readable labels and display previews.

No LLM, no paraphrasing, no summarization. Truncation is display-only.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence, Tuple


PREVIEW_MAX_CHARS = 110
ELLIPSIS = "..."

NODE_LABELS = {
    "build_outline": "Build outline",
    "generate_learning_objectives": "Create learning objectives",
    "initial_retrieve_evidence_for_all_sections": "Retrieve teaching evidence",
    "retrieve_evidence": "Retrieve teaching evidence",
    "deduplicate_evidence": "Deduplicate teaching evidence",
    "curate_evidence": "Curate teaching evidence",
    "atomize_evidence": "Extract knowledge statements",
    "atomize_new_evidence": "Extract new knowledge statements",
    "update_section_evidence_status": "Update evidence status",
    "repair_missing_evidence": "Repair missing evidence",
    "plan_mcq_blueprints": "Plan question",
    "generate_candidate_mcqs_from_blueprints": "Generate question",
    "verify_candidate_mcqs_against_evidence": "Verify against teaching material",
    "cross_cove_verify_candidate_mcqs": "Independent answer check",
    "apply_acceptance_gate": "Evaluate acceptance",
    "revise_candidate_mcqs": "Revise question",
    "check_coverage": "Check coverage",
    "generate_repair_blueprints": "Plan repair questions",
    "package_final_quiz": "Package quiz",
    "verify_candidates": "Verify candidates",
    "package_or_end": "Package or end",
    "end_with_error": "End with error",
}

ARTIFACT_LABELS = {
    "DocSection": "Document section",
    "LearningObjective": "Learning objective",
    "EvidenceChunk": "Teaching evidence",
    "KnowledgeAtom": "Knowledge statement",
    "MCQBlueprint": "Question plan",
    "CandidateMCQ": "Question",
    "EvidenceCheckReport": "Evidence verification",
    "CrossCoVeReport": "Independent answer check",
    "SupervisorDecision": "Acceptance decision",
    "OutcomeAssessment": "Outcome criteria",
}

LANE_LABELS = {
    "artifacts": "Artifacts",
    "factual": "Factual execution",
    "noop": "NOOP control",
    "counterfactual": "Counterfactual",
    "comparison": "Comparison",
}

# Display labels for exported information-role findings. Keys match
# experiment.information_role role_type / causal_classification strings.
INFORMATION_ROLE_LABELS = {
    "focus_shaping": "▲ Focus-shaping",
    "answer_anchoring": "▲ Answer-anchoring",
    "distractor_shaping": "▲ Distractor-shaping",
    "formulation_shaping": "▲ Formulation-shaping",
    "necessary_for_reproduction": "▲ Necessary for reproduction",
    "influential_non_decisive": "▲ Influential, non-decisive",
    "robust_to_tested_information": "○ Robust to tested information",
    "no_demonstrated_role": "○ No demonstrated effect",
    "no_demonstrated_effect": "○ No demonstrated effect",
    "inconclusive": "? Inconclusive",
    "artifact_level_only": "○ Artifact-level only",
    "artifact_level_downstream": "○ Artifact-level downstream",
    "decisive": "▲ Decisive",
}

SYMBOL_UNCHANGED = "="
SYMBOL_CHANGED = "Δ"
SYMBOL_INCONCLUSIVE = "?"
SYMBOL_NOOP = "Ø"

ESSENTIAL_OCCURRENCE_NODE_IDS = frozenset(
    {
        "generate_learning_objectives",
        "atomize_evidence",
        "atomize_new_evidence",
        "plan_mcq_blueprints",
        "generate_candidate_mcqs_from_blueprints",
        "verify_candidate_mcqs_against_evidence",
        "cross_cove_verify_candidate_mcqs",
        "apply_acceptance_gate",
    }
)

_WS_RE = re.compile(r"\s+")
_SNAKE_RE = re.compile(r"_+")


def normalize_whitespace(text: Any) -> str:
    if text is None:
        return ""
    return _WS_RE.sub(" ", str(text)).strip()


def make_preview(text: Any, max_chars: int = PREVIEW_MAX_CHARS) -> str:
    """Deterministic display truncation. Full text belongs in node.details."""
    normalized = normalize_whitespace(text)
    if max_chars <= 0:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= len(ELLIPSIS):
        return ELLIPSIS[:max_chars]
    return normalized[: max_chars - len(ELLIPSIS)].rstrip() + ELLIPSIS


def sentence_case_identifier(identifier: str) -> str:
    raw = str(identifier or "").strip()
    if not raw:
        return "Unknown"
    if raw in NODE_LABELS:
        return NODE_LABELS[raw]
    if raw in ARTIFACT_LABELS:
        return ARTIFACT_LABELS[raw]
    spaced = _SNAKE_RE.sub(" ", raw).strip()
    if not spaced:
        return "Unknown"
    return spaced[0].upper() + spaced[1:]


def occurrence_label(node_id: str) -> str:
    return NODE_LABELS.get(str(node_id or ""), sentence_case_identifier(node_id))


def artifact_label(artifact_type: str) -> str:
    return ARTIFACT_LABELS.get(
        str(artifact_type or ""), sentence_case_identifier(artifact_type)
    )


def lane_label(lane_id: str) -> str:
    return LANE_LABELS.get(str(lane_id or ""), sentence_case_identifier(lane_id))


def effect_symbol(changed: Optional[bool]) -> str:
    if changed is True:
        return SYMBOL_CHANGED
    if changed is False:
        return SYMBOL_UNCHANGED
    return SYMBOL_INCONCLUSIVE


def effect_word(changed: Optional[bool]) -> str:
    if changed is True:
        return "changed"
    if changed is False:
        return "unchanged"
    return "unavailable"


def information_role_label(role: Optional[str]) -> str:
    if not role:
        return ""
    key = str(role)
    if key in INFORMATION_ROLE_LABELS:
        return INFORMATION_ROLE_LABELS[key]
    return f"{SYMBOL_INCONCLUSIVE} {sentence_case_identifier(key)}"


def _first_text(*values: Any) -> str:
    for value in values:
        text = normalize_whitespace(value)
        if text:
            return text
    return ""


def artifact_preview(artifact_type: str, content: Mapping[str, Any]) -> str:
    data = dict(content or {})
    kind = str(artifact_type or "")
    if kind == "LearningObjective":
        return make_preview(data.get("objective_text"))
    if kind == "EvidenceChunk":
        return make_preview(data.get("content"))
    if kind == "KnowledgeAtom":
        return make_preview(data.get("content"))
    if kind == "MCQBlueprint":
        goal = make_preview(data.get("generation_goal"))
        meta_parts = [
            str(data[k])
            for k in ("difficulty", "bloom_level", "question_type")
            if data.get(k)
        ]
        meta = make_preview(" · ".join(meta_parts), max_chars=60) if meta_parts else ""
        if goal and meta:
            return make_preview(f"{goal} ({meta})", max_chars=PREVIEW_MAX_CHARS)
        return goal or meta
    if kind == "CandidateMCQ":
        return make_preview(data.get("stem"))
    if kind == "EvidenceCheckReport":
        supported = data.get("supported_by_pdf")
        status = "supported" if supported else "unsupported"
        conf = data.get("confidence")
        if conf is None:
            return status
        return make_preview(f"{status} · confidence {conf}")
    if kind == "CrossCoVeReport":
        majority = data.get("majority_answer")
        passed = data.get("passed")
        conf = data.get("confidence")
        bits = []
        if majority not in (None, ""):
            bits.append(f"majority {majority}")
        bits.append("passed" if passed else "failed")
        if conf is not None:
            bits.append(f"confidence {conf}")
        return make_preview(" · ".join(bits))
    if kind == "SupervisorDecision":
        return make_preview(data.get("decision"))
    if kind == "OutcomeAssessment":
        return make_preview(outcome_assessment_preview(data))
    if kind == "DocSection":
        return make_preview(_first_text(data.get("title"), data.get("content_summary")))
    return make_preview(_first_text(data.get("content"), data.get("stem"), kind))


def outcome_assessment_preview(content: Mapping[str, Any]) -> str:
    criteria = list(content.get("criteria") or [])
    passed = 0
    failed = 0
    other = 0
    for item in criteria:
        if not isinstance(item, Mapping):
            other += 1
            continue
        value = item.get("value")
        if value is True:
            passed += 1
        elif value is False:
            failed += 1
        else:
            other += 1
    total = len(criteria)
    if total == 0:
        return "no criteria"
    if other and not passed and not failed:
        return f"{total} criteria"
    return f"{passed} passed / {failed} failed"


def format_decision_content_line(
    *,
    decision_changed: Optional[bool],
    question_content_changed: Optional[bool],
) -> str:
    return (
        f"Question content: {effect_symbol(question_content_changed)} "
        f"{effect_word(question_content_changed)}; "
        f"Acceptance: {effect_symbol(decision_changed)} "
        f"{effect_word(decision_changed)}"
    )


def strip_trace_mcq_content(content: Mapping[str, Any]) -> Mapping[str, Any]:
    """Question content allowed in Trace; omit source/provenance attachments."""
    data = dict(content or {})
    options = []
    for opt in data.get("options") or []:
        if not isinstance(opt, Mapping):
            continue
        options.append(
            {
                "option_id": opt.get("option_id"),
                "text": opt.get("text"),
                "is_correct": opt.get("is_correct"),
                "explanation": opt.get("explanation"),
                "misconception": opt.get("misconception"),
            }
        )
    kept = {
        "q_id": data.get("q_id"),
        "blueprint_id": data.get("blueprint_id"),
        "stem": data.get("stem"),
        "options": options,
        "correct_option_id": data.get("correct_option_id"),
        "answer_explanation": data.get("answer_explanation"),
        "status": data.get("status"),
        "revision_count": data.get("revision_count"),
        "flags": list(data.get("flags") or []),
    }
    return kept
