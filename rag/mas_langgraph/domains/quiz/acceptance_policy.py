"""Single authoritative MCQ accept/revise/reject policy (delta_alpha).

Both ``apply_acceptance_gate`` and ``McqOutcomeEvaluator`` must call
``decide_acceptance`` so factual gate semantics cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ...artifacts.models import ArtifactRef
from ...core.outcomes import CriterionValue, DecisionArtifact, OutcomeAssessment

HARD_REJECTION_FLAGS = frozenset({
    "insufficient_atoms",
    "insufficient_evidence",
    "missing_source_refs",
    "no_correct_option",
    "multiple_correct_options",
    "too_few_options",
    "evidence_not_supported",
    "answer_not_supported",
    "distractors_not_valid",
    "cross_cove_failed",
    "cross_cove_no_majority",
    "max_revisions_reached",
})

STRUCTURAL_REJECT_FLAGS = frozenset({
    "no_correct_option",
    "multiple_correct_options",
    "too_few_options",
})

# Deterministic Gamma order when YAML order is unavailable.
DEFAULT_CRITERION_ORDER: Tuple[str, ...] = (
    "revision_count",
    "max_revisions",
    "candidate_status",
    "max_revisions_reached_flag",
    "evidence_report_available",
    "cross_cove_required",
    "cross_cove_report_available",
    "hard_blocking_flags",
    "structural_reject_flags",
    "pdf_supported",
    "cross_cove_passed",
    "cross_cove_majority_present",
    "source_refs_present",
    "correct_option_count",
    "blueprint_available",
    "target_atom_count",
    "blueprint_evidence_count",
    "answer_supported",
    "stem_supported",
    "distractors_valid",
    "explanation_supported",
    "evidence_confidence",
)

DECISION_RELEVANT_DEFAULTS: Dict[str, bool] = {
    "revision_count": True,
    "max_revisions": True,
    "candidate_status": True,
    "max_revisions_reached_flag": True,
    "evidence_report_available": True,
    "cross_cove_required": True,
    "cross_cove_report_available": True,
    "hard_blocking_flags": True,
    "structural_reject_flags": True,
    "pdf_supported": True,
    "cross_cove_passed": True,
    "cross_cove_majority_present": True,
    "source_refs_present": True,
    "correct_option_count": True,
    "blueprint_available": True,
    "target_atom_count": True,
    "blueprint_evidence_count": True,
    "answer_supported": False,
    "stem_supported": False,
    "distractors_valid": False,
    "explanation_supported": False,
    "evidence_confidence": False,
}

VALUE_TYPE_DEFAULTS: Dict[str, str] = {
    "revision_count": "int",
    "max_revisions": "int",
    "candidate_status": "str",
    "max_revisions_reached_flag": "bool",
    "evidence_report_available": "bool",
    "cross_cove_required": "bool",
    "cross_cove_report_available": "bool",
    "hard_blocking_flags": "tuple[str, ...]",
    "structural_reject_flags": "tuple[str, ...]",
    "pdf_supported": "bool",
    "cross_cove_passed": "bool",
    "cross_cove_majority_present": "bool",
    "source_refs_present": "bool",
    "correct_option_count": "int",
    "blueprint_available": "bool",
    "target_atom_count": "int",
    "blueprint_evidence_count": "int",
    "answer_supported": "bool",
    "stem_supported": "bool",
    "distractors_valid": "bool",
    "explanation_supported": "bool",
    "evidence_confidence": "float",
}


@dataclass(frozen=True)
class GateInputs:
    """All factual inputs required to reproduce the acceptance gate."""

    revision_count: int
    max_revisions: int
    candidate_status: str
    flags: Tuple[str, ...]
    evidence_report_available: bool
    cross_cove_required: bool
    cross_cove_report_available: bool
    pdf_supported: bool
    cross_cove_passed: bool
    cross_cove_majority_present: bool
    source_refs_present: bool
    correct_option_count: int
    blueprint_available: bool
    target_atom_count: int
    blueprint_evidence_count: int
    answer_supported: bool = False
    stem_supported: bool = False
    distractors_valid: bool = False
    explanation_supported: bool = False
    evidence_confidence: float = 0.0


@dataclass(frozen=True)
class GateDecision:
    decision: str
    reasons: Tuple[str, ...]
    required_fixes: Tuple[str, ...]
    append_max_revisions_flag: bool = False


def status_for_decision(decision: str) -> str:
    if decision == "accept":
        return "accepted"
    if decision == "reject":
        return "rejected"
    return "needs_revision"


def assessment_ref_for(target_ref: ArtifactRef) -> ArtifactRef:
    return ArtifactRef(
        logical_id=f"assessment:{target_ref.logical_id}",
        version=target_ref.version,
    )


def decision_ref_for(target_ref: ArtifactRef) -> ArtifactRef:
    return ArtifactRef(
        logical_id=f"decision:{target_ref.logical_id}",
        version=target_ref.version,
    )


def build_gate_inputs(
    *,
    revision_count: int,
    max_revisions: int,
    candidate_status: str,
    flags: Sequence[str],
    evidence_report: Any = None,
    cross_cove_report: Any = None,
    cross_cove_required: bool = True,
    source_refs_present: bool = False,
    correct_option_count: int = 0,
    blueprint: Any = None,
) -> GateInputs:
    """Normalize domain objects into gate inputs."""
    flag_tuple = tuple(flags or ())
    evidence_available = evidence_report is not None
    cove_available = cross_cove_report is not None
    pdf_supported = bool(getattr(evidence_report, "supported_by_pdf", False)) if evidence_available else False
    answer_supported = bool(getattr(evidence_report, "answer_supported", False)) if evidence_available else False
    stem_supported = bool(getattr(evidence_report, "stem_supported", False)) if evidence_available else False
    distractors_valid = bool(getattr(evidence_report, "distractors_valid", False)) if evidence_available else False
    explanation_supported = (
        bool(getattr(evidence_report, "explanation_supported", False)) if evidence_available else False
    )
    evidence_confidence = float(getattr(evidence_report, "confidence", 0.0) or 0.0) if evidence_available else 0.0
    cross_cove_passed = bool(getattr(cross_cove_report, "passed", False)) if cove_available else False
    majority = getattr(cross_cove_report, "majority_answer", None) if cove_available else None
    cross_cove_majority_present = bool(majority) if cove_available else False
    blueprint_available = blueprint is not None
    target_atom_count = len(getattr(blueprint, "target_atom_ids", None) or []) if blueprint_available else 0
    blueprint_evidence_count = (
        len(getattr(blueprint, "evidence_chunk_ids", None) or []) if blueprint_available else 0
    )
    return GateInputs(
        revision_count=int(revision_count or 0),
        max_revisions=int(max_revisions),
        candidate_status=str(candidate_status or "draft"),
        flags=flag_tuple,
        evidence_report_available=evidence_available,
        cross_cove_required=bool(cross_cove_required),
        cross_cove_report_available=cove_available,
        pdf_supported=pdf_supported,
        cross_cove_passed=cross_cove_passed,
        cross_cove_majority_present=cross_cove_majority_present,
        source_refs_present=bool(source_refs_present),
        correct_option_count=int(correct_option_count),
        blueprint_available=blueprint_available,
        target_atom_count=target_atom_count,
        blueprint_evidence_count=blueprint_evidence_count,
        answer_supported=answer_supported,
        stem_supported=stem_supported,
        distractors_valid=distractors_valid,
        explanation_supported=explanation_supported,
        evidence_confidence=evidence_confidence,
    )


def decide_acceptance(inputs: GateInputs) -> GateDecision:
    """Executable delta_alpha — preserves factual apply_acceptance_gate order."""
    reasons: List[str] = []
    required_fixes: List[str] = []
    hard_flags = [f for f in inputs.flags if f in HARD_REJECTION_FLAGS]
    append_max = False

    if "max_revisions_reached" in inputs.flags or (
        inputs.revision_count >= inputs.max_revisions
        and inputs.candidate_status == "needs_revision"
    ):
        decision = "reject"
        reasons.append("Max revisions reached; candidate still fails verification.")
        required_fixes.append("max_revisions_reached")
    elif not inputs.evidence_report_available:
        reasons.append("No evidence verification report available.")
        required_fixes.append("needs_evidence_verification")
        decision = "revise"
    elif inputs.cross_cove_required and not inputs.cross_cove_report_available:
        reasons.append("No Cross-CoVe verification report available.")
        required_fixes.append("needs_cross_cove_verification")
        decision = "revise"
    elif hard_flags:
        if any(f in STRUCTURAL_REJECT_FLAGS for f in hard_flags):
            decision = "reject"
            reasons.append(
                f"Candidate has hard blocking flag(s): {', '.join(hard_flags)}."
            )
        else:
            decision = "revise"
            reasons.append(
                f"Candidate has hard blocking flag(s): {', '.join(hard_flags)}."
            )
        required_fixes.extend(hard_flags)
    elif not inputs.pdf_supported:
        decision = "revise"
        reasons.append("Evidence report says content is not supported by PDF.")
        if not inputs.answer_supported:
            reasons.append("Evidence report says answer is not supported by PDF.")
        required_fixes.append("evidence_not_supported")
    elif inputs.cross_cove_required and inputs.cross_cove_report_available and not inputs.cross_cove_passed:
        decision = "revise"
        reasons.append("Cross-CoVe majority answer does not match generator answer.")
        required_fixes.append("cross_cove_failed")
    elif (
        inputs.cross_cove_required
        and inputs.cross_cove_report_available
        and not inputs.cross_cove_majority_present
    ):
        decision = "revise"
        reasons.append("Cross-CoVe produced no majority answer.")
        required_fixes.append("cross_cove_no_majority")
    elif not inputs.source_refs_present:
        decision = "revise"
        reasons.append("Candidate has no source references.")
        required_fixes.append("missing_source_refs")
    elif inputs.correct_option_count != 1:
        decision = "reject"
        reasons.append(
            f"Candidate has {inputs.correct_option_count} correct option(s); expected exactly one."
        )
        required_fixes.append("invalid_correct_option_count")
    elif not inputs.blueprint_available:
        decision = "revise"
        reasons.append("Blueprint not found for candidate.")
        required_fixes.append("missing_blueprint")
    elif inputs.target_atom_count <= 0:
        decision = "revise"
        reasons.append("Candidate has no related atoms through blueprint.target_atom_ids.")
        required_fixes.append("insufficient_atoms")
    elif inputs.blueprint_evidence_count <= 0:
        decision = "revise"
        reasons.append("Blueprint has no attached evidence chunks.")
        required_fixes.append("insufficient_evidence")
    else:
        decision = "accept"
        reasons.append("Candidate passed structural, evidence, and Cross-CoVe checks.")

    if decision == "revise" and inputs.revision_count >= inputs.max_revisions:
        decision = "reject"
        reasons.append("Max revisions reached.")
        if "max_revisions_reached" not in required_fixes:
            required_fixes.append("max_revisions_reached")
        append_max = True

    return GateDecision(
        decision=decision,
        reasons=tuple(reasons),
        required_fixes=tuple(sorted(set(required_fixes))),
        append_max_revisions_flag=append_max,
    )


def gate_inputs_from_criteria(criteria: Sequence[CriterionValue]) -> GateInputs:
    by_name = {c.name: c.value for c in criteria}
    hard = tuple(by_name.get("hard_blocking_flags") or ())
    flags = list(hard)
    if by_name.get("max_revisions_reached_flag"):
        if "max_revisions_reached" not in flags:
            flags.append("max_revisions_reached")
    return GateInputs(
        revision_count=int(by_name.get("revision_count", 0) or 0),
        max_revisions=int(by_name.get("max_revisions", 2) or 2),
        candidate_status=str(by_name.get("candidate_status", "draft")),
        flags=tuple(flags),
        evidence_report_available=bool(by_name.get("evidence_report_available", False)),
        cross_cove_required=bool(by_name.get("cross_cove_required", True)),
        cross_cove_report_available=bool(by_name.get("cross_cove_report_available", False)),
        pdf_supported=bool(by_name.get("pdf_supported", False)),
        cross_cove_passed=bool(by_name.get("cross_cove_passed", False)),
        cross_cove_majority_present=bool(by_name.get("cross_cove_majority_present", False)),
        source_refs_present=bool(by_name.get("source_refs_present", False)),
        correct_option_count=int(by_name.get("correct_option_count", 0) or 0),
        blueprint_available=bool(by_name.get("blueprint_available", False)),
        target_atom_count=int(by_name.get("target_atom_count", 0) or 0),
        blueprint_evidence_count=int(by_name.get("blueprint_evidence_count", 0) or 0),
        answer_supported=bool(by_name.get("answer_supported", False)),
        stem_supported=bool(by_name.get("stem_supported", False)),
        distractors_valid=bool(by_name.get("distractors_valid", False)),
        explanation_supported=bool(by_name.get("explanation_supported", False)),
        evidence_confidence=float(by_name.get("evidence_confidence", 0.0) or 0.0),
    )


def criteria_from_gate_inputs(
    inputs: GateInputs,
    *,
    criteria_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    criteria_order: Optional[Sequence[str]] = None,
) -> Tuple[CriterionValue, ...]:
    """Build deterministic Gamma_alpha from gate inputs."""
    meta = criteria_meta or {}
    order = list(criteria_order) if criteria_order else list(DEFAULT_CRITERION_ORDER)
    hard_flags = tuple(f for f in inputs.flags if f in HARD_REJECTION_FLAGS)
    structural = tuple(f for f in hard_flags if f in STRUCTURAL_REJECT_FLAGS)
    values: Dict[str, Any] = {
        "revision_count": inputs.revision_count,
        "max_revisions": inputs.max_revisions,
        "candidate_status": inputs.candidate_status,
        "max_revisions_reached_flag": "max_revisions_reached" in inputs.flags,
        "evidence_report_available": inputs.evidence_report_available,
        "cross_cove_required": inputs.cross_cove_required,
        "cross_cove_report_available": inputs.cross_cove_report_available,
        "hard_blocking_flags": hard_flags,
        "structural_reject_flags": structural,
        "pdf_supported": inputs.pdf_supported,
        "cross_cove_passed": inputs.cross_cove_passed,
        "cross_cove_majority_present": inputs.cross_cove_majority_present,
        "source_refs_present": inputs.source_refs_present,
        "correct_option_count": inputs.correct_option_count,
        "blueprint_available": inputs.blueprint_available,
        "target_atom_count": inputs.target_atom_count,
        "blueprint_evidence_count": inputs.blueprint_evidence_count,
        "answer_supported": inputs.answer_supported,
        "stem_supported": inputs.stem_supported,
        "distractors_valid": inputs.distractors_valid,
        "explanation_supported": inputs.explanation_supported,
        "evidence_confidence": inputs.evidence_confidence,
    }
    out: List[CriterionValue] = []
    for name in order:
        if name not in values:
            # Skip unimplemented declarative entries (e.g. required_tool_success).
            item = meta.get(name) or {}
            if item.get("implemented") is False:
                continue
            continue
        item = meta.get(name) or {}
        out.append(
            CriterionValue(
                name=name,
                value=values[name],
                value_type=str(item.get("value_type") or VALUE_TYPE_DEFAULTS.get(name, "any")),
                label=str(item.get("label") or name),
                explanation=str(item.get("explanation") or ""),
                decision_relevant=bool(
                    item.get(
                        "decision_relevant",
                        DECISION_RELEVANT_DEFAULTS.get(name, True),
                    )
                ),
            )
        )
    return tuple(out)


def build_outcome_assessment(
    target_ref: ArtifactRef,
    inputs: GateInputs,
    *,
    criteria_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    criteria_order: Optional[Sequence[str]] = None,
    notes: Iterable[str] = (),
) -> OutcomeAssessment:
    criteria = criteria_from_gate_inputs(
        inputs, criteria_meta=criteria_meta, criteria_order=criteria_order
    )
    return OutcomeAssessment(
        assessment_ref=assessment_ref_for(target_ref),
        target_ref=target_ref,
        criteria=criteria,
        notes=tuple(notes),
    )


def build_decision_artifact(
    assessment: OutcomeAssessment,
    gate: GateDecision,
) -> DecisionArtifact:
    return DecisionArtifact(
        decision_ref=decision_ref_for(assessment.target_ref),
        target_ref=assessment.target_ref,
        assessment_ref=assessment.assessment_ref,
        decision=gate.decision,
        reasons=gate.reasons,
        required_fixes=gate.required_fixes,
    )
