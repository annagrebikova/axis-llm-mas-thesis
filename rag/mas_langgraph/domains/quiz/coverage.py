"""Coverage requirements, acceptance gating, and repair blueprint planning."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from .blueprint_planner import (
    _atoms_for_lo,
    _chunks_for_atoms,
    _generation_goal,
    _infer_bloom_level,
    _normalize_difficulty,
    _section_tagged_chunks,
)
from .acceptance_policy import (
    HARD_REJECTION_FLAGS,
    STRUCTURAL_REJECT_FLAGS,
    build_gate_inputs,
    build_outcome_assessment,
    decide_acceptance,
    status_for_decision,
)
from .entities import MCQBlueprint, MCQ_QUESTION_TYPES, SupervisorDecision
from .generation import log_ok, log_warn
from .section_evidence import update_section_evidence_status
from .versioning import (
    candidate_version_number,
    get_operational_cross_cove_report,
    get_operational_evidence_report,
    get_operational_supervisor_decision,
    report_targets_candidate,
)

if TYPE_CHECKING:
    from ...core.state import CooperativeState

# Re-exported for existing imports; authoritative definitions live in acceptance_policy.


def _section_atoms(state: "CooperativeState", section_id: str) -> List[str]:
    return [
        atom.atom_id
        for atom in state.knowledge_atoms.values()
        if atom.section_id == section_id
    ]


def _section_chunks(state: "CooperativeState", section_id: str) -> List[str]:
    """Section evidence from atoms or explicit section tags — no global fallback."""
    atom_ids = _section_atoms(state, section_id)
    chunk_ids = _chunks_for_atoms(state, atom_ids)
    if chunk_ids:
        return chunk_ids
    return _section_tagged_chunks(state, section_id)


def _section_has_evidence_and_atoms(state: "CooperativeState", section_id: str) -> bool:
    atoms = _section_atoms(state, section_id)
    chunks = _section_chunks(state, section_id)
    return bool(atoms and chunks)


def _lo_has_evidence_and_atoms(
    state: "CooperativeState",
    lo_id: str,
    section_id: str,
) -> bool:
    atom_ids = _atoms_for_lo(state, lo_id, section_id)
    chunk_ids = _chunks_for_atoms(state, atom_ids)
    if not chunk_ids:
        chunk_ids = _section_tagged_chunks(state, section_id)
    return bool(atom_ids and chunk_ids)


def _section_coverage_category(
    state: "CooperativeState",
    section_id: str,
    *,
    has_accepted_mcq: bool,
) -> str:
    status_map = update_section_evidence_status(state)
    status = status_map.get(section_id)
    if has_accepted_mcq:
        return "covered"
    if status is not None and status.unsupported:
        return "unsupported_after_retrieval_attempts"
    if _section_has_evidence_and_atoms(state, section_id):
        return "uncovered_but_supported"
    if status is not None and not status.evidence_chunk_ids:
        return "uncovered_due_to_missing_evidence"
    return "uncovered_due_to_missing_evidence"


def _blueprint_for_candidate(state: "CooperativeState", candidate) -> Optional[MCQBlueprint]:
    return state.mcq_blueprints.get(candidate.blueprint_id)


def _accepted_mcqs(state: "CooperativeState"):
    accepted: List[Any] = []
    for mcq in state.candidate_mcqs.values():
        if mcq.status != "accepted":
            continue
        raw_decision = state.supervisor_decisions.get(mcq.q_id)
        if raw_decision is not None:
            if not report_targets_candidate(raw_decision, mcq):
                continue
            if raw_decision.decision != "accept":
                continue
        accepted.append(mcq)
    return accepted


def build_coverage_requirements(state: "CooperativeState") -> Dict[str, Any]:
    """Build per-section and per-LO MCQ coverage requirements from full doc outline."""
    update_section_evidence_status(state)
    sections: Dict[str, Dict[str, Any]] = {}
    learning_objectives: Dict[str, Dict[str, Any]] = {}
    section_titles = {s.section_id: s.title for s in state.doc_outline}

    for section in state.doc_outline:
        status = state.section_evidence_status.get(section.section_id)
        has_atoms = bool(_section_atoms(state, section.section_id))
        has_evidence = bool(_section_chunks(state, section.section_id))
        unsupported = bool(status and status.unsupported)
        require_mcq = has_atoms and has_evidence and not unsupported
        sections[section.section_id] = {
            "section_id": section.section_id,
            "title": section.title,
            "require_mcq": require_mcq,
            "has_evidence": has_evidence,
            "has_atoms": has_atoms,
            "unsupported": unsupported,
            "importance_weight": section.importance_weight,
            "retrieval_attempts": status.retrieval_attempts if status else 0,
        }

    for lo in state.learning_objectives:
        section_meta = sections.get(lo.section_id, {})
        has_atoms = bool(_atoms_for_lo(state, lo.lo_id, lo.section_id))
        atom_ids = _atoms_for_lo(state, lo.lo_id, lo.section_id)
        has_evidence = bool(_chunks_for_atoms(state, atom_ids)) or bool(
            _section_tagged_chunks(state, lo.section_id)
        )
        unsupported = section_meta.get("unsupported", False)
        require_mcq = has_atoms and has_evidence and not unsupported
        learning_objectives[lo.lo_id] = {
            "lo_id": lo.lo_id,
            "section_id": lo.section_id,
            "section_title": section_titles.get(lo.section_id, lo.section_id),
            "objective_text": lo.objective_text,
            "require_mcq": require_mcq,
            "has_evidence": has_evidence,
            "has_atoms": has_atoms,
            "unsupported": unsupported,
            "difficulty": _normalize_difficulty(lo.difficulty_level),
        }

    if not learning_objectives and sections:
        for section_id, meta in sections.items():
            if meta.get("unsupported"):
                continue
            synthetic_lo = f"lo_{section_id}"
            learning_objectives[synthetic_lo] = {
                "lo_id": synthetic_lo,
                "section_id": section_id,
                "section_title": meta["title"],
                "objective_text": meta["title"],
                "require_mcq": meta["require_mcq"],
                "has_evidence": meta["has_evidence"],
                "has_atoms": meta["has_atoms"],
                "unsupported": meta.get("unsupported", False),
                "difficulty": "intermediate",
            }

    outline_total = len(state.doc_outline)
    unsupported_sections = [
        sid for sid, meta in sections.items() if meta.get("unsupported")
    ]

    return {
        "sections": sections,
        "learning_objectives": learning_objectives,
        "sections_requiring_mcq": [
            sid for sid, meta in sections.items() if meta["require_mcq"]
        ],
        "los_requiring_mcq": [
            lid for lid, meta in learning_objectives.items() if meta["require_mcq"]
        ],
        "outline_sections_total": outline_total,
        "unsupported_sections": unsupported_sections,
    }


def apply_acceptance_gate(
    candidate,
    state: "CooperativeState",
    *,
    cross_cove_required: bool = True,
    max_revisions: int = 2,
    criteria_meta: Optional[Dict[str, Any]] = None,
    criteria_order: Optional[List[str]] = None,
) -> SupervisorDecision:
    """Decide whether a CandidateMCQ may be accepted for the final quiz pack.

    Uses the single authoritative ``decide_acceptance`` policy and persists
    an ``OutcomeAssessment`` (Gamma) alongside the ``SupervisorDecision``.
    """
    from ...artifacts.models import ArtifactRef

    blueprint = _blueprint_for_candidate(state, candidate)
    evidence_report = get_operational_evidence_report(state, candidate)
    cross_cove_report = get_operational_cross_cove_report(state, candidate)
    correct_count = sum(1 for o in candidate.options if o.is_correct)

    inputs = build_gate_inputs(
        revision_count=int(getattr(candidate, "revision_count", 0) or 0),
        max_revisions=max_revisions,
        candidate_status=str(getattr(candidate, "status", "draft") or "draft"),
        flags=list(getattr(candidate, "flags", []) or []),
        evidence_report=evidence_report,
        cross_cove_report=cross_cove_report,
        cross_cove_required=cross_cove_required,
        source_refs_present=bool(getattr(candidate, "source_refs", None)),
        correct_option_count=correct_count,
        blueprint=blueprint,
    )
    gate = decide_acceptance(inputs)

    if gate.append_max_revisions_flag and "max_revisions_reached" not in candidate.flags:
        candidate.flags.append("max_revisions_reached")

    candidate.status = status_for_decision(gate.decision)

    version = candidate_version_number(candidate)
    target_ref = ArtifactRef(logical_id=str(candidate.q_id), version=version)
    assessment = build_outcome_assessment(
        target_ref,
        inputs,
        criteria_meta=criteria_meta,
        criteria_order=criteria_order,
    )
    if not hasattr(state, "outcome_assessments") or getattr(state, "outcome_assessments") is None:
        state.outcome_assessments = {}
    if hasattr(state, "store_outcome_assessment") and callable(
        getattr(state, "store_outcome_assessment")
    ):
        state.store_outcome_assessment(assessment)
    else:
        state.outcome_assessments[str(candidate.q_id)] = assessment

    return SupervisorDecision(
        q_id=candidate.q_id,
        decision=gate.decision,
        reasons=list(gate.reasons),
        candidate_version=version,
        required_fixes=list(gate.required_fixes),
        assessment_ref=assessment.assessment_ref,
    )


def apply_acceptance_gate_to_all(
    state: "CooperativeState",
    *,
    force: bool = False,
    cross_cove_required: bool = True,
    max_revisions: int = 2,
    criteria_meta: Optional[Dict[str, Any]] = None,
    criteria_order: Optional[List[str]] = None,
    max_final_questions: Optional[int] = None,
) -> List[SupervisorDecision]:
    """Run acceptance gate on all candidates and store supervisor decisions."""
    decisions: List[SupervisorDecision] = []
    remaining = None
    if max_final_questions is not None:
        from .question_limit import remaining_final_question_capacity

        remaining = remaining_final_question_capacity(
            state, {"max_final_questions": max_final_questions}
        )
    candidates = sorted(
        state.candidate_mcqs.values(),
        key=lambda mcq: str(getattr(mcq, "q_id", "")),
    )
    for candidate in candidates:
        if not force and candidate.status in ("accepted", "rejected"):
            existing = get_operational_supervisor_decision(state, candidate)
            if existing is not None:
                decisions.append(existing)
                continue
        decision = apply_acceptance_gate(
            candidate,
            state,
            cross_cove_required=cross_cove_required,
            max_revisions=max_revisions,
            criteria_meta=criteria_meta,
            criteria_order=criteria_order,
        )
        if (
            remaining is not None
            and decision.decision == "accept"
        ):
            if remaining <= 0:
                # Quality passed, but the final quiz is full. Do not count or
                # revise this extra accept; packaging never sees it.
                candidate.status = "draft"
            else:
                remaining = max(0, remaining - 1)
        state.store_supervisor_decision(decision)
        state.store_candidate_mcq(candidate)
        decisions.append(decision)
    return decisions


def compute_mcq_coverage(state: "CooperativeState") -> Dict[str, Any]:
    """Summarize accepted MCQ coverage against the full document outline."""
    requirements = build_coverage_requirements(state)
    accepted = _accepted_mcqs(state)

    per_section: Dict[str, int] = Counter()
    per_lo: Dict[str, int] = Counter()
    per_question_type: Dict[str, int] = Counter()
    per_difficulty: Dict[str, int] = Counter()

    for mcq in accepted:
        blueprint = _blueprint_for_candidate(state, mcq)
        if blueprint is None:
            continue
        per_section[blueprint.section_id] += 1
        per_lo[blueprint.lo_id] += 1
        per_question_type[blueprint.question_type] += 1
        per_difficulty[blueprint.difficulty] += 1

    status_counts = Counter(mcq.status for mcq in state.candidate_mcqs.values())
    hard_flag_count = sum(
        1
        for mcq in state.candidate_mcqs.values()
        if any(flag in HARD_REJECTION_FLAGS for flag in mcq.flags)
    )

    confidences = [
        get_operational_evidence_report(state, mcq).confidence
        for mcq in accepted
        if get_operational_evidence_report(state, mcq) is not None
    ]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    outline_total = requirements["outline_sections_total"]
    unsupported = requirements["unsupported_sections"]
    supported_section_ids = [
        sid
        for sid in requirements["sections"]
        if sid not in unsupported
        and _section_has_evidence_and_atoms(state, sid)
    ]

    section_categories: Dict[str, str] = {}
    covered_sections: List[str] = []
    uncovered_supported: List[str] = []
    uncovered_missing_evidence: List[str] = []
    unsupported_after_retrieval: List[str] = []

    for section in state.doc_outline:
        sid = section.section_id
        has_mcq = per_section.get(sid, 0) > 0
        category = _section_coverage_category(state, sid, has_accepted_mcq=has_mcq)
        section_categories[sid] = category
        if category == "covered":
            covered_sections.append(sid)
        elif category == "uncovered_but_supported":
            uncovered_supported.append(sid)
        elif category == "unsupported_after_retrieval_attempts":
            unsupported_after_retrieval.append(sid)
        else:
            uncovered_missing_evidence.append(sid)

    los_requiring = requirements["los_requiring_mcq"]
    covered_los = [lid for lid in los_requiring if per_lo.get(lid, 0) > 0]
    uncovered_los = [lid for lid in los_requiring if per_lo.get(lid, 0) == 0]

    los_with_evidence = sum(
        1 for meta in requirements["learning_objectives"].values() if meta["has_evidence"]
    )
    sections_with_evidence = len(
        state.get_sections_with_sufficient_evidence()
    )

    return {
        "accepted_count": len(accepted),
        "per_section": dict(per_section),
        "per_lo": dict(per_lo),
        "per_question_type": dict(per_question_type),
        "per_difficulty": dict(per_difficulty),
        "needs_revision_count": status_counts.get("needs_revision", 0),
        "rejected_count": status_counts.get("rejected", 0),
        "draft_count": status_counts.get("draft", 0),
        "evidence_verified_count": status_counts.get("evidence_verified", 0),
        "total_candidates": len(state.candidate_mcqs),
        "uncovered_sections": uncovered_supported,
        "uncovered_learning_objectives": uncovered_los,
        "uncovered_due_to_missing_evidence": uncovered_missing_evidence,
        "unsupported_sections": unsupported_after_retrieval,
        "uncovered_supported_sections": uncovered_supported,
        "section_categories": section_categories,
        "coverage_sections_total": outline_total,
        "coverage_sections_covered": len(covered_sections),
        "coverage_sections_supported_total": len(supported_section_ids),
        "coverage_los_total": len(requirements["learning_objectives"]),
        "coverage_los_covered": len(covered_los),
        "learning_objectives_with_evidence": los_with_evidence,
        "sections_with_evidence": sections_with_evidence,
        "avg_evidence_confidence": avg_confidence,
        "hard_flag_count": hard_flag_count,
        "requirements": requirements,
    }


def find_coverage_gaps(state: "CooperativeState") -> List[Dict[str, Any]]:
    """Return uncovered section/LO targets that have evidence and atoms (repairable)."""
    coverage = compute_mcq_coverage(state)
    requirements = coverage["requirements"]
    gaps: List[Dict[str, Any]] = []
    seen: Set[tuple] = set()

    section_titles = {
        s.section_id: s.title for s in state.doc_outline
    }

    for section_id in coverage["uncovered_supported_sections"]:
        meta = requirements["sections"].get(section_id, {})
        if meta.get("unsupported"):
            continue
        if not _section_has_evidence_and_atoms(state, section_id):
            continue
        key = (section_id, "")
        if key in seen:
            continue
        seen.add(key)
        lo_id = f"lo_{section_id}"
        if lo_id not in requirements["learning_objectives"]:
            for lo in state.learning_objectives:
                if lo.section_id == section_id:
                    lo_id = lo.lo_id
                    break
        gaps.append({
            "section_id": section_id,
            "lo_id": lo_id,
            "section_title": section_titles.get(section_id, section_id),
            "reason": "no accepted MCQ for section",
            "suggested_question_type": MCQ_QUESTION_TYPES[len(gaps) % len(MCQ_QUESTION_TYPES)],
            "suggested_difficulty": "intermediate",
        })

    for lo_id in coverage["uncovered_learning_objectives"]:
        lo_meta = requirements["learning_objectives"].get(lo_id, {})
        if not lo_meta.get("require_mcq"):
            continue
        if not _lo_has_evidence_and_atoms(state, lo_id, lo_meta.get("section_id", "")):
            continue
        key = (lo_meta.get("section_id", ""), lo_id)
        if key in seen:
            continue
        seen.add(key)
        gaps.append({
            "section_id": lo_meta.get("section_id", ""),
            "lo_id": lo_id,
            "section_title": lo_meta.get("section_title", lo_meta.get("section_id", "")),
            "reason": "no accepted MCQ for learning objective",
            "suggested_question_type": MCQ_QUESTION_TYPES[len(gaps) % len(MCQ_QUESTION_TYPES)],
            "suggested_difficulty": lo_meta.get("difficulty", "intermediate"),
        })

    gaps.sort(
        key=lambda g: requirements["sections"]
        .get(g["section_id"], {})
        .get("importance_weight", 0.0),
        reverse=True,
    )
    return gaps


def _existing_coverage_keys(state: "CooperativeState") -> Set[tuple]:
    keys: Set[tuple] = set()
    for bp in state.mcq_blueprints.values():
        if bp.status in ("planned", "generated"):
            keys.add((bp.section_id, bp.lo_id))
    for mcq in state.candidate_mcqs.values():
        if mcq.status == "accepted":
            bp = _blueprint_for_candidate(state, mcq)
            if bp is not None:
                keys.add((bp.section_id, bp.lo_id))
    return keys


def create_repair_blueprints(
    state: "CooperativeState",
    gaps: List[Dict[str, Any]],
) -> List[MCQBlueprint]:
    """Create MCQBlueprints for uncovered sections/LOs with real evidence and atoms."""
    blueprints: List[MCQBlueprint] = []
    existing = _existing_coverage_keys(state)
    sections = {s.section_id: s for s in state.doc_outline}
    lo_texts = {lo.lo_id: lo.objective_text for lo in state.learning_objectives}

    for gap in gaps:
        section_id = gap.get("section_id") or ""
        lo_id = gap.get("lo_id") or f"lo_{section_id}"
        if not section_id:
            continue
        if (section_id, lo_id) in existing:
            continue
        if not _lo_has_evidence_and_atoms(state, lo_id, section_id):
            continue

        section = sections.get(section_id)
        section_title = section.title if section else gap.get("section_title", section_id)
        if section_title.lower() == "general":
            continue

        atom_ids = _atoms_for_lo(state, lo_id, section_id)
        chunk_ids = _chunks_for_atoms(state, atom_ids)
        if not chunk_ids:
            chunk_ids = _section_tagged_chunks(state, section_id)
        if not atom_ids or not chunk_ids:
            continue

        question_type = gap.get("suggested_question_type") or MCQ_QUESTION_TYPES[0]
        difficulty = _normalize_difficulty(gap.get("suggested_difficulty", "intermediate"))
        lo_text = lo_texts.get(lo_id, section_title)
        bloom_level = _infer_bloom_level(question_type, lo_text)
        goal = (
            f"Repair coverage: {gap.get('reason', 'missing accepted MCQ')} "
            f"for section {section_title!r}."
        )

        blueprint = MCQBlueprint(
            blueprint_id=state.generate_id("bp"),
            section_id=section_id,
            lo_id=lo_id,
            target_atom_ids=atom_ids,
            evidence_chunk_ids=chunk_ids,
            question_type=question_type,
            difficulty=difficulty,
            bloom_level=bloom_level,
            generation_goal=goal,
            status="planned",
        )
        blueprints.append(blueprint)
        existing.add((section_id, lo_id))

    return blueprints


def select_final_mcqs_for_quiz(state: "CooperativeState", *, limit: Optional[int] = None):
    """Return accepted CandidateMCQs eligible for final packaging."""
    from .question_limit import select_final_mcqs_within_limit

    return select_final_mcqs_within_limit(_accepted_mcqs(state), limit)


def summarize_acceptance_decisions(decisions: List[SupervisorDecision]) -> Dict[str, Any]:
    """Aggregate acceptance gate outcomes for logging."""
    accepted = [d for d in decisions if d.decision == "accept"]
    revised = [d for d in decisions if d.decision == "revise"]
    rejected = [d for d in decisions if d.decision == "reject"]
    reason_counter: Counter = Counter()
    for decision in decisions:
        for reason in decision.reasons:
            reason_counter[reason.split(":")[0].strip()] += 1
        for fix in decision.required_fixes:
            reason_counter[fix] += 1
    common_issue = reason_counter.most_common(1)[0][0] if reason_counter else ""
    return {
        "accepted": len(accepted),
        "needs_revision": len(revised),
        "rejected": len(rejected),
        "common_issue": common_issue,
        "reason_counts": dict(reason_counter),
    }


def store_coverage_summary_on_state(
    state: "CooperativeState",
    coverage: Dict[str, Any],
    gaps: List[Dict[str, Any]],
) -> None:
    """Persist coverage summary and gaps on cooperative state."""
    summary = {
        **{k: v for k, v in coverage.items() if k != "requirements"},
        "gaps": gaps,
        "gap_count": len(gaps),
    }
    if hasattr(state, "store_mcq_coverage_summary"):
        state.store_mcq_coverage_summary(summary)
    else:
        state.mcq_coverage_summary = summary  # type: ignore[attr-defined]
