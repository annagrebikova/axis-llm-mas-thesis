"""Attach Step 2 MCQ fields to any cooperative blackboard (mas_langgraph or RAG monolith)."""

from __future__ import annotations

import types
from typing import Any, Callable, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .entities import MCQBlueprint


def _is_local_callable(value: Any) -> bool:
    fn = getattr(value, "__func__", value)
    qual = str(getattr(fn, "__qualname__", "") or "")
    return "<locals>" in qual


def _should_bind_helper(state: Any, name: str) -> bool:
    inst = getattr(state, "__dict__", None)
    if isinstance(inst, dict) and name in inst:
        existing = inst[name]
        return (not callable(existing)) or _is_local_callable(existing)
    return not callable(getattr(state, name, None))


def _bind_helper(state: Any, name: str, func: Callable[..., Any]) -> None:
    setattr(state, name, types.MethodType(func, state))


def _helper_update_section_evidence_status(self: Any, *, min_chunks: int = 2):
    from .section_evidence import update_section_evidence_status as _update

    return _update(self, min_chunks=min_chunks)


def _helper_get_sections_without_evidence(self: Any):
    from .section_evidence import get_sections_without_evidence as _get

    return _get(self)


def _helper_get_sections_with_sufficient_evidence(self: Any):
    from .section_evidence import get_sections_with_sufficient_evidence as _get

    return _get(self)


def _helper_store_mcq_blueprints(self: Any, blueprints: List["MCQBlueprint"]) -> None:
    for blueprint in blueprints:
        self.mcq_blueprints[blueprint.blueprint_id] = blueprint


def _helper_planned_blueprints(self: Any) -> List["MCQBlueprint"]:
    return [
        bp
        for bp in self.mcq_blueprints.values()
        if getattr(bp, "status", "planned") == "planned"
    ]


def _helper_store_candidate_mcq(self: Any, candidate: Any) -> None:
    self.candidate_mcqs[candidate.q_id] = candidate


def _helper_store_candidate_mcqs(self: Any, candidates: List[Any]) -> None:
    for candidate in candidates:
        self.store_candidate_mcq(candidate)


def _helper_store_evidence_report(self: Any, report: Any) -> None:
    self.evidence_reports[report.q_id] = report


def _helper_store_evidence_reports(self: Any, reports: List[Any]) -> None:
    for report in reports:
        self.store_evidence_report(report)


def _helper_store_supervisor_decision(self: Any, decision: Any) -> None:
    self.supervisor_decisions[decision.q_id] = decision


def _helper_store_outcome_assessment(self: Any, assessment: Any) -> None:
    q_id = getattr(getattr(assessment, "target_ref", None), "logical_id", None)
    if q_id is None:
        return
    if not hasattr(self, "outcome_assessments") or self.outcome_assessments is None:
        self.outcome_assessments = {}
    self.outcome_assessments[str(q_id)] = assessment


def _helper_store_mcq_coverage_summary(self: Any, summary: Dict[str, Any]) -> None:
    self.mcq_coverage_summary = summary


def _helper_store_cross_cove_report(self: Any, report: Any) -> None:
    self.cross_cove_reports[report.q_id] = report


def _helper_store_cross_cove_reports(self: Any, reports: List[Any]) -> None:
    for report in reports:
        self.store_cross_cove_report(report)


def _helper_add_revision_record(self: Any, record: Any) -> None:
    self.revision_history.append(record)


def _helper_get_revision_count(self: Any, q_id: str) -> int:
    candidate = (getattr(self, "candidate_mcqs", {}) or {}).get(q_id)
    if candidate is None:
        return 0
    return getattr(candidate, "revision_count", 0)


STEP2_HELPER_FUNCS: Dict[str, Callable[..., Any]] = {
    "update_section_evidence_status": _helper_update_section_evidence_status,
    "get_sections_without_evidence": _helper_get_sections_without_evidence,
    "get_sections_with_sufficient_evidence": _helper_get_sections_with_sufficient_evidence,
    "store_mcq_blueprints": _helper_store_mcq_blueprints,
    "planned_blueprints": _helper_planned_blueprints,
    "store_candidate_mcq": _helper_store_candidate_mcq,
    "store_candidate_mcqs": _helper_store_candidate_mcqs,
    "store_evidence_report": _helper_store_evidence_report,
    "store_evidence_reports": _helper_store_evidence_reports,
    "store_supervisor_decision": _helper_store_supervisor_decision,
    "store_outcome_assessment": _helper_store_outcome_assessment,
    "store_mcq_coverage_summary": _helper_store_mcq_coverage_summary,
    "store_cross_cove_report": _helper_store_cross_cove_report,
    "store_cross_cove_reports": _helper_store_cross_cove_reports,
    "add_revision_record": _helper_add_revision_record,
    "get_revision_count": _helper_get_revision_count,
}


def ensure_step2_mcq_state(state: Any) -> None:
    """Initialize Step 2 MCQ containers and helper methods on ``state`` if missing."""
    if not hasattr(state, "mcq_blueprints"):
        state.mcq_blueprints = {}
    if not hasattr(state, "candidate_mcqs"):
        state.candidate_mcqs = {}
    if not hasattr(state, "language_reports"):
        state.language_reports = {}
    if not hasattr(state, "evidence_reports"):
        state.evidence_reports = {}
    if not hasattr(state, "cross_cove_reports"):
        state.cross_cove_reports = {}
    if not hasattr(state, "supervisor_decisions"):
        state.supervisor_decisions = {}
    if not hasattr(state, "outcome_assessments"):
        state.outcome_assessments = {}
    if not hasattr(state, "revision_history"):
        state.revision_history = []
    if not hasattr(state, "mcq_coverage_summary"):
        state.mcq_coverage_summary = {}
    if not hasattr(state, "section_evidence_status"):
        state.section_evidence_status = {}
    if not hasattr(state, "_id_counters") or not isinstance(
        getattr(state, "_id_counters", None), dict
    ):
        state._id_counters = {}
    if not hasattr(state, "quiz_generation_rounds"):
        state.quiz_generation_rounds = 0

    for name, func in STEP2_HELPER_FUNCS.items():
        if _should_bind_helper(state, name):
            _bind_helper(state, name, func)


INSTANCE_HELPER_ATTRS = tuple(STEP2_HELPER_FUNCS) + ("snapshot_counts",)


def _detach_instance_helpers(state: Any) -> None:
    """Drop bound helpers so pickle stores dataclass fields only."""
    inst = getattr(state, "__dict__", None)
    if not isinstance(inst, dict):
        return
    for name in INSTANCE_HELPER_ATTRS:
        if name in inst:
            try:
                delattr(state, name)
            except AttributeError:
                pass


def looks_like_cooperative_state(obj: Any) -> bool:
    if obj is None or isinstance(obj, dict):
        return False
    try:
        from unittest.mock import Mock

        if isinstance(obj, Mock):
            return False
    except Exception:
        pass
    evidence = getattr(obj, "evidence_pool", None)
    outline = getattr(obj, "doc_outline", None)
    return isinstance(evidence, dict) and isinstance(outline, (list, tuple))


def rebind_pickleable_state_helpers(state: Any) -> None:
    """Replace nested/local instance helpers with module-level bound methods.

    LectureBank eval uses ``rag_agents_cooperative.CooperativeState``, which has
    no Step-2 methods on the class. Production attaches them via
    ``ensure_step2_mcq_state``. Nested closures cannot be pickled; module-level
    ``types.MethodType`` bindings can, and they rebind to the restored instance
    rather than a discarded process-local blackboard.
    """
    if state is None or not looks_like_cooperative_state(state):
        return
    ensure_step2_mcq_state(state)
    inst = getattr(state, "__dict__", None)
    has_instance_snapshot = isinstance(inst, dict) and "snapshot_counts" in inst
    class_has_snapshot = callable(getattr(type(state), "snapshot_counts", None))
    if has_instance_snapshot or not class_has_snapshot:
        _bind_helper(state, "snapshot_counts", rag_monolith_snapshot_counts)


def _step2_snapshot_fields(state: Any) -> Dict[str, int]:
    return {
        "mcq_blueprints_len": len(getattr(state, "mcq_blueprints", {}) or {}),
        "candidate_mcqs_len": len(getattr(state, "candidate_mcqs", {}) or {}),
        "evidence_reports_len": len(getattr(state, "evidence_reports", {}) or {}),
        "supervisor_decisions_len": len(getattr(state, "supervisor_decisions", {}) or {}),
        "outcome_assessments_len": len(getattr(state, "outcome_assessments", {}) or {}),
        "accepted_mcqs_len": sum(
            1
            for mcq in (getattr(state, "candidate_mcqs", {}) or {}).values()
            if getattr(mcq, "status", "") == "accepted"
        ),
        "cross_cove_reports_len": len(getattr(state, "cross_cove_reports", {}) or {}),
        "revision_history_len": len(getattr(state, "revision_history", []) or []),
    }


def rag_monolith_snapshot_counts(self: Any) -> Dict[str, Any]:
    """Module-level snapshot_counts for RAG CooperativeState (pickleable)."""
    canonical_chunks = sum(
        1
        for c in (getattr(self, "evidence_pool", {}) or {}).values()
        if getattr(c, "duplicate_of", None) is None
    )
    low_quality_chunks = sum(
        1
        for c in (getattr(self, "evidence_pool", {}) or {}).values()
        if "low_quality" in getattr(c, "flags", [])
    )
    quality = getattr(self, "quality_metrics", None)
    out: Dict[str, Any] = {
        "doc_outline_len": len(getattr(self, "doc_outline", []) or []),
        "learning_objectives_len": len(getattr(self, "learning_objectives", []) or []),
        "evidence_pool_len": len(getattr(self, "evidence_pool", {}) or {}),
        "canonical_chunks": canonical_chunks,
        "low_quality_chunks": low_quality_chunks,
        "knowledge_atoms_len": len(getattr(self, "knowledge_atoms", {}) or {}),
        "candidate_questions_len": len(getattr(self, "candidate_questions", {}) or {}),
        "validated_questions_len": len(getattr(self, "validated_questions", {}) or {}),
        "rejected_questions_len": len(getattr(self, "rejected_questions", {}) or {}),
        "duplication_score": getattr(quality, "duplication_score", None),
        "rejection_rate": getattr(quality, "rejection_rate", None),
        "avg_evidence_strength": getattr(quality, "avg_evidence_strength", None),
        "final_quiz_packs_len": len(getattr(self, "final_quiz_packs", []) or []),
    }
    out.update(_step2_snapshot_fields(self))
    return out


rag_monolith_snapshot_counts._step2_wrapped = True  # type: ignore[attr-defined]


def attach_step2_snapshot_counts(state: Any) -> None:
    """Attach pickleable Step-2 helpers and RAG ``snapshot_counts`` when needed."""
    rebind_pickleable_state_helpers(state)
