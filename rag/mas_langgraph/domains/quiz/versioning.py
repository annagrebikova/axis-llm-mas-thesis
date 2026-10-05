"""Candidate/report version helpers for immutable MCQ revisions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ...core.state import CooperativeState
    from .entities import CandidateMCQ, CrossCoVeReport, EvidenceCheckReport, SupervisorDecision


def candidate_version_number(candidate: "CandidateMCQ") -> int:
    """1-based artifact version for a candidate domain object."""
    return int(getattr(candidate, "revision_count", 0)) + 1


def _report_version(report: Any) -> int:
    return int(getattr(report, "candidate_version", 1) or 1)


def report_targets_candidate(report: Any, candidate: "CandidateMCQ") -> bool:
    """Return True when a report/decision targets the exact candidate version."""
    return (
        getattr(report, "q_id", None) == candidate.q_id
        and _report_version(report) == candidate_version_number(candidate)
    )


def get_operational_evidence_report(
    state: "CooperativeState",
    candidate: "CandidateMCQ",
) -> Optional["EvidenceCheckReport"]:
    report = state.evidence_reports.get(candidate.q_id)
    if report is None:
        return None
    if not report_targets_candidate(report, candidate):
        return None
    return report


def get_operational_cross_cove_report(
    state: "CooperativeState",
    candidate: "CandidateMCQ",
) -> Optional["CrossCoVeReport"]:
    report = state.cross_cove_reports.get(candidate.q_id)
    if report is None:
        return None
    if not report_targets_candidate(report, candidate):
        return None
    return report


def get_operational_supervisor_decision(
    state: "CooperativeState",
    candidate: "CandidateMCQ",
) -> Optional["SupervisorDecision"]:
    decision = state.supervisor_decisions.get(candidate.q_id)
    if decision is None:
        return None
    if not report_targets_candidate(decision, candidate):
        return None
    return decision


def clear_operational_reports(state: "CooperativeState", q_id: str) -> None:
    """Clear latest operational report pointers without deleting artifact history."""
    state.evidence_reports.pop(q_id, None)
    state.cross_cove_reports.pop(q_id, None)
    state.supervisor_decisions.pop(q_id, None)
    if hasattr(state, "outcome_assessments"):
        state.outcome_assessments.pop(q_id, None)
