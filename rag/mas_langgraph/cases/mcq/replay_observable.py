"""MCQ target-observability predicate for AXIS bounded replay.

A target is observable only when ``MCQRunObservationAdapter.observe_replay``
can reconstruct a terminal CaseRunObservation for the explained MCQ — the same
observation a full replay-to-END would yield for that target.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Tuple

from ...artifacts.channel_state import load_artifact_channels
from ...artifacts.models import ArtifactRef
from ...artifacts.utils import payloads_equivalent
from ...axis.models import ExplainedCase
from ...domains.quiz.versioning import (
    get_operational_cross_cove_report,
    get_operational_evidence_report,
    report_targets_candidate,
)
from .comparison import resolve_current_branch_subject

TERMINAL_DECISIONS = frozenset({"accept", "reject"})


class ReplayObservableStatus(str, Enum):
    """Result of evaluating whether bounded replay may stop."""

    CONTINUE = "continue"
    COMPLETE = "complete"
    FALLBACK_FULL = "fallback_full"


def _decision_value(decision_payload: Any) -> str:
    raw = getattr(decision_payload, "decision", None) or getattr(
        decision_payload, "outcome", None
    )
    return str(raw or "").strip().lower()


def evaluate_mcq_replay_observable(
    final_state: Any,
    explained_case: ExplainedCase,
    *,
    branch_id: str = "",
) -> ReplayObservableStatus:
    """Return whether the explained MCQ outcome is fully observable.

    COMPLETE
        Target has a terminal accept/reject decision, matching assessment, and
        required catalog envelopes — safe to stop.

    CONTINUE
        Target not yet terminal (missing, still revising, or mid-verification).

    FALLBACK_FULL
        State is inconsistent / ambiguous for safe early stop.
    """
    if explained_case is None:
        return ReplayObservableStatus.FALLBACK_FULL
    if not isinstance(final_state, dict):
        return ReplayObservableStatus.CONTINUE

    cs = final_state.get("cooperative_state")
    if cs is None:
        return ReplayObservableStatus.CONTINUE

    candidates = getattr(cs, "candidate_mcqs", None) or {}
    if not isinstance(candidates, dict):
        return ReplayObservableStatus.CONTINUE

    # Ambiguous / inconsistent correspondence → do not guess; fall back to END.
    key = getattr(explained_case, "correspondence_key", None)
    q_id = explained_case.artifact_ref.logical_id
    if key:
        matches = [
            cid
            for cid, cand in list(candidates.items())
            if str(getattr(cand, "blueprint_id", "") or "") == str(key)
        ]
        if len(matches) > 1:
            return ReplayObservableStatus.FALLBACK_FULL
        if q_id in candidates:
            own = candidates[q_id]
            if str(getattr(own, "blueprint_id", "") or "") != str(key) and not matches:
                # Explained q_id was renumbered onto a different blueprint and the
                # original correspondence key is gone — definitive absence only at END.
                pass

    try:
        registry, _resolver = load_artifact_channels(final_state)
    except Exception:
        return ReplayObservableStatus.FALLBACK_FULL

    candidate, replay_q_id, version, subject_ref, cand_env, status = (
        resolve_current_branch_subject(
            explained_case=explained_case,
            candidates=candidates,
            registry=registry,
            branch_id=branch_id,
        )
    )
    if status == "absent" or candidate is None or replay_q_id is None:
        # Absence mid-graph is not proof — keep running until END.
        return ReplayObservableStatus.CONTINUE
    if status == "missing_envelope" or cand_env is None or version is None or subject_ref is None:
        # Candidate in operational dict but not cataloged yet.
        return ReplayObservableStatus.CONTINUE

    decisions = getattr(cs, "supervisor_decisions", None) or {}
    assessments = getattr(cs, "outcome_assessments", None) or {}
    if not isinstance(decisions, dict):
        decisions = {}
    if not isinstance(assessments, dict):
        assessments = {}

    decision_payload = decisions.get(replay_q_id)
    assessment_payload = assessments.get(replay_q_id)

    if decision_payload is None or assessment_payload is None:
        return ReplayObservableStatus.CONTINUE

    if not report_targets_candidate(decision_payload, candidate):
        # Stale decision for a newer revision still in progress.
        return ReplayObservableStatus.CONTINUE

    decision = _decision_value(decision_payload)
    if decision == "revise":
        return ReplayObservableStatus.CONTINUE
    if decision not in TERMINAL_DECISIONS:
        return ReplayObservableStatus.CONTINUE

    dver = getattr(decision_payload, "candidate_version", None)
    if dver is not None and int(dver) != int(version):
        return ReplayObservableStatus.FALLBACK_FULL

    decision_ref = ArtifactRef(f"decision:{replay_q_id}", version)
    assessment_ref = ArtifactRef(f"assessment:{replay_q_id}", version)
    linked = getattr(decision_payload, "assessment_ref", None)
    if linked is not None:
        assessment_ref = linked

    denv = registry.get_by_ref(decision_ref, branch_id) or registry.get_by_ref(
        decision_ref
    )
    if denv is None:
        return ReplayObservableStatus.CONTINUE

    aenv = registry.get_by_ref(assessment_ref, branch_id) or registry.get_by_ref(
        assessment_ref
    )
    if aenv is None:
        return ReplayObservableStatus.CONTINUE

    target = getattr(assessment_payload, "target_ref", None)
    if target is not None and target != subject_ref:
        return ReplayObservableStatus.FALLBACK_FULL

    a_ref = getattr(assessment_payload, "assessment_ref", None)
    if linked is not None and a_ref is not None and linked != a_ref:
        return ReplayObservableStatus.FALLBACK_FULL

    # Prefer exact-version evidence / Cross-CoVe when present; stale → wait.
    ev = getattr(cs, "evidence_reports", None) or {}
    if isinstance(ev, dict) and replay_q_id in ev:
        if get_operational_evidence_report(cs, candidate) is None:
            return ReplayObservableStatus.CONTINUE

    cove = getattr(cs, "cross_cove_reports", None) or {}
    if isinstance(cove, dict) and replay_q_id in cove:
        if get_operational_cross_cove_report(cs, candidate) is None:
            return ReplayObservableStatus.CONTINUE

    # Accept paths always produce versioned evidence + Cross-CoVe before the gate.
    if decision == "accept":
        if get_operational_evidence_report(cs, candidate) is None:
            return ReplayObservableStatus.CONTINUE
        if get_operational_cross_cove_report(cs, candidate) is None:
            return ReplayObservableStatus.CONTINUE

    def _catalog_has(ref: ArtifactRef) -> bool:
        return (
            registry.get_by_ref(ref, branch_id) is not None
            or registry.get_by_ref(ref) is not None
        )

    # Require immutable envelopes for reports that exist at this version so
    # observe_replay / AXIS comparison can use catalog-backed history.
    if get_operational_evidence_report(cs, candidate) is not None:
        eref = ArtifactRef(f"evidence-report:{replay_q_id}", version)
        if not _catalog_has(eref):
            return ReplayObservableStatus.CONTINUE
    if get_operational_cross_cove_report(cs, candidate) is not None:
        cref = ArtifactRef(f"cross-cove-report:{replay_q_id}", version)
        if not _catalog_has(cref):
            return ReplayObservableStatus.CONTINUE

    # Catalog payload should match the operational candidate for this version.
    if not payloads_equivalent(cand_env.payload, candidate):
        # Trust catalog; still complete if decision/assessment target this version.
        pass

    return ReplayObservableStatus.COMPLETE


def mcq_case_replay_observable(
    final_state: Any,
    explained_case: ExplainedCase,
    *,
    branch_id: str = "",
) -> bool:
    """True iff the explained MCQ has a terminal, catalog-backed outcome."""
    return (
        evaluate_mcq_replay_observable(
            final_state, explained_case, branch_id=branch_id
        )
        == ReplayObservableStatus.COMPLETE
    )


def default_mcq_stop_predicate(
    final_state: Any,
    explained_case: ExplainedCase,
    *,
    branch_id: str = "",
) -> ReplayObservableStatus:
    """Case-level callback used by LangGraphReplayExecutor."""
    return evaluate_mcq_replay_observable(
        final_state, explained_case, branch_id=branch_id
    )
