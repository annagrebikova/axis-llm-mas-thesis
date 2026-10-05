"""MCQ-specific run observation adapter for AXIS replay comparison."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ...artifacts.channel_state import load_artifact_channels
from ...artifacts.models import ArtifactRef
from ...artifacts.utils import payloads_equivalent
from ...axis.comparison import CaseRunObservation
from ...axis.models import ExplainedCase
from .artifacts import candidate_artifact_version


def _branch_type(run_result: Any) -> str:
    meta = getattr(run_result, "branch_metadata", None)
    if meta is not None:
        return str(getattr(meta, "branch_type", "") or "")
    return str(getattr(run_result, "branch_type", "") or "")


def resolve_corresponding_candidate(
    *,
    explained_case: ExplainedCase,
    candidates: dict,
) -> Tuple[Optional[Any], Optional[str]]:
    """Resolve the replay counterpart of the explained CandidateMCQ.

    When ``correspondence_key`` (blueprint_id) is available it is authoritative:
    an exact ``q_id`` match is accepted only if that candidate's blueprint also
    matches. Positional q_id renumbering after interventions must not silently
    remap the explained question onto a different blueprint.

    Without a correspondence key, fall back to exact logical_id only.
    Never fuzzy-match by stem.
    """
    if not isinstance(candidates, dict) or not candidates:
        return None, None
    q_id = explained_case.artifact_ref.logical_id
    key = getattr(explained_case, "correspondence_key", None)

    if key:
        if q_id in candidates:
            cand = candidates[q_id]
            if str(getattr(cand, "blueprint_id", "") or "") == str(key):
                return cand, str(q_id)
        matches = [
            (cid, cand)
            for cid, cand in list(candidates.items())
            if str(getattr(cand, "blueprint_id", "") or "") == str(key)
        ]
        if len(matches) == 1:
            return matches[0][1], str(matches[0][0])
        if len(matches) > 1:
            # Ambiguous: several current candidates share the correspondence key.
            # Do not pick the explained *factual* version from among them.
            return None, None
        return None, None

    if q_id in candidates:
        return candidates[q_id], str(q_id)
    return None, None


def resolve_current_branch_subject(
    *,
    explained_case: ExplainedCase,
    candidates: dict,
    registry: Any,
    branch_id: str = "",
) -> Tuple[Optional[Any], Optional[str], Optional[int], Optional[ArtifactRef], Optional[Any], str]:
    """Resolve the current corresponding MCQ version on a replay branch.

    Correspondence (``q_id`` / ``correspondence_key``) identifies the lineage.
    The *operational* candidate then determines the current immutable version.
    Historical catalog presence of the explained factual version must not override
    that current version.

    Returns ``(candidate, q_id, version, subject_ref, catalog_env, status)``
    where status is ``absent``, ``ok``, or ``missing_envelope``.
    """
    candidate, replay_q_id = resolve_corresponding_candidate(
        explained_case=explained_case,
        candidates=candidates,
    )
    if candidate is None or replay_q_id is None:
        return None, None, None, None, None, "absent"

    version = candidate_artifact_version(candidate)
    subject_ref = ArtifactRef(replay_q_id, version)
    cand_env = None
    if registry is not None:
        cand_env = registry.get_by_ref(subject_ref, branch_id) or registry.get_by_ref(
            subject_ref
        )
    if cand_env is None:
        return candidate, replay_q_id, version, subject_ref, None, "missing_envelope"
    payload = getattr(cand_env, "payload", None)
    if payload is not None:
        candidate = payload
    return candidate, replay_q_id, version, subject_ref, cand_env, "ok"


class MCQRunObservationAdapter:
    """Extract MCQ-specific outcome observations from workflow branches."""

    def __init__(
        self,
        *,
        case_slice: Any = None,
        checkpoint_index: Any = None,
        occurrences: Any = None,
        produced_by: Any = None,
    ) -> None:
        self.case_slice = case_slice
        self.checkpoint_index = checkpoint_index
        self.occurrences = occurrences
        self.produced_by = produced_by

    def observe_factual(
        self,
        *,
        explained_case: ExplainedCase,
        case_slice: Any,
        registry: Any,
    ) -> CaseRunObservation:
        """Build factual observation from Prompt-12 exact artifacts."""
        self.case_slice = case_slice
        subject_payload = None
        if registry is not None:
            env = registry.get_by_ref(
                explained_case.artifact_ref, explained_case.branch_id
            ) or registry.get_by_ref(explained_case.artifact_ref)
            if env is not None:
                subject_payload = getattr(env, "payload", None)
        return CaseRunObservation(
            run_id=explained_case.run_id,
            branch_id=explained_case.branch_id,
            subject_ref=explained_case.artifact_ref,
            assessment_ref=getattr(case_slice, "assessment_ref", None)
            or explained_case.assessment_ref,
            decision_ref=explained_case.decision_ref,
            decision=getattr(case_slice, "decision", None) or explained_case.outcome,
            criterion_values=tuple(getattr(case_slice, "criterion_values", ())),
            decision_reasons=tuple(getattr(case_slice, "decision_reasons", ())),
            required_fixes=tuple(getattr(case_slice, "required_fixes", ())),
            subject_payload=subject_payload,
            subject_present=True,
            outcome_available=True,
        )

    def compare_subject_content(
        self,
        *,
        factual: CaseRunObservation,
        noop: CaseRunObservation,
        counterfactual: CaseRunObservation,
        intervention_before_subject_generation: Optional[bool] = None,
        subject_generation_relation: Optional[str] = None,
    ):
        from .content_comparison import compare_mcq_subject_content

        return compare_mcq_subject_content(
            factual=factual,
            noop=noop,
            counterfactual=counterfactual,
            intervention_before_subject_generation=intervention_before_subject_generation,
            subject_generation_relation=subject_generation_relation,
        )

    def infer_subject_generation_relation(
        self,
        *,
        signature: Any,
        explained_case: ExplainedCase,
    ):
        from .content_comparison import infer_subject_generation_relation

        return infer_subject_generation_relation(
            signature=signature,
            explained_case=explained_case,
            case_slice=self.case_slice,
            checkpoint_index=self.checkpoint_index,
            occurrences=self.occurrences or (),
            produced_by=self.produced_by or (),
        )

    def observe_replay(
        self,
        *,
        explained_case: ExplainedCase,
        run_result: Any,
    ) -> CaseRunObservation:
        """Extract observation from a replay branch (noop or counterfactual)."""
        final_state = getattr(run_result, "final_state", {}) or {}
        branch_id = getattr(run_result, "branch_id", "?")
        run_id = getattr(run_result, "run_id", explained_case.run_id)
        error = getattr(run_result, "error", None)
        btype = _branch_type(run_result)

        if error:
            return CaseRunObservation(
                run_id=run_id,
                branch_id=branch_id,
                subject_present=False,
                outcome_available=False,
                error=error,
            )

        cs = final_state.get("cooperative_state")
        candidates = {}
        if cs is not None:
            raw = getattr(cs, "candidate_mcqs", None) or {}
            if isinstance(raw, dict):
                candidates = raw

        candidate, replay_q_id = resolve_corresponding_candidate(
            explained_case=explained_case,
            candidates=candidates,
        )

        if candidate is None or replay_q_id is None:
            # Subject absence: control failure for NOOP; observable effect for CF.
            return CaseRunObservation(
                run_id=run_id,
                branch_id=branch_id,
                subject_ref=None,
                subject_present=False,
                outcome_available=True,
                error=(
                    "NOOP control failed: corresponding explained CandidateMCQ "
                    "was not produced in this replay"
                    if btype == "noop"
                    else None
                ),
            )

        try:
            registry, _resolver = load_artifact_channels(final_state)
        except Exception as exc:
            return self._inconsistent(
                run_id, branch_id, f"branch artifact history unavailable: {exc}"
            )

        candidate, replay_q_id, version, subject_ref, cand_env, status = (
            resolve_current_branch_subject(
                explained_case=explained_case,
                candidates=candidates,
                registry=registry,
                branch_id=branch_id,
            )
        )
        if status == "absent" or candidate is None or replay_q_id is None or version is None:
            return CaseRunObservation(
                run_id=run_id,
                branch_id=branch_id,
                subject_ref=None,
                subject_present=False,
                outcome_available=True,
                error=(
                    "NOOP control failed: corresponding explained CandidateMCQ "
                    "was not produced in this replay"
                    if btype == "noop"
                    else None
                ),
            )
        if status == "missing_envelope" or cand_env is None or subject_ref is None:
            return self._inconsistent(
                run_id,
                branch_id,
                f"branch catalog missing candidate envelope {subject_ref}",
            )
        if not payloads_equivalent(cand_env.payload, candidate):
            candidate = cand_env.payload

        decisions = {}
        assessments = {}
        if cs is not None:
            raw_d = getattr(cs, "supervisor_decisions", None) or {}
            decisions = raw_d if isinstance(raw_d, dict) else {}
            raw_a = getattr(cs, "outcome_assessments", None) or {}
            assessments = raw_a if isinstance(raw_a, dict) else {}

        decision_payload = decisions.get(replay_q_id)
        assessment_payload = assessments.get(replay_q_id)

        if decision_payload is None and assessment_payload is None:
            return CaseRunObservation(
                run_id=run_id,
                branch_id=branch_id,
                subject_ref=subject_ref,
                decision=None,
                subject_present=True,
                subject_payload=candidate,
                outcome_available=True,
            )

        if decision_payload is not None:
            dqid = getattr(decision_payload, "q_id", None)
            if dqid is not None and str(dqid) != str(replay_q_id):
                return self._inconsistent(
                    run_id,
                    branch_id,
                    f"decision q_id {dqid!r} does not match branch subject {replay_q_id!r}",
                )
            dver = getattr(decision_payload, "candidate_version", None)
            if dver is not None and int(dver) != int(version):
                return self._inconsistent(
                    run_id,
                    branch_id,
                    f"stale decision candidate_version={dver} for branch {subject_ref}",
                )

        decision_ref = ArtifactRef(f"decision:{replay_q_id}", version)
        assessment_ref = ArtifactRef(f"assessment:{replay_q_id}", version)
        if decision_payload is not None:
            linked = getattr(decision_payload, "assessment_ref", None)
            if linked is not None:
                assessment_ref = linked

        if decision_payload is not None:
            denv = registry.get_by_ref(decision_ref, branch_id) or registry.get_by_ref(
                decision_ref
            )
            if denv is None:
                return self._inconsistent(
                    run_id,
                    branch_id,
                    f"branch catalog missing decision envelope {decision_ref}",
                )

        if assessment_payload is not None:
            aenv = registry.get_by_ref(assessment_ref, branch_id) or registry.get_by_ref(
                assessment_ref
            )
            if aenv is None:
                return self._inconsistent(
                    run_id,
                    branch_id,
                    f"branch catalog missing assessment envelope {assessment_ref}",
                )
            target = getattr(assessment_payload, "target_ref", None)
            if target is not None and target != subject_ref:
                return self._inconsistent(
                    run_id,
                    branch_id,
                    f"stale assessment target {target} for branch {subject_ref}",
                )
            if decision_payload is not None:
                linked = getattr(decision_payload, "assessment_ref", None)
                a_ref = getattr(assessment_payload, "assessment_ref", None)
                if linked is not None and a_ref is not None and linked != a_ref:
                    return self._inconsistent(
                        run_id,
                        branch_id,
                        "decision.assessment_ref does not match assessment.assessment_ref",
                    )

        criterion_values = ()
        if assessment_payload is not None:
            cv = getattr(assessment_payload, "criteria", None)
            if cv:
                criterion_values = tuple(cv)

        decision_value = None
        decision_reasons = ()
        required_fixes = ()
        if decision_payload is not None:
            decision_value = getattr(decision_payload, "decision", None) or getattr(
                decision_payload, "outcome", None
            )
            reasons = getattr(decision_payload, "reasons", None) or ()
            decision_reasons = tuple(reasons)
            fixes = getattr(decision_payload, "required_fixes", None) or ()
            required_fixes = tuple(fixes)

        return CaseRunObservation(
            run_id=run_id,
            branch_id=branch_id,
            subject_ref=subject_ref,
            assessment_ref=assessment_ref if assessment_payload is not None else None,
            decision_ref=decision_ref if decision_payload is not None else None,
            decision=decision_value,
            criterion_values=criterion_values,
            decision_reasons=decision_reasons,
            required_fixes=required_fixes,
            subject_payload=candidate,
            subject_present=True,
            outcome_available=True,
        )

    @staticmethod
    def _inconsistent(run_id: str, branch_id: str, message: str) -> CaseRunObservation:
        return CaseRunObservation(
            run_id=run_id,
            branch_id=branch_id,
            subject_present=False,
            outcome_available=False,
            error=message,
        )
