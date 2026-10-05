"""Current-version correspondence for revised counterfactual MCQs."""

from __future__ import annotations

import contextvars
import unittest
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Optional
from unittest.mock import patch

from mas_langgraph.artifacts.channel_state import (
    dump_artifact_channels,
    load_artifact_channels,
)
from mas_langgraph.artifacts.models import ArtifactEnvelope, ArtifactRef
from mas_langgraph.axis.comparison import (
    DOWNSTREAM_EFFECT,
    INTERVENTION_ERROR,
    CaseRunObservation,
    ReplayComparator,
    _determine_evidence_status,
)
from mas_langgraph.axis.models import ExplainedCase, QuerySignature
from mas_langgraph.cases.mcq.comparison import (
    MCQRunObservationAdapter,
    resolve_current_branch_subject,
)
from mas_langgraph.experiment.information_role import (
    ROLE_INCONCLUSIVE,
    ROLE_NECESSARY_FOR_REPRODUCTION,
    build_information_role_finding,
)
from mas_langgraph.replay.executor import WorkflowRunResult
from mas_langgraph.tests.test_bounded_replay import (
    _accept_and_register,
    _explained_for,
    _make_mcq,
    _seed_state,
)
from mas_langgraph.cases.mcq.replay_observable import (
    ReplayObservableStatus,
    evaluate_mcq_replay_observable,
)


@dataclass
class _Cand:
    q_id: str
    revision_count: int = 0
    stem: str = "stem"
    blueprint_id: str = "bp_A"


@dataclass
class _Decision:
    q_id: str
    decision: str
    reasons: list
    required_fixes: list
    candidate_version: int
    assessment_ref: Optional[ArtifactRef] = None


@dataclass
class _Assessment:
    assessment_ref: ArtifactRef
    target_ref: ArtifactRef
    criteria: tuple = ()


@dataclass
class _Report:
    q_id: str
    candidate_version: int = 1


@dataclass
class _CS:
    candidate_mcqs: dict
    supervisor_decisions: dict = field(default_factory=dict)
    outcome_assessments: dict = field(default_factory=dict)
    evidence_reports: dict = field(default_factory=dict)
    cross_cove_reports: dict = field(default_factory=dict)


def _explained(q_id="q_00000004", version=1, key="bp_A"):
    return ExplainedCase(
        run_id="run1",
        artifact_ref=ArtifactRef(q_id, version),
        branch_id="factual",
        decision_ref=ArtifactRef(f"decision:{q_id}", version),
        outcome="accept",
        correspondence_key=key,
        assessment_ref=ArtifactRef(f"assessment:{q_id}", version),
    )


def _run(cs, catalog, branch="cf"):
    return WorkflowRunResult(
        run_id="run1",
        branch_id=branch,
        final_state={
            "cooperative_state": cs,
            "artifact_catalog": catalog,
            "branch_id": branch,
        },
    )


def _env(logical_id, version, payload, typ="CandidateMCQ", branch="cf"):
    return ArtifactEnvelope.create(
        logical_id=logical_id,
        version=version,
        artifact_type=typ,
        payload=payload,
        branch_id=branch,
        created_in_run="run1",
    )


class TestCurrentVersionCorrespondence(unittest.TestCase):
    def test_valid_revision_uses_current_v2(self):
        cand = _Cand("q_00000004", revision_count=1, blueprint_id="bp_A")
        v1 = _Cand("q_00000004", revision_count=0, blueprint_id="bp_A", stem="old")
        cenv1 = _env("q_00000004", 1, v1)
        cenv2 = _env("q_00000004", 2, cand)
        aref = ArtifactRef("assessment:q_00000004", 2)
        assess = _Assessment(aref, ArtifactRef("q_00000004", 2))
        dec = _Decision("q_00000004", "accept", ["ok"], [], 2, assessment_ref=aref)
        aenv = _env("assessment:q_00000004", 2, assess, "OutcomeAssessment")
        denv = _env("decision:q_00000004", 2, dec, "SupervisorDecision")
        ev = _env("evidence-report:q_00000004", 2, _Report("q_00000004", 2), "EvidenceCheckReport")
        cove = _env(
            "cross-cove-report:q_00000004", 2, _Report("q_00000004", 2), "CrossCoVeReport"
        )
        catalog = {
            e.instance_id: e for e in (cenv1, cenv2, aenv, denv, ev, cove)
        }
        obs = MCQRunObservationAdapter().observe_replay(
            explained_case=_explained(),
            run_result=_run(
                _CS(
                    candidate_mcqs={"q_00000004": cand},
                    supervisor_decisions={"q_00000004": dec},
                    outcome_assessments={"q_00000004": assess},
                    evidence_reports={"q_00000004": _Report("q_00000004", 2)},
                    cross_cove_reports={"q_00000004": _Report("q_00000004", 2)},
                ),
                catalog,
            ),
        )
        self.assertTrue(obs.subject_present)
        self.assertTrue(obs.outcome_available)
        self.assertIsNone(obs.error)
        self.assertEqual(obs.subject_ref, ArtifactRef("q_00000004", 2))
        self.assertNotEqual(obs.subject_ref, ArtifactRef("q_00000004", 1))
        self.assertEqual(obs.subject_payload.revision_count, 1)

    def test_historical_v1_does_not_override_current_v2(self):
        cand_v2 = _Cand("q_00000004", revision_count=1, blueprint_id="bp_A", stem="v2-stem")
        cand_v1 = _Cand("q_00000004", revision_count=0, blueprint_id="bp_A", stem="v1-stem")
        catalog = {
            e.instance_id: e
            for e in (
                _env("q_00000004", 1, cand_v1),
                _env("q_00000004", 2, cand_v2),
            )
        }
        obs = MCQRunObservationAdapter().observe_replay(
            explained_case=_explained(),
            run_result=_run(_CS(candidate_mcqs={"q_00000004": cand_v2}), catalog),
        )
        self.assertEqual(obs.subject_ref.version, 2)
        self.assertEqual(obs.subject_payload.stem, "v2-stem")
        self.assertIsNone(obs.error)

    def test_stale_decision_v2_candidate_v1_decision(self):
        cand = _Cand("q_00000004", revision_count=1, blueprint_id="bp_A")
        cenv = _env("q_00000004", 2, cand)
        dec = _Decision("q_00000004", "accept", [], [], 1)
        obs = MCQRunObservationAdapter().observe_replay(
            explained_case=_explained(),
            run_result=_run(
                _CS(candidate_mcqs={"q_00000004": cand}, supervisor_decisions={"q_00000004": dec}),
                {cenv.instance_id: cenv},
            ),
        )
        self.assertFalse(obs.outcome_available)
        self.assertIsNotNone(obs.error)
        self.assertIn("stale decision", obs.error or "")
        self.assertFalse(obs.subject_present)

    def test_stale_assessment_targets_v1(self):
        cand = _Cand("q_00000004", revision_count=1, blueprint_id="bp_A")
        cenv = _env("q_00000004", 2, cand)
        aref = ArtifactRef("assessment:q_00000004", 2)
        assess = _Assessment(aref, ArtifactRef("q_00000004", 1))
        aenv = _env("assessment:q_00000004", 2, assess, "OutcomeAssessment")
        obs = MCQRunObservationAdapter().observe_replay(
            explained_case=_explained(),
            run_result=_run(
                _CS(
                    candidate_mcqs={"q_00000004": cand},
                    outcome_assessments={"q_00000004": assess},
                ),
                {cenv.instance_id: cenv, aenv.instance_id: aenv},
            ),
        )
        self.assertFalse(obs.outcome_available)
        self.assertIn("stale assessment", obs.error or "")

    def test_inconsistency_is_not_disappearance(self):
        cf = CaseRunObservation(
            run_id="r",
            branch_id="cf",
            subject_present=False,
            outcome_available=False,
            error="stale decision candidate_version=2 for branch q_00000004:v1",
        )
        status, _, _ = _determine_evidence_status(
            CaseRunObservation(
                run_id="r", branch_id="factual", decision="accept", subject_present=True
            ),
            CaseRunObservation(
                run_id="r",
                branch_id="noop",
                decision="accept",
                subject_present=True,
                outcome_available=True,
            ),
            cf,
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            (),
        )
        self.assertEqual(status, INTERVENTION_ERROR)
        self.assertNotEqual(status, DOWNSTREAM_EFFECT)
        finding = build_information_role_finding({"evidence_status": status})
        self.assertIsNotNone(finding)
        self.assertEqual(finding.role_type, ROLE_INCONCLUSIVE)
        self.assertNotEqual(finding.role_type, ROLE_NECESSARY_FOR_REPRODUCTION)
        self.assertFalse(finding.diagnostics()["information_role_supported"])

    def test_genuine_absence_has_no_error(self):
        obs = MCQRunObservationAdapter().observe_replay(
            explained_case=_explained(),
            run_result=_run(_CS(candidate_mcqs={}), {}),
        )
        self.assertFalse(obs.subject_present)
        self.assertIsNone(obs.error)
        self.assertTrue(obs.outcome_available)
        status, _, _ = _determine_evidence_status(
            CaseRunObservation(
                run_id="r", branch_id="factual", decision="accept", subject_present=True
            ),
            CaseRunObservation(
                run_id="r",
                branch_id="noop",
                decision="accept",
                subject_present=True,
                outcome_available=True,
            ),
            obs,
            (),
            (),
            (),
            (),
            (),
            (),
            (),
            (),
        )
        self.assertEqual(status, DOWNSTREAM_EFFECT)


class TestBoundedReplayCurrentVersion(unittest.TestCase):
    def test_terminal_v2_is_complete_for_explained_v1(self):
        state = _seed_state()
        mcq = _make_mcq(state)
        explained = _explained_for(mcq, version=1)
        v1_payload = deepcopy(mcq)
        mcq.revision_count = 1
        mcq.stem = "What is a revised k-means assignment question?"
        gstate = _accept_and_register(state, mcq, run_id="run1")
        gstate["branch_id"] = "cf"
        reg, resolver = load_artifact_channels(gstate)
        v1_env = ArtifactEnvelope.create(
            logical_id=mcq.q_id,
            version=1,
            artifact_type="CandidateMCQ",
            payload=v1_payload,
            branch_id="cf",
            created_in_run="run1",
        )
        catalog = dict(gstate.get("artifact_catalog") or {})
        catalog[v1_env.instance_id] = v1_env
        gstate["artifact_catalog"] = catalog
        gstate.update(dump_artifact_channels(reg, resolver))
        gstate["artifact_catalog"][v1_env.instance_id] = v1_env

        status = evaluate_mcq_replay_observable(gstate, explained, branch_id="cf")
        self.assertEqual(status, ReplayObservableStatus.COMPLETE)

        obs = MCQRunObservationAdapter().observe_replay(
            explained_case=explained,
            run_result=WorkflowRunResult(
                run_id="run1",
                branch_id="cf",
                final_state=gstate,
            ),
        )
        self.assertTrue(obs.subject_present)
        self.assertEqual(obs.subject_ref, ArtifactRef(mcq.q_id, 2))
        self.assertIsNone(obs.error)
        self.assertTrue(obs.outcome_available)


_CF_BRANCH = contextvars.ContextVar("cv_corr_cf_branch", default="")


def _catalog_env(registry, ref, branch_id=""):
    return registry.get_by_ref(ref, branch_id) or registry.get_by_ref(ref)


def _grounded_stem_and_key(_context):
    """Stem/answer overlap the seeded k-means evidence so factual v1 can accept."""
    return {
        "stem": "What happens when k-means assigns points to nearest centroids?",
        "correct_text": "k-means assigns points to nearest centroids then updates them.",
    }


def _passing_evidence_report(candidate, *, version: int) -> Any:
    from mas_langgraph.domains.quiz.entities import EvidenceCheckReport
    from mas_langgraph.domains.quiz.verification import validate_evidence_report_structure

    return validate_evidence_report_structure(
        EvidenceCheckReport(
            q_id=candidate.q_id,
            supported_by_pdf=True,
            unsupported_claims=[],
            source_refs=list(getattr(candidate, "source_refs", None) or []),
            confidence=0.9,
            candidate_version=int(version),
            answer_supported=True,
            stem_supported=True,
            distractors_valid=True,
            explanation_supported=True,
            required_fixes=[],
            notes=["Deterministic passing evidence report for corresponding v2."],
        )
    )


def _force_cf_v1_revision(*, correspondence_key: str, q_id: str):
    """Fail only the explained corresponding v1 on a counterfactual branch.

    The compiled graph already closed over production node functions, so this
    patches the module globals those nodes look up at call time.
    """
    from mas_langgraph.domains.quiz.entities import EvidenceCheckReport
    from mas_langgraph.domains.quiz.verification import (
        apply_candidate_status_from_report,
        validate_evidence_report_structure,
        verify_all_candidate_mcqs as real_verify_all,
        verify_candidate_against_evidence,
    )
    from mas_langgraph.domains.quiz.versioning import candidate_version_number
    from mas_langgraph.orchestrator import langgraph_workflow as lg

    real_prepare = lg.prepare_state_transition

    def tracking_prepare(state):
        try:
            _CF_BRANCH.set(str((state or {}).get("branch_id") or ""))
        except Exception:
            pass
        return real_prepare(state)

    def scripted_verify_all(cs, llm=None, max_candidates=None, *, force=False):
        branch = str(_CF_BRANCH.get() or "")
        if not branch.startswith("cf"):
            return real_verify_all(
                cs, llm=llm, max_candidates=max_candidates, force=force
            )
        reports = []
        candidates = list(cs.candidate_mcqs.values())
        if max_candidates is not None:
            candidates = candidates[:max_candidates]
        from mas_langgraph.domains.quiz.versioning import report_targets_candidate

        for candidate in candidates:
            version = candidate_version_number(candidate)
            bp = str(getattr(candidate, "blueprint_id", "") or "")
            is_target = str(candidate.q_id) == str(q_id) or (
                correspondence_key and bp == str(correspondence_key)
            )
            existing = cs.evidence_reports.get(candidate.q_id)
            if (
                not force
                and existing is not None
                and report_targets_candidate(existing, candidate)
            ):
                reports.append(existing)
                continue
            if is_target and version == 1:
                report = validate_evidence_report_structure(
                    EvidenceCheckReport(
                        q_id=candidate.q_id,
                        supported_by_pdf=False,
                        unsupported_claims=[
                            "scripted first-cycle CF evidence check requires revision"
                        ],
                        source_refs=list(getattr(candidate, "source_refs", None) or []),
                        confidence=0.2,
                        candidate_version=1,
                        answer_supported=True,
                        stem_supported=False,
                        distractors_valid=True,
                        explanation_supported=True,
                        required_fixes=["clarify_stem"],
                        notes=[
                            "Deterministic integration test: force corresponding v1 "
                            "through revise_candidate_mcqs."
                        ],
                    )
                )
                apply_candidate_status_from_report(candidate, report)
                cs.store_evidence_report(report)
                cs.store_candidate_mcq(candidate)
                reports.append(report)
                continue
            if is_target and version == 2:
                report = _passing_evidence_report(candidate, version=2)
                apply_candidate_status_from_report(candidate, report)
                cs.store_evidence_report(report)
                cs.store_candidate_mcq(candidate)
                reports.append(report)
                continue
            report = verify_candidate_against_evidence(candidate, cs, llm=llm)
            cs.store_evidence_report(report)
            cs.store_candidate_mcq(candidate)
            reports.append(report)
        return reports

    return (
        patch.object(lg, "prepare_state_transition", tracking_prepare),
        patch.object(lg, "verify_all_candidate_mcqs", scripted_verify_all),
    )


class TestLiveReplayTracksRevisedCorrespondingSubject(unittest.TestCase):
    """Real LangGraph/checkpointer path: factual v1, CF revise to v2."""

    def test_live_replay_tracks_revised_corresponding_subject_v1_to_v2(self):
        from mas_langgraph.artifacts.channel_state import load_artifact_channels
        from mas_langgraph.cases.mcq.explained_case import MCQExplainedCaseResolver
        from mas_langgraph.domains.quiz.versioning import (
            candidate_version_number,
            get_operational_cross_cove_report,
            get_operational_evidence_report,
            get_operational_supervisor_decision,
        )
        from mas_langgraph.replay.branch_artifacts import artifact_exists_at_checkpoint
        from mas_langgraph.replay.checkpoints import BoundarySelector
        from mas_langgraph.replay.interventions import ArtifactIntervention
        from mas_langgraph.tests.test_prompt16_end_to_end import (
            _build_seeded_factual_run,
            _seeded_cooperative_state,
        )
        from mas_langgraph.tests.test_replay_state_tracking import (
            _accepted_explained,
            _origin_before_generation,
        )

        cs0 = _seeded_cooperative_state()
        cs0._id_counters = dict(getattr(cs0, "_id_counters", None) or {})
        cs0._id_counters["q"] = 3
        from mas_langgraph.domains.quiz import generation as generation_mod

        with patch.object(
            generation_mod,
            "generate_stem_and_key_from_blueprint",
            _grounded_stem_and_key,
        ):
            _case, wex, factual = _build_seeded_factual_run(
                run_id="run_cv_live_v1v2",
                cooperative_state=cs0,
            )
        self.assertIsNone(factual.error)

        resolver = MCQExplainedCaseResolver()
        refs = resolver.list_explainable_artifacts(factual_run_result=factual)
        q4 = ArtifactRef("q_00000004", 1)
        explained = None
        if any(r == q4 for r in refs):
            explained = resolver.resolve(factual_run_result=factual, artifact_ref=q4)
        if explained is None:
            explained = _accepted_explained(factual)
        self.assertEqual(explained.artifact_ref, ArtifactRef("q_00000004", 1))
        self.assertEqual(explained.outcome, "accept")
        self.assertTrue(explained.correspondence_key)

        factual_cs = factual.final_state.get("cooperative_state")
        factual_cand = (getattr(factual_cs, "candidate_mcqs", None) or {}).get(
            explained.artifact_ref.logical_id
        )
        self.assertIsNotNone(factual_cand)
        self.assertEqual(candidate_version_number(factual_cand), 1)
        self.assertEqual(factual_cand.blueprint_id, explained.correspondence_key)

        adapter = MCQRunObservationAdapter()
        factual_obs = adapter.observe_factual(
            explained_case=explained,
            case_slice=type(
                "S",
                (),
                {
                    "assessment_ref": explained.assessment_ref,
                    "decision": explained.outcome,
                    "criterion_values": (),
                    "decision_reasons": (),
                    "required_fixes": (),
                },
            )(),
            registry=factual.artifact_registry,
        )
        self.assertTrue(factual_obs.subject_present)
        self.assertTrue(factual_obs.outcome_available)
        self.assertEqual(factual_obs.subject_ref, ArtifactRef("q_00000004", 1))
        self.assertEqual(str(factual_obs.decision or "").lower(), "accept")

        replay = wex.replay_executor
        origin = _origin_before_generation(factual, replay)
        origin_snap = replay.get_checkpoint_snapshot(
            factual.run_id, BoundarySelector.by_alias(origin.alias)
        )
        origin_values = getattr(origin_snap, "values", None) or {}
        target = ArtifactRef("E18", 1)
        if not artifact_exists_at_checkpoint(origin_values, target):
            target = ArtifactRef("E17", 1)
        self.assertTrue(
            artifact_exists_at_checkpoint(origin_values, target),
            msg=f"expected evidence artifact at origin {origin.alias}",
        )
        intervention = ArtifactIntervention(
            operation="REMOVE",
            artifact=target,
            checkpoint=origin,
            description="deterministic CF fork before generation",
        )

        patches = _force_cf_v1_revision(
            correspondence_key=str(explained.correspondence_key),
            q_id=explained.artifact_ref.logical_id,
        )
        with patch.object(
            generation_mod,
            "generate_stem_and_key_from_blueprint",
            _grounded_stem_and_key,
        ), patches[0], patches[1]:
            noop = replay.fork_noop(
                factual.run_id,
                BoundarySelector.by_alias(origin.alias),
                explained_case=explained,
                stop_when_observable=True,
            )
            bounded = replay.fork_with_intervention(
                factual.run_id,
                intervention,
                artifact_type="EvidenceChunk",
                explained_case=explained,
                stop_when_observable=True,
            )
            full = replay.fork_with_intervention(
                factual.run_id,
                intervention,
                artifact_type="EvidenceChunk",
                explained_case=explained,
                stop_when_observable=False,
            )

        self.assertIsNone(noop.error, msg=noop.error)
        self.assertIsNone(bounded.error, msg=bounded.error)
        self.assertIsNone(full.error, msg=full.error)

        self.assertIn("revise_candidate_mcqs", bounded.executed_nodes)
        self.assertEqual(bounded.replay_completion_mode, "target_observable")
        self.assertEqual(bounded.stopped_after_node, "apply_acceptance_gate")
        self.assertTrue(bounded.target_observable)
        self.assertNotEqual(bounded.replay_completion_mode, "graph_end")
        self.assertEqual(full.replay_completion_mode, "graph_end")

        b_cs = bounded.final_state.get("cooperative_state")
        b_reg, _ = load_artifact_channels(bounded.final_state)
        cand, q_id, version, subject_ref, cand_env, status = resolve_current_branch_subject(
            explained_case=explained,
            candidates=getattr(b_cs, "candidate_mcqs", None) or {},
            registry=b_reg,
            branch_id=bounded.branch_id,
        )
        self.assertEqual(status, "ok")
        self.assertEqual(q_id, "q_00000004")
        self.assertEqual(version, 2)
        self.assertEqual(subject_ref, ArtifactRef("q_00000004", 2))
        self.assertEqual(candidate_version_number(cand), 2)
        self.assertEqual(cand.blueprint_id, explained.correspondence_key)
        self.assertEqual(factual_cand.blueprint_id, cand.blueprint_id)
        self.assertNotEqual(ArtifactRef("q_00000004", 1), ArtifactRef("q_00000004", 2))

        env_v1 = _catalog_env(b_reg, ArtifactRef("q_00000004", 1), bounded.branch_id)
        env_v2 = _catalog_env(b_reg, ArtifactRef("q_00000004", 2), bounded.branch_id)
        self.assertIsNotNone(env_v1)
        self.assertIsNotNone(env_v2)
        self.assertIsNotNone(cand_env)
        self.assertEqual(cand_env.ref, ArtifactRef("q_00000004", 2))
        self.assertNotEqual(env_v1.ref, env_v2.ref)
        self.assertEqual(getattr(env_v1.payload, "revision_count", None), 0)
        self.assertEqual(getattr(env_v2.payload, "revision_count", None), 1)

        decision = get_operational_supervisor_decision(b_cs, cand)
        assessment = (getattr(b_cs, "outcome_assessments", None) or {}).get(q_id)
        evidence = get_operational_evidence_report(b_cs, cand)
        cove = get_operational_cross_cove_report(b_cs, cand)
        self.assertIsNotNone(decision)
        self.assertEqual(int(decision.candidate_version), 2)
        self.assertEqual(str(decision.decision or "").lower(), "accept")
        self.assertIsNotNone(assessment)
        self.assertEqual(assessment.target_ref, ArtifactRef("q_00000004", 2))
        self.assertIsNotNone(evidence)
        self.assertEqual(int(evidence.candidate_version), 2)
        self.assertIsNotNone(cove)
        self.assertEqual(int(cove.candidate_version), 2)
        self.assertTrue(
            any(
                getattr(rec, "q_id", None) == q_id
                for rec in (getattr(b_cs, "revision_records", None) or [])
            )
            or int(getattr(cand, "revision_count", 0) or 0) == 1
        )

        cf_obs = adapter.observe_replay(explained_case=explained, run_result=bounded)
        self.assertTrue(cf_obs.subject_present)
        self.assertTrue(cf_obs.outcome_available)
        self.assertIsNone(cf_obs.error)
        self.assertEqual(cf_obs.subject_ref, ArtifactRef("q_00000004", 2))
        self.assertNotEqual(cf_obs.subject_ref, ArtifactRef("q_00000004", 1))
        self.assertNotEqual(cf_obs.subject_ref, factual_obs.subject_ref)

        status_obs = evaluate_mcq_replay_observable(
            bounded.final_state, explained, branch_id=bounded.branch_id
        )
        self.assertEqual(status_obs, ReplayObservableStatus.COMPLETE)

        comparison = ReplayComparator().compare(
            signature=QuerySignature(
                operation="REMOVE",
                target_ref=target,
                checkpoint_id=origin.checkpoint_id,
                checkpoint_alias=origin.alias,
            ),
            factual_observation=factual_obs,
            noop_result=noop,
            counterfactual_result=bounded,
            observation_adapter=adapter,
            explained_case=explained,
        )
        self.assertEqual(comparison.factual.subject_ref, ArtifactRef("q_00000004", 1))
        self.assertEqual(comparison.counterfactual.subject_ref, ArtifactRef("q_00000004", 2))
        self.assertTrue(comparison.counterfactual.subject_present)
        self.assertTrue(comparison.counterfactual.outcome_available)
        self.assertIsNone(comparison.counterfactual.error)
        self.assertIsNone(comparison.noop.error)
        self.assertTrue(comparison.noop.subject_present)

        full_obs = adapter.observe_replay(explained_case=explained, run_result=full)
        self.assertEqual(full_obs.subject_ref, ArtifactRef("q_00000004", 2))
        self.assertEqual(full_obs.subject_ref, cf_obs.subject_ref)
        self.assertEqual(
            str(full_obs.decision or "").lower(),
            str(cf_obs.decision or "").lower(),
        )
        self.assertEqual(
            getattr(full_obs.subject_payload, "stem", None),
            getattr(cf_obs.subject_payload, "stem", None),
        )

        stale_dec = replace(decision, candidate_version=1)
        catalog = {
            env_v1.instance_id: env_v1,
            env_v2.instance_id: env_v2,
        }
        stale_obs = adapter.observe_replay(
            explained_case=explained,
            run_result=_run(
                _CS(
                    candidate_mcqs={q_id: cand},
                    supervisor_decisions={q_id: stale_dec},
                ),
                catalog,
                branch=bounded.branch_id,
            ),
        )
        self.assertIsNotNone(stale_obs.error)
        self.assertIn("stale decision", stale_obs.error or "")
        self.assertFalse(stale_obs.outcome_available)


if __name__ == "__main__":
    unittest.main()

