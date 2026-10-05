"""Target-bounded AXIS replay: observability predicate and stream stop."""

from __future__ import annotations

import unittest
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import MagicMock

from mas_langgraph.artifacts.channel_state import (
    dump_artifact_channels,
    empty_artifact_channels,
    load_artifact_channels,
)
from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.axis.models import ExplainedCase
from mas_langgraph.cases.mcq.artifact_registration import register_mcq_artifacts_for_stage
from mas_langgraph.cases.mcq.comparison import MCQRunObservationAdapter
from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
from mas_langgraph.cases.mcq.replay_observable import (
    ReplayObservableStatus,
    evaluate_mcq_replay_observable,
    mcq_case_replay_observable,
)
from mas_langgraph.core.entities import DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
from mas_langgraph.core.outcomes import OutcomeAssessment
from mas_langgraph.domains.quiz.blueprint_planner import create_mcq_blueprints_from_coverage
from mas_langgraph.domains.quiz.coverage import apply_acceptance_gate_to_all
from mas_langgraph.domains.quiz.cross_cove import run_cross_cove_for_candidate
from mas_langgraph.domains.quiz.entities import (
    AnswerOption,
    CandidateMCQ,
    CrossCoVeReport,
    EvidenceCheckReport,
    SourceRef,
    SupervisorDecision,
)
from mas_langgraph.domains.quiz.generation import register_candidate_mcq
from mas_langgraph.domains.quiz.verification import verify_candidate_against_evidence
from mas_langgraph.replay.executor import LangGraphReplayExecutor, WorkflowRunResult
from mas_langgraph.replay.branches import BranchMetadata


def _seed_state() -> CooperativeState:
    state = CooperativeState()
    state.add_section(
        DocSection(
            section_id="sec_kmeans",
            title="k-Means",
            slide_range=(1, 5),
            importance_weight=1.0,
            content_summary="k-means clustering",
        )
    )
    state.add_learning_objective(
        LearningObjective(
            lo_id="lo_kmeans",
            section_id="sec_kmeans",
            objective_text="Explain k-means assignment.",
            difficulty_level="beginner",
        )
    )
    state.add_evidence_chunk(
        EvidenceChunk(
            chunk_id="chunk_kmeans",
            content="k-means assigns each point to the nearest centroid.",
            metadata={"section_id": "sec_kmeans", "page_number": 2, "pdf_id": "141"},
            quality_score=0.9,
        )
    )
    state.add_knowledge_atom(
        KnowledgeAtom(
            atom_id="atom_kmeans",
            content="k-means assigns each point to the nearest centroid.",
            atom_type="definition",
            citations=[{"chunk_id": "chunk_kmeans", "page": 2}],
            section_id="sec_kmeans",
            lo_id="lo_kmeans",
        )
    )
    state.link_chunk_to_atom("chunk_kmeans", "atom_kmeans")
    return state


def _make_mcq(state: CooperativeState, *, q_id: str = "q_00000001") -> CandidateMCQ:
    bp = create_mcq_blueprints_from_coverage(state)[0]
    state.store_mcq_blueprints([bp])
    mcq = CandidateMCQ(
        q_id=q_id,
        blueprint_id=bp.blueprint_id,
        stem="What does k-means assign each point to?",
        options=[
            AnswerOption(option_id="A", text="Nearest centroid", is_correct=True),
            AnswerOption(option_id="B", text="Random cluster", is_correct=False),
            AnswerOption(option_id="C", text="Previous label", is_correct=False),
            AnswerOption(option_id="D", text="Outlier bucket", is_correct=False),
        ],
        correct_option_id="A",
        answer_explanation="Assignment uses nearest centroid.",
        source_refs=[SourceRef(doc_id="141", page=2, chunk_id="chunk_kmeans")],
        status="draft",
    )
    register_candidate_mcq(state, mcq, bp)
    return mcq


def _explained_for(mcq: CandidateMCQ, *, version: int = 1) -> ExplainedCase:
    return ExplainedCase(
        run_id="run1",
        artifact_ref=ArtifactRef(mcq.q_id, version),
        branch_id="factual",
        decision_ref=ArtifactRef(f"decision:{mcq.q_id}", version),
        outcome="accept",
        correspondence_key=mcq.blueprint_id,
        assessment_ref=ArtifactRef(f"assessment:{mcq.q_id}", version),
    )


def _graph_state_with_catalog(state: CooperativeState, *, run_id: str = "run1") -> dict:
    gstate = {
        "cooperative_state": state,
        "run_id": run_id,
        "branch_id": "factual",
        "step_logs": [],
        "error": None,
        **empty_artifact_channels(),
    }
    reg, resolver = load_artifact_channels(gstate)
    register_mcq_artifacts_for_stage(
        reg, resolver, state, stage="candidates", run_id=run_id, branch_id="factual"
    )
    gstate.update(dump_artifact_channels(reg, resolver))
    return gstate


def _accept_and_register(state: CooperativeState, mcq: CandidateMCQ, *, run_id: str = "run1") -> dict:
    from mas_langgraph.domains.quiz.coverage import apply_acceptance_gate_to_all

    state.store_evidence_report(verify_candidate_against_evidence(mcq, state, llm=None))
    state.store_cross_cove_report(run_cross_cove_for_candidate(mcq, verifier_llm=None))
    # Persist decision + assessment the same way the workflow node does.
    apply_acceptance_gate_to_all(state)
    gstate = {
        "cooperative_state": state,
        "run_id": run_id,
        "branch_id": "factual",
        "step_logs": [],
        "error": None,
        **empty_artifact_channels(),
    }
    reg, resolver = load_artifact_channels(gstate)
    for stage in (
        "candidates",
        "evidence_reports",
        "cross_cove_reports",
        "supervisor_decisions",
        "outcome_assessments",
    ):
        register_mcq_artifacts_for_stage(
            reg,
            resolver,
            state,
            stage=stage,
            run_id=run_id,
            branch_id="factual",
        )
    gstate.update(dump_artifact_channels(reg, resolver))
    gstate["cooperative_state"] = state
    return gstate


class TestMcqReplayObservablePredicate(unittest.TestCase):
    def test_complete_after_terminal_accept(self):
        state = _seed_state()
        mcq = _make_mcq(state)
        gstate = _accept_and_register(state, mcq)
        explained = _explained_for(mcq)
        status = evaluate_mcq_replay_observable(gstate, explained, branch_id="factual")
        self.assertEqual(status, ReplayObservableStatus.COMPLETE)
        self.assertTrue(mcq_case_replay_observable(gstate, explained, branch_id="factual"))

    def test_continue_while_revise(self):
        state = _seed_state()
        mcq = _make_mcq(state)
        gstate = _graph_state_with_catalog(state)
        state.store_supervisor_decision(
            SupervisorDecision(
                q_id=mcq.q_id,
                decision="revise",
                reasons=["needs fix"],
                candidate_version=1,
                required_fixes=["fix stem"],
            )
        )
        state.store_outcome_assessment(
            OutcomeAssessment(
                assessment_ref=ArtifactRef(f"assessment:{mcq.q_id}", 1),
                target_ref=ArtifactRef(mcq.q_id, 1),
                criteria=(),
            )
        )
        explained = _explained_for(mcq)
        status = evaluate_mcq_replay_observable(gstate, explained, branch_id="factual")
        self.assertEqual(status, ReplayObservableStatus.CONTINUE)

    def test_continue_when_candidate_missing(self):
        state = _seed_state()
        explained = ExplainedCase(
            run_id="run1",
            artifact_ref=ArtifactRef("q_missing", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:q_missing", 1),
            outcome="accept",
            correspondence_key="bp_absent",
        )
        gstate = {"cooperative_state": state, **empty_artifact_channels()}
        status = evaluate_mcq_replay_observable(gstate, explained, branch_id="cf_1")
        self.assertEqual(status, ReplayObservableStatus.CONTINUE)

    def test_continue_before_decision(self):
        state = _seed_state()
        mcq = _make_mcq(state)
        gstate = _graph_state_with_catalog(state)
        explained = _explained_for(mcq)
        status = evaluate_mcq_replay_observable(gstate, explained, branch_id="factual")
        self.assertEqual(status, ReplayObservableStatus.CONTINUE)


class TestBoundedVsFullObservationEquivalence(unittest.TestCase):
    def test_observe_replay_identical_for_same_terminal_state(self):
        """Bounded and full share the same final observable state for the target."""
        state = _seed_state()
        mcq = _make_mcq(state)
        gstate = _accept_and_register(state, mcq)
        explained = _explained_for(mcq)

        full = WorkflowRunResult(
            run_id="run1",
            branch_id="cf_full",
            final_state=dict(gstate),
            branch_metadata=BranchMetadata(
                run_id="run1",
                branch_id="cf_full",
                parent_branch_id="factual",
                branch_type="counterfactual",
                origin_checkpoint_alias="a",
                origin_checkpoint_id="c",
            ),
            replay_completion_mode="graph_end",
        )
        bounded = WorkflowRunResult(
            run_id="run1",
            branch_id="cf_bounded",
            final_state=dict(gstate),
            branch_metadata=BranchMetadata(
                run_id="run1",
                branch_id="cf_bounded",
                parent_branch_id="factual",
                branch_type="counterfactual",
                origin_checkpoint_alias="a",
                origin_checkpoint_id="c",
            ),
            replay_completion_mode="target_observable",
            target_observable=True,
            stopped_after_node="apply_acceptance_gate",
            executed_nodes=["generate_candidate_mcqs_from_blueprints", "apply_acceptance_gate"],
        )
        adapter = MCQRunObservationAdapter()
        o_full = adapter.observe_replay(explained_case=explained, run_result=full)
        o_b = adapter.observe_replay(explained_case=explained, run_result=bounded)
        self.assertTrue(o_full.subject_present)
        self.assertTrue(o_b.subject_present)
        self.assertEqual(o_full.decision, o_b.decision)
        self.assertEqual(o_full.subject_ref, o_b.subject_ref)
        self.assertEqual(o_full.decision_reasons, o_b.decision_reasons)
        self.assertEqual(o_full.required_fixes, o_b.required_fixes)
        self.assertEqual(o_full.criterion_values, o_b.criterion_values)
        self.assertEqual(o_full.outcome_available, o_b.outcome_available)
        self.assertIsNone(o_full.error)
        self.assertIsNone(o_b.error)

    def test_revision_v1_not_complete_until_v2_terminal(self):
        state = _seed_state()
        mcq = _make_mcq(state)
        # v1 revise only — incomplete
        mcq.revision_count = 0
        state.store_candidate_mcq(mcq)
        gstate = _graph_state_with_catalog(state)
        state.store_supervisor_decision(
            SupervisorDecision(
                q_id=mcq.q_id,
                decision="revise",
                reasons=["r"],
                candidate_version=1,
            )
        )
        state.store_outcome_assessment(
            OutcomeAssessment(
                assessment_ref=ArtifactRef(f"assessment:{mcq.q_id}", 1),
                target_ref=ArtifactRef(mcq.q_id, 1),
                criteria=(),
            )
        )
        explained = _explained_for(mcq, version=1)
        self.assertEqual(
            evaluate_mcq_replay_observable(gstate, explained, branch_id="factual"),
            ReplayObservableStatus.CONTINUE,
        )

        # Simulate v2 accept terminal
        mcq.revision_count = 1
        mcq.status = "draft"
        state.store_candidate_mcq(mcq)
        state.store_evidence_report(
            EvidenceCheckReport(
                q_id=mcq.q_id,
                supported_by_pdf=True,
                unsupported_claims=[],
                source_refs=mcq.source_refs,
                confidence=0.9,
                candidate_version=2,
                answer_supported=True,
                stem_supported=True,
                distractors_valid=True,
                explanation_supported=True,
            )
        )
        state.store_cross_cove_report(
            CrossCoVeReport(
                q_id=mcq.q_id,
                generator_answer="A",
                verifier_answers=["A", "A", "A"],
                majority_answer="A",
                passed=True,
                candidate_version=2,
                confidence=1.0,
            )
        )
        apply_acceptance_gate_to_all(state)
        # Re-register catalog for v2 terminal artifacts.
        gstate2 = {
            "cooperative_state": state,
            "run_id": "run1",
            "branch_id": "factual",
            "step_logs": [],
            "error": None,
            **empty_artifact_channels(),
        }
        reg, resolver = load_artifact_channels(gstate2)
        for stage in (
            "candidates",
            "evidence_reports",
            "cross_cove_reports",
            "supervisor_decisions",
            "outcome_assessments",
        ):
            register_mcq_artifacts_for_stage(
                reg, resolver, state, stage=stage, run_id="run1", branch_id="factual"
            )
        gstate2.update(dump_artifact_channels(reg, resolver))
        explained_v2 = _explained_for(mcq, version=2)
        self.assertEqual(
            evaluate_mcq_replay_observable(
                gstate2, explained_v2, branch_id="factual"
            ),
            ReplayObservableStatus.COMPLETE,
        )


@dataclass
class _FakeSnap:
    values: Dict[str, Any]


class _FakeCompiledGraph:
    """Minimal stream/invoke stand-in for bounded-replay unit tests."""

    def __init__(self, updates: List[Dict[str, Any]], values_seq: List[Dict[str, Any]]):
        self.updates = updates
        self.values_seq = values_seq
        self.invoke_calls = 0
        self.stream_calls = 0
        self._idx = -1

    def stream(self, input, config=None, context=None, stream_mode=None, **kwargs):
        self.stream_calls += 1
        modes = stream_mode
        if isinstance(modes, str):
            modes = [modes]
        modes = list(modes or ["updates"])
        dual = "updates" in modes and "values" in modes
        if dual:
            yield ("values", dict(self.values_seq[0]))
            for i, upd in enumerate(self.updates):
                self._idx = i
                yield ("updates", upd)
                vals = self.values_seq[min(i, len(self.values_seq) - 1)]
                yield ("values", dict(vals))
            return
        if modes == ["values"]:
            for i, vals in enumerate(self.values_seq):
                self._idx = i
                yield dict(vals)
            return
        for i, upd in enumerate(self.updates):
            self._idx = i
            yield upd

    def invoke(self, input, config=None, context=None, **kwargs):
        self.invoke_calls += 1
        return dict(self.values_seq[-1])

    def get_state(self, config=None):
        # Mimic LangGraph: pinned checkpoint_id returns fork (index 0);
        # thread-only config returns latest.
        configurable = (config or {}).get("configurable") or {}
        if configurable.get("checkpoint_id"):
            i = 0
        else:
            i = max(self._idx, 0)
        i = min(i, len(self.values_seq) - 1)
        return _FakeSnap(values=dict(self.values_seq[i]))

    def get_state_history(self, config=None):
        return []


class TestStreamUntilObservable(unittest.TestCase):
    def test_stops_when_predicate_complete(self):
        state = _seed_state()
        mcq = _make_mcq(state)
        incomplete = deepcopy(_graph_state_with_catalog(state))
        complete = _accept_and_register(state, mcq)
        explained = _explained_for(mcq)
        self.assertEqual(
            evaluate_mcq_replay_observable(incomplete, explained, branch_id="factual"),
            ReplayObservableStatus.CONTINUE,
        )
        self.assertEqual(
            evaluate_mcq_replay_observable(complete, explained, branch_id="factual"),
            ReplayObservableStatus.COMPLETE,
        )

        # First update: incomplete; second: complete → stop before later nodes.
        updates = [
            {"generate_candidate_mcqs_from_blueprints": {}},
            {"apply_acceptance_gate": {}},
            {"generate_candidate_mcqs_from_blueprints": {}},  # should not run
        ]
        # values_seq[i] is state AFTER updates[i]
        values_seq = [incomplete, complete, complete]
        fake = _FakeCompiledGraph(updates, values_seq)

        ex = LangGraphReplayExecutor(compiled_graph=fake)
        # Bypass runtime requirements by calling helper directly.
        raw, final, mode, stopped, nodes, obs, skipped = ex._stream_until_observable(
            fork_config={"configurable": {"thread_id": "run1", "checkpoint_id": "fork0"}},
            runtime=None,
            branch_id="cf_1",
            explained_case=explained,
            stop_predicate=None,
        )
        self.assertEqual(mode, "target_observable")
        self.assertTrue(obs)
        self.assertEqual(stopped, "apply_acceptance_gate")
        self.assertEqual(
            nodes,
            [
                "generate_candidate_mcqs_from_blueprints",
                "apply_acceptance_gate",
            ],
        )
        self.assertEqual(fake.invoke_calls, 0)
        self.assertEqual(fake.stream_calls, 1)
        self.assertNotIn("package_final_quiz", nodes)
        # Must use actual post-node state, not the pinned fork snapshot.
        self.assertTrue(
            evaluate_mcq_replay_observable(final, explained, branch_id="cf_1")
            == ReplayObservableStatus.COMPLETE
            or final.get("cooperative_state") is complete.get("cooperative_state")
        )

    def test_fallback_invokes_to_end(self):
        def boom(*_a, **_k):
            raise RuntimeError("predicate failed")

        fake = _FakeCompiledGraph(
            updates=[{"generate_candidate_mcqs_from_blueprints": {}}],
            values_seq=[{"cooperative_state": CooperativeState(), **empty_artifact_channels()}],
        )
        ex = LangGraphReplayExecutor(compiled_graph=fake)
        explained = ExplainedCase(
            run_id="run1",
            artifact_ref=ArtifactRef("q_x", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:q_x", 1),
            outcome="accept",
        )
        _raw, _final, mode, _stopped, nodes, obs, _skipped = ex._stream_until_observable(
            fork_config={"configurable": {"thread_id": "run1"}},
            runtime=None,
            branch_id="cf_1",
            explained_case=explained,
            stop_predicate=boom,
        )
        self.assertEqual(mode, "fallback_full")
        self.assertFalse(obs)
        self.assertEqual(fake.invoke_calls, 1)
        self.assertGreaterEqual(len(nodes), 1)


class TestAxisExecutorPassesBoundedFlags(unittest.TestCase):
    def test_fork_kwargs_include_explained_case(self):
        from mas_langgraph.axis.executor import AxisQueryExecutor
        from mas_langgraph.axis.models import AxisQueryContext

        replay = MagicMock()
        replay.compiled_graph = object()
        replay.fork_noop.return_value = MagicMock(branch_id="noop_1")
        replay.fork_with_intervention.return_value = MagicMock(
            branch_id="cf_1",
        )
        replay.get_resolver.return_value = MagicMock(is_available=lambda *_: True)

        explained = ExplainedCase(
            run_id="run1",
            artifact_ref=ArtifactRef("q_1", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:q_1", 1),
            outcome="accept",
            correspondence_key="bp_1",
        )
        ctx = MagicMock()
        ctx.explained_case = explained
        ctx.checkpoint_index.get_alias.return_value = MagicMock(
            alias="after_x", checkpoint_id="ckpt1"
        )
        ctx.registry = MagicMock()
        ctx.intervention_registry = MagicMock()
        ctx.contracts = MagicMock()

        # Avoid real ResolvedQuery wiring — call _get_or_run_noop only.
        ex = AxisQueryExecutor(
            context=ctx,
            replay_executor=replay,
            noop_cache=None,
            stop_when_observable=True,
        )
        self.assertTrue(getattr(replay, "target_bounded_replay", False))
        ckpt = MagicMock(alias="after_x", checkpoint_id="ckpt1")
        ex._get_or_run_noop(run_id="run1", ckpt=ckpt)
        kwargs = replay.fork_noop.call_args.kwargs
        self.assertIs(kwargs.get("explained_case"), explained)
        # NOOP control is always full-to-END (Design B).
        self.assertFalse(kwargs.get("stop_when_observable"))

    def test_noop_cache_config_is_full_mode(self):
        from mas_langgraph.axis.executor import _replay_configuration_id

        replay = MagicMock()
        replay.compiled_graph = None
        replay.replay_configuration_id = None
        replay.target_bounded_replay = True
        # NOOP cache fingerprint ignores bounded flag — always full.
        self.assertEqual(_replay_configuration_id(replay, mode="full"), "fixture:full")
        self.assertEqual(_replay_configuration_id(replay), "fixture:full")


class TestRejectAndAmbiguity(unittest.TestCase):
    def test_complete_after_terminal_reject(self):
        state = _seed_state()
        mcq = _make_mcq(state)
        mcq.revision_count = 2
        mcq.status = "draft"
        mcq.flags = ["max_revisions_reached"]
        state.store_candidate_mcq(mcq)
        state.store_evidence_report(
            EvidenceCheckReport(
                q_id=mcq.q_id,
                supported_by_pdf=False,
                unsupported_claims=["stem"],
                source_refs=mcq.source_refs,
                confidence=0.1,
                candidate_version=3,
                answer_supported=False,
                stem_supported=False,
                distractors_valid=False,
                explanation_supported=False,
            )
        )
        state.store_cross_cove_report(
            CrossCoVeReport(
                q_id=mcq.q_id,
                generator_answer="A",
                verifier_answers=["B", "B", "B"],
                majority_answer="B",
                passed=False,
                candidate_version=3,
                confidence=1.0,
            )
        )
        apply_acceptance_gate_to_all(state)
        decision = state.supervisor_decisions[mcq.q_id]
        self.assertEqual(decision.decision, "reject")
        gstate = {
            "cooperative_state": state,
            "run_id": "run1",
            "branch_id": "factual",
            "step_logs": [],
            "error": None,
            **empty_artifact_channels(),
        }
        reg, resolver = load_artifact_channels(gstate)
        for stage in (
            "candidates",
            "evidence_reports",
            "cross_cove_reports",
            "supervisor_decisions",
            "outcome_assessments",
        ):
            register_mcq_artifacts_for_stage(
                reg, resolver, state, stage=stage, run_id="run1", branch_id="factual"
            )
        gstate.update(dump_artifact_channels(reg, resolver))
        explained = _explained_for(mcq, version=3)
        explained = ExplainedCase(
            run_id=explained.run_id,
            artifact_ref=ArtifactRef(mcq.q_id, 3),
            branch_id="factual",
            decision_ref=ArtifactRef(f"decision:{mcq.q_id}", 3),
            outcome="reject",
            correspondence_key=mcq.blueprint_id,
            assessment_ref=ArtifactRef(f"assessment:{mcq.q_id}", 3),
        )
        self.assertEqual(
            evaluate_mcq_replay_observable(gstate, explained, branch_id="factual"),
            ReplayObservableStatus.COMPLETE,
        )

    def test_ambiguous_correspondence_falls_back(self):
        state = _seed_state()
        mcq_a = _make_mcq(state, q_id="q_00000001")
        bp_id = mcq_a.blueprint_id
        mcq_b = CandidateMCQ(
            q_id="q_00000002",
            blueprint_id=bp_id,
            stem="Other stem?",
            options=mcq_a.options,
            correct_option_id="A",
            answer_explanation="x",
            source_refs=mcq_a.source_refs,
            status="draft",
        )
        state.store_candidate_mcq(mcq_b)
        explained = ExplainedCase(
            run_id="run1",
            artifact_ref=ArtifactRef("q_gone", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:q_gone", 1),
            outcome="accept",
            correspondence_key=bp_id,
        )
        gstate = {"cooperative_state": state, **empty_artifact_channels()}
        self.assertEqual(
            evaluate_mcq_replay_observable(gstate, explained, branch_id="cf"),
            ReplayObservableStatus.FALLBACK_FULL,
        )

    def test_complete_requires_target_artifact_envelopes(self):
        state = _seed_state()
        mcq = _make_mcq(state)
        gstate = _accept_and_register(state, mcq)
        explained = _explained_for(mcq)
        reg, _ = load_artifact_channels(gstate)
        branch_id = "factual"
        for lid in (
            mcq.q_id,
            f"decision:{mcq.q_id}",
            f"assessment:{mcq.q_id}",
            f"evidence-report:{mcq.q_id}",
            f"cross-cove-report:{mcq.q_id}",
        ):
            ref = ArtifactRef(lid, 1)
            env = reg.get_by_ref(ref, branch_id) or reg.get_by_ref(ref)
            self.assertIsNotNone(env, msg=f"missing {ref}")
        self.assertEqual(
            evaluate_mcq_replay_observable(gstate, explained, branch_id=branch_id),
            ReplayObservableStatus.COMPLETE,
        )


class TestBoundedVsFullComparator(unittest.TestCase):
    def test_replay_comparator_same_classification(self):
        from mas_langgraph.axis.comparison import ReplayComparator, QuerySignature

        state = _seed_state()
        mcq = _make_mcq(state)
        gstate = _accept_and_register(state, mcq)
        explained = _explained_for(mcq)
        adapter = MCQRunObservationAdapter()
        registry, _ = load_artifact_channels(gstate)
        factual = adapter.observe_factual(
            explained_case=explained,
            case_slice=MagicMock(
                assessment_ref=explained.assessment_ref,
                decision=explained.outcome,
                criterion_values=(),
                decision_reasons=(),
                required_fixes=(),
            ),
            registry=registry,
        )
        full = WorkflowRunResult(
            run_id="run1",
            branch_id="cf_full",
            final_state=dict(gstate),
            replay_completion_mode="graph_end",
        )
        bounded = WorkflowRunResult(
            run_id="run1",
            branch_id="cf_bounded",
            final_state=dict(gstate),
            replay_completion_mode="target_observable",
            target_observable=True,
            executed_nodes=["apply_acceptance_gate"],
        )
        sig = QuerySignature(operation="REMOVE", target_ref=ArtifactRef("E1", 1))
        comp = ReplayComparator()
        r_full = comp.compare(
            signature=sig,
            factual_observation=factual,
            noop_result=full,
            counterfactual_result=full,
            observation_adapter=adapter,
            explained_case=explained,
        )
        r_b = comp.compare(
            signature=sig,
            factual_observation=factual,
            noop_result=bounded,
            counterfactual_result=bounded,
            observation_adapter=adapter,
            explained_case=explained,
        )
        self.assertEqual(r_full.evidence_status, r_b.evidence_status)
        self.assertEqual(r_full.noop.decision, r_b.noop.decision)
        self.assertEqual(r_full.counterfactual.decision, r_b.counterfactual.decision)
        self.assertEqual(r_full.noop.criterion_values, r_b.noop.criterion_values)
        full_content = getattr(r_full.subject_content_delta, "effect_status", None)
        b_content = getattr(r_b.subject_content_delta, "effect_status", None)
        self.assertEqual(full_content, b_content)


if __name__ == "__main__":
    unittest.main()
