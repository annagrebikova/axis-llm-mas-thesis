"""Regression: NOOP/CF replay must observe the *latest* branch state.

Live failure (PDF-141): stream used get_state(fork_config) with a pinned
checkpoint_id, so observe_replay saw the pre-generation fork snapshot even
after package_final_quiz executed — yielding noop_error.
"""

from __future__ import annotations

import operator
import unittest
from typing import Annotated, Any, Dict, TypedDict

from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.axis.models import ExplainedCase
from mas_langgraph.cases.mcq.comparison import (
    MCQRunObservationAdapter,
    resolve_corresponding_candidate,
)
from mas_langgraph.cases.mcq.explained_case import MCQExplainedCaseResolver
from mas_langgraph.experiment.information_role import (
    ROLE_INCONCLUSIVE,
    build_information_role_finding,
)
from mas_langgraph.replay.checkpoints import BoundarySelector
from mas_langgraph.tests.test_prompt16_end_to_end import _build_seeded_factual_run


def _accepted_explained(factual):
    refs = MCQExplainedCaseResolver().list_explainable_artifacts(
        factual_run_result=factual
    )
    assert refs, "expected explainable CandidateMCQ refs"
    # Prefer the terminal accepted/rejected version (may follow a revise cycle).
    for ref in reversed(refs):
        explained = MCQExplainedCaseResolver().resolve(
            factual_run_result=factual, artifact_ref=ref
        )
        if str(explained.outcome or "").lower() in {"accept", "reject"}:
            return explained
    return MCQExplainedCaseResolver().resolve(
        factual_run_result=factual, artifact_ref=refs[-1]
    )


def _origin_before_generation(factual, replay_executor):
    idx = factual.checkpoint_index
    for rec in idx.records:
        if "generate_candidate_mcqs_from_blueprints" in (rec.next_nodes or ()):
            return rec
    for rec in idx.records:
        snap = replay_executor.get_checkpoint_snapshot(
            factual.run_id, BoundarySelector.by_alias(rec.alias)
        )
        vals = getattr(snap, "values", None) or {}
        cs = vals.get("cooperative_state")
        cands = getattr(cs, "candidate_mcqs", None) or {}
        if not cands:
            return rec
    raise AssertionError("no suitable origin checkpoint before MCQ generation")


class TestPinnedForkStateIsStale(unittest.TestCase):
    """LangGraph API contract used by the live failure diagnosis."""

    def test_get_state_with_checkpoint_id_stays_at_fork(self):
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph

        class S(TypedDict):
            n: int
            path: Annotated[list, operator.add]

        def a(s: S) -> dict:
            return {"n": s["n"] + 1, "path": ["a"]}

        def b(s: S) -> dict:
            return {"n": s["n"] + 10, "path": ["b"]}

        g = StateGraph(S)
        g.add_node("a", a)
        g.add_node("b", b)
        g.add_edge(START, "a")
        g.add_edge("a", "b")
        g.add_edge("b", END)
        app = g.compile(checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "stale_fork"}}
        app.invoke({"n": 0, "path": []}, config=cfg)
        hist = list(app.get_state_history(cfg))
        after_a = next(h for h in hist if h.next == ("b",))
        fork = app.update_state(
            after_a.config, {"n": after_a.values["n"], "path": ["fork"]}
        )
        self.assertIn("checkpoint_id", fork["configurable"])
        for _ in app.stream(None, config=fork, stream_mode="updates"):
            pinned = app.get_state(fork).values
            self.assertEqual(pinned["path"], ["a", "fork"])
        pinned_final = app.get_state(fork).values
        latest = app.get_state({"configurable": {"thread_id": "stale_fork"}}).values
        self.assertEqual(pinned_final["path"], ["a", "fork"])
        self.assertIn("b", latest["path"])
        self.assertNotEqual(pinned_final, latest)


class TestRealGraphNoopObservation(unittest.TestCase):
    def test_noop_observe_replay_finds_target_after_bounded_stream(self):
        _case, wex, factual = _build_seeded_factual_run(run_id="run_noop_obs")
        self.assertIsNone(factual.error)
        explained = _accepted_explained(factual)
        self.assertTrue(explained.correspondence_key)
        origin = _origin_before_generation(factual, wex.replay_executor)

        replay = wex.replay_executor
        noop = replay.fork_noop(
            factual.run_id,
            BoundarySelector.by_alias(origin.alias),
            explained_case=explained,
            stop_when_observable=True,
        )
        self.assertIsNone(noop.error)
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
        noop_obs = adapter.observe_replay(explained_case=explained, run_result=noop)
        self.assertTrue(factual_obs.subject_present)
        self.assertTrue(
            noop_obs.subject_present,
            msg=f"NOOP missing subject: error={noop_obs.error!r} "
            f"mode={noop.replay_completion_mode} "
            f"stopped={noop.stopped_after_node} nodes={noop.executed_nodes}",
        )
        self.assertIsNone(noop_obs.error)
        self.assertEqual(
            str(noop_obs.decision or "").lower(),
            str(factual_obs.decision or explained.outcome or "").lower(),
        )
        self.assertEqual(
            getattr(noop_obs.subject_payload, "blueprint_id", None),
            explained.correspondence_key,
        )
        self.assertEqual(
            getattr(noop_obs.subject_payload, "stem", None),
            getattr(factual_obs.subject_payload, "stem", None),
        )

    def test_bounded_stops_before_packaging_when_target_complete(self):
        _case, wex, factual = _build_seeded_factual_run(run_id="run_bound_stop")
        explained = _accepted_explained(factual)
        origin = _origin_before_generation(factual, wex.replay_executor)

        replay = wex.replay_executor
        bounded = replay.fork_noop(
            factual.run_id,
            BoundarySelector.by_alias(origin.alias),
            explained_case=explained,
            stop_when_observable=True,
        )
        full = replay.fork_noop(
            factual.run_id,
            BoundarySelector.by_alias(origin.alias),
            explained_case=explained,
            stop_when_observable=False,
        )
        self.assertIsNone(bounded.error)
        self.assertIsNone(full.error)

        self.assertEqual(bounded.replay_completion_mode, "target_observable")
        self.assertEqual(bounded.stopped_after_node, "apply_acceptance_gate")
        self.assertNotIn("package_final_quiz", bounded.executed_nodes)
        self.assertTrue(bounded.executed_nodes)
        self.assertEqual(full.replay_completion_mode, "graph_end")
        # Full invoke-to-END does not populate executed_nodes; assert packaging ran
        # via final cooperative state instead.
        full_cs = full.final_state.get("cooperative_state")
        self.assertTrue(getattr(full_cs, "final_quiz_packs", None))
        self.assertLessEqual(
            len(bounded.executed_nodes),
            20,
            msg="bounded early-target replay should not continue through later cycles",
        )

        adapter = MCQRunObservationAdapter()
        o_b = adapter.observe_replay(explained_case=explained, run_result=bounded)
        o_f = adapter.observe_replay(explained_case=explained, run_result=full)
        self.assertTrue(o_b.subject_present)
        self.assertTrue(o_f.subject_present)
        self.assertEqual(o_b.decision, o_f.decision)
        self.assertEqual(o_b.criterion_values, o_f.criterion_values)
        self.assertEqual(
            getattr(o_b.subject_payload, "stem", None),
            getattr(o_f.subject_payload, "stem", None),
        )


class TestCorrespondenceIgnoresPositionalQid(unittest.TestCase):
    def test_q_id_collision_uses_blueprint_key(self):
        explained = ExplainedCase(
            run_id="r",
            artifact_ref=ArtifactRef("q_00000008", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:q_00000008", 1),
            outcome="accept",
            correspondence_key="bp_eigenfaces",
        )

        class _C:
            def __init__(self, q_id, bp, rev=0):
                self.q_id = q_id
                self.blueprint_id = bp
                self.revision_count = rev

        candidates = {
            "q_00000008": _C("q_00000008", "bp_isomap"),
            "q_00000009": _C("q_00000009", "bp_novelty"),
        }
        cand, qid = resolve_corresponding_candidate(
            explained_case=explained, candidates=candidates
        )
        self.assertIsNone(cand)
        self.assertIsNone(qid)

        candidates2 = {
            "q_00000008": _C("q_00000008", "bp_isomap"),
            "q_00000007": _C("q_00000007", "bp_eigenfaces"),
        }
        cand2, qid2 = resolve_corresponding_candidate(
            explained_case=explained, candidates=candidates2
        )
        self.assertEqual(qid2, "q_00000007")
        self.assertEqual(cand2.blueprint_id, "bp_eigenfaces")


class TestEpistemicGateInvalidNoop(unittest.TestCase):
    def test_noop_error_not_supported_causal_role(self):
        finding = build_information_role_finding(
            {
                "evidence_status": "noop_error",
                "intervention_signature": {
                    "operation": "REMOVE",
                    "target_ref": {"logical_id": "chunk_1", "key": "chunk_1:v1"},
                    "artifact_type": "EvidenceChunk",
                },
                "question_content_comparison": {
                    "details": {
                        "question_content_effect_status": "question_not_reproduced",
                        "stem_effect": "unavailable",
                        "correct_answer_effect": "unavailable",
                        "distractor_effect": "unavailable",
                        "answer_explanation_effect": "unavailable",
                    }
                },
            }
        )
        self.assertIsNotNone(finding)
        self.assertEqual(finding.role_type, ROLE_INCONCLUSIVE)
        self.assertFalse(finding.diagnostics()["information_role_supported"])


class TestRouteAfterCoverageAfterDeserialize(unittest.TestCase):
    def test_rag_like_state_without_helpers_routes(self):
        from mas_langgraph.orchestrator.langgraph_workflow import route_after_coverage

        class RagLike:
            def __init__(self):
                self.mcq_blueprints = {}
                self.candidate_mcqs = {}
                self.section_evidence_status = {}
                self.doc_outline = []
                self.learning_objectives = []
                self.evidence_pool = {}
                self.knowledge_atoms = {}

        cs = RagLike()
        self.assertFalse(hasattr(cs, "planned_blueprints"))
        state = {
            "cooperative_state": cs,
            "evidence_repair_iterations": 0,
            "max_evidence_repair_iterations": 2,
            "blueprint_repair_iterations": 0,
            "max_blueprint_repair_iterations": 2,
            "coverage_gaps": [],
        }
        route = route_after_coverage(state)
        self.assertEqual(route, "package_final_quiz")
        self.assertTrue(callable(getattr(cs, "planned_blueprints", None)))


class TestDictIterationSnapshot(unittest.TestCase):
    def test_add_while_iterating_items_snapshot_safe(self):
        pool: Dict[str, Any] = {"a": object(), "b": object()}
        seen = []
        for cid, _chunk in list(pool.items()):
            seen.append(cid)
            if cid == "a":
                pool["c"] = object()
        self.assertEqual(seen, ["a", "b"])
        self.assertIn("c", pool)

        pool2: Dict[str, Any] = {"a": object(), "b": object()}
        with self.assertRaises(RuntimeError):
            for cid, _chunk in pool2.items():
                if cid == "a":
                    pool2["c"] = object()


class TestRagStep2CheckpointPersistence(unittest.TestCase):
    """c_9-class failure: monkey-patched Step-2 fields vanished on deserialize."""

    def test_rag_cooperative_state_preserves_mcq_blueprints(self):
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        from mas_langgraph.core.checkpointer import default_factual_checkpointer
        from mas_langgraph.domains.quiz.entities import MCQBlueprint
        from mas_langgraph.domains.quiz.state_integration import ensure_step2_mcq_state
        from mas_langgraph.integrations.rag_agents_bridge import attach_snapshot_counts
        from rag_agents_cooperative import (
            CooperativeState,
            DocSection,
            EvidenceChunk,
            KnowledgeAtom,
            LearningObjective,
        )

        class S(TypedDict):
            cooperative_state: Any
            n: int

        def plan(s: S):
            cs = s["cooperative_state"]
            ensure_step2_mcq_state(cs)
            attach_snapshot_counts(cs)
            cs.mcq_blueprints["bp1"] = MCQBlueprint(
                blueprint_id="bp1",
                section_id="sec",
                lo_id="lo",
                target_atom_ids=["a1"],
                evidence_chunk_ids=["c1"],
                question_type="basic_definition",
                difficulty="easy",
                bloom_level="remember",
                generation_goal="g",
                status="planned",
            )
            return {"cooperative_state": cs, "n": 1}

        def gen(s: S):
            return {"n": 2}

        g = StateGraph(S)
        g.add_node("plan", plan)
        g.add_node("gen", gen)
        g.add_edge(START, "plan")
        g.add_edge("plan", "gen")
        g.add_edge("gen", END)
        app = g.compile(checkpointer=default_factual_checkpointer())

        cs = CooperativeState()
        cs.add_section(DocSection("sec", "Title", (1, 2), 0.9))
        cs.add_learning_objective(LearningObjective("lo", "sec", "obj", "basic"))
        cs.add_evidence_chunk(
            EvidenceChunk("c1", "chunk", {"page_number": 1, "pdf_id": "141"})
        )
        cs.add_knowledge_atom(
            KnowledgeAtom(
                "a1", "atom", "fact", [{"chunk_id": "c1"}], section_id="sec", lo_id="lo"
            )
        )
        attach_snapshot_counts(cs)
        cfg = {"configurable": {"thread_id": "step2_persist"}}
        app.invoke({"cooperative_state": cs, "n": 0}, config=cfg)
        before_gen = next(
            h for h in app.get_state_history(cfg) if h.next == ("gen",)
        )
        restored = before_gen.values["cooperative_state"]
        ensure_step2_mcq_state(restored)
        attach_snapshot_counts(restored)
        self.assertIn("bp1", restored.mcq_blueprints)
        self.assertEqual(
            [bp.blueprint_id for bp in restored.planned_blueprints()], ["bp1"]
        )


class TestNoopFromBeforeGenerateLedgerHits(unittest.TestCase):
    """Full-to-END NOOP from before generate_candidate_mcqs_from_blueprints#1."""

    def test_noop_before_generate_reuses_factual_ledger(self):
        import json
        from dataclasses import dataclass
        from typing import List

        from mas_langgraph.artifacts.channel_state import (
            dump_artifact_channels,
            empty_artifact_channels,
            load_artifact_channels,
        )
        from mas_langgraph.artifacts.models import ArtifactRef
        from mas_langgraph.cases.mcq import MCQCaseDefinition
        from mas_langgraph.cases.mcq.artifact_registration import (
            register_mcq_artifacts_for_stage,
        )
        from mas_langgraph.core.checkpointer import default_factual_checkpointer
        from mas_langgraph.core.graph_builder import build_mcq_workflow_graph
        from mas_langgraph.core.runtime import WorkflowRuntime
        from mas_langgraph.core.workflow_executor import WorkflowExecutor
        from mas_langgraph.orchestrator.needs import NeedsEvaluator
        from mas_langgraph.provenance.channel_state import (
            dump_provenance_channels,
            empty_provenance_channels,
            load_provenance_recorder,
        )
        from mas_langgraph.replay.checkpoints import BoundarySelector
        from mas_langgraph.tests.test_prompt16_end_to_end import (
            _seeded_cooperative_state,
            _stub_runner,
        )

        MCQ_JSON = json.dumps(
            {
                "stem": "What does k-means do?",
                "options": [
                    {
                        "option_id": "A",
                        "text": "Assigns points to nearest centroids",
                        "is_correct": True,
                        "explanation": "Supported",
                        "misconception": "",
                    },
                    {
                        "option_id": "B",
                        "text": "Builds decision trees",
                        "is_correct": False,
                        "explanation": "Wrong",
                        "misconception": "x",
                    },
                    {
                        "option_id": "C",
                        "text": "Trains neural nets",
                        "is_correct": False,
                        "explanation": "Wrong",
                        "misconception": "x",
                    },
                    {
                        "option_id": "D",
                        "text": "Sorts arrays",
                        "is_correct": False,
                        "explanation": "Wrong",
                        "misconception": "x",
                    },
                ],
                "correct_option_id": "A",
                "answer_explanation": "Because evidence says so.",
            }
        )
        EVID_OK = json.dumps(
            {
                "supported_by_pdf": True,
                "unsupported_claims": [],
                "required_fixes": [],
                "notes": "ok",
            }
        )
        CROSS = json.dumps({"selected_option_id": "A", "reason": "ok"})

        @dataclass
        class AIMessage:
            content: str

        class ScriptedLLM:
            def __init__(self) -> None:
                self.model_name = "scripted"
                self.temperature = 0
                self.calls = 0
                self.prompts: List[str] = []

            def invoke(self, prompt, config=None, **kwargs):
                self.calls += 1
                text = prompt if isinstance(prompt, str) else str(prompt)
                self.prompts.append(text)
                if "You are generating a multiple-choice" in text:
                    return AIMessage(MCQ_JSON)
                if "Cross-CoVe" in text or "selected_option_id" in text:
                    return AIMessage(CROSS)
                return AIMessage(EVID_OK)

        class Rag:
            def __init__(self, llm):
                self.llm = llm

        run_id = "run_c9_ledger"
        case = MCQCaseDefinition()
        cs = _seeded_cooperative_state()
        cp = default_factual_checkpointer()
        graph = build_mcq_workflow_graph(checkpointer=cp)
        runner = _stub_runner()
        base = ScriptedLLM()
        rag = Rag(base)
        rt = WorkflowRuntime(
            agent_runner=runner,
            needs_evaluator=NeedsEvaluator(cs, None, agent_runner=runner),
            case_definition=case,
            rag_agents=rag,
        )
        ex = WorkflowExecutor(runtime=rt, checkpointer=cp, compiled_graph=graph)
        state: Dict[str, Any] = {
            "cooperative_state": cs,
            "run_id": run_id,
            "branch_id": "factual",
            "searched_queries": set(),
            "last_agent": None,
            "step_logs": [],
            "error": None,
            "retrieve_attempts": 0,
            "quiz_verify_loops": 0,
            "repair_iterations": 0,
            "coverage_gaps": [],
            "revision_iterations": 0,
            "max_revision_iterations": 2,
            "pending_evidence_repair_atomize": False,
            "return_to_coverage_check": False,
            "evidence_repair_iterations": 0,
            "max_evidence_repair_iterations": 2,
            "blueprint_repair_iterations": 0,
            "max_blueprint_repair_iterations": 2,
            **empty_artifact_channels(),
            **empty_provenance_channels(),
        }
        reg, resolver = load_artifact_channels(state)
        for stage in ("outline", "learning_objectives", "evidence", "atoms"):
            register_mcq_artifacts_for_stage(
                reg, resolver, cs, stage=stage, run_id=run_id, branch_id="factual"
            )
        state.update(dump_artifact_channels(reg, resolver))
        rec = load_provenance_recorder(
            state, schema=case.load_provenance_schema(), run_id=run_id
        )
        for ref, atype, node in [
            (ArtifactRef("sec_kmeans", 1), "DocSection", "build_outline"),
            (
                ArtifactRef("lo_kmeans", 1),
                "LearningObjective",
                "generate_learning_objectives",
            ),
            (
                ArtifactRef("E17", 1),
                "EvidenceChunk",
                "initial_retrieve_evidence_for_all_sections",
            ),
            (
                ArtifactRef("E18", 1),
                "EvidenceChunk",
                "initial_retrieve_evidence_for_all_sections",
            ),
            (ArtifactRef("A3", 1), "KnowledgeAtom", "atomize_evidence"),
        ]:
            occ = rec.begin_occurrence(node)
            rec.complete_occurrence(
                occ, produced_artifacts=(ref,), producer_artifact_types={ref: atype}
            )
        state.update(dump_provenance_channels(rec))
        factual = ex.run(
            state,
            runtime=rt,
            run_id=run_id,
            resume_after_node="update_section_evidence_status",
        )
        self.assertIsNone(factual.error)
        explained = _accepted_explained(factual)
        origin = _origin_before_generation(factual, ex.replay_executor)
        self.assertIn(
            "generate_candidate_mcqs_from_blueprints", origin.next_nodes or ()
        )

        # Origin checkpoint must retain planned blueprints after deserialize.
        snap = ex.replay_executor.get_checkpoint_snapshot(
            run_id, BoundarySelector.by_alias(origin.alias)
        )
        origin_cs = (getattr(snap, "values", None) or {}).get("cooperative_state")
        self.assertIsNotNone(origin_cs)
        from mas_langgraph.domains.quiz.state_integration import ensure_step2_mcq_state

        ensure_step2_mcq_state(origin_cs)
        self.assertTrue(origin_cs.planned_blueprints())

        ledger = ex.replay_executor.get_or_create_llm_ledger(run_id)
        self.assertGreater(ledger.factual_llm_request_count, 0)

        noop = ex.replay_executor.fork_noop(
            run_id,
            BoundarySelector.by_alias(origin.alias),
            explained_case=explained,
            stop_when_observable=False,
        )
        diag = ledger.last_branch_diagnostics
        self.assertIsNone(noop.error)
        self.assertGreater(diag["request_count"], 0)
        self.assertEqual(diag["factual_response_hits"], diag["request_count"])
        self.assertEqual(diag["request_misses"], 0)
        self.assertEqual(diag["live_provider_calls"], 0)

        from mas_langgraph.cases.mcq.comparison import MCQRunObservationAdapter

        obs = MCQRunObservationAdapter().observe_replay(
            explained_case=explained,
            run_result=noop,
        )
        self.assertTrue(obs.subject_present)
        self.assertIsNone(obs.error)
        self.assertEqual(obs.decision, explained.outcome)


if __name__ == "__main__":
    unittest.main()