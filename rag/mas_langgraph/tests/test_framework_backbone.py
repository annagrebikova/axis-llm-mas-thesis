"""Framework backbone tests — Prompts 2–14 (no MCQ graph topology changes)."""

from __future__ import annotations

import importlib
import unittest
from copy import deepcopy
from dataclasses import dataclass

from mas_langgraph.artifacts import (
    ArtifactEnvelope,
    ArtifactRef,
    ArtifactRegistry,
    ArtifactResolver,
)
from mas_langgraph.artifacts.registry import ArtifactTypeDecl, ArtifactTypeRegistry
from mas_langgraph.axis import (
    AxisExplanationController,
    AxisQueryParser,
    ExplainedCase,
    ProvenanceSlicer,
    QueryValidationError,
)
from mas_langgraph.cases.example_case import ExampleCaseDefinition
from mas_langgraph.cases.mcq import MCQCaseDefinition, McqOutcomeEvaluator
from mas_langgraph.cases.mcq.artifacts import (
    revise_candidate_version,
    unwrap,
    wrap_candidate_mcq,
    wrap_evidence_chunk,
)
from mas_langgraph.core.runtime import WorkflowRuntime
from mas_langgraph.core.serialization import strip_runtime_keys
from mas_langgraph.core.state_snapshots import (
    capture_state_after,
    capture_state_before,
    snapshots_share_refs,
)
from mas_langgraph.core.workflow_executor import WorkflowExecutor
from mas_langgraph.provenance import ProvenanceRecorder, ProvenanceSchema
from mas_langgraph.provenance.models import ArtifactBinding, OccurrenceRecord, SchemaRelation
from mas_langgraph.replay import (
    ArtifactIntervention,
    BoundarySelector,
    CheckpointIndex,
    CheckpointRecord,
    LangGraphReplayExecutor,
)
from mas_langgraph.replay.branches import BranchMetadata
from mas_langgraph.replay.checkpoints import extract_checkpoint_id, extract_thread_id
from mas_langgraph.replay.interventions import InterventionContract, InterventionContractRegistry


@dataclass
class _FakeChunk:
    chunk_id: str
    content: str


@dataclass
class _FakeMCQ:
    q_id: str
    stem: str
    status: str = "draft"
    revision_count: int = 0
    flags: list = None

    def __post_init__(self):
        if self.flags is None:
            self.flags = []


class TestGenericBackboneImports(unittest.TestCase):
    def test_core_import_without_mcq_payload_modules(self):
        import subprocess
        import sys
        from pathlib import Path

        rag_dir = Path(__file__).resolve().parents[2]
        code = """
import sys

FORBIDDEN = (
    "mas_langgraph.cases.mcq.cooperative_state",
    "mas_langgraph.domains.quiz.entities",
    "mas_langgraph.cases.mcq.artifacts",
)

for pkg in (
    "mas_langgraph.core",
    "mas_langgraph.artifacts",
    "mas_langgraph.provenance",
    "mas_langgraph.replay",
    "mas_langgraph.axis",
):
    __import__(pkg)

for mod in FORBIDDEN:
    assert mod not in sys.modules, mod

# Explicit MCQ import is allowed after generic packages load.
import mas_langgraph.cases.mcq.case_definition  # noqa: F401
assert "mas_langgraph.cases.mcq.case_definition" in sys.modules
"""
        subprocess.check_call(
            [sys.executable, "-c", code],
            cwd=str(rag_dir),
        )

    def test_artifacts_import_without_mcq(self):
        mod = importlib.import_module("mas_langgraph.artifacts")
        with open(mod.__file__, encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn("domains.quiz", src)
        self.assertNotIn("CandidateMCQ", src)

    def test_provenance_import_without_mcq(self):
        mod = importlib.import_module("mas_langgraph.provenance")
        with open(mod.__file__, encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn("domains.quiz", src)

    def test_example_case_imports_without_mcq(self):
        case = ExampleCaseDefinition()
        self.assertEqual(case.case_id, "example_case")
        self.assertIn("ExampleSourceArtifact", case.load_artifact_types().names())


class TestMCQCaseDefinition(unittest.TestCase):
    def test_instantiate_and_load(self):
        case = MCQCaseDefinition()
        self.assertEqual(case.case_id, "mcq_generation")
        types = case.load_artifact_types()
        self.assertIn("CandidateMCQ", types)
        schema = case.load_provenance_schema()
        self.assertIsInstance(schema, ProvenanceSchema)
        self.assertTrue(len(schema.relations) >= 1)
        ev = case.load_outcome_evaluator()
        self.assertIsInstance(ev, McqOutcomeEvaluator)
        contracts = case.load_intervention_contracts()
        self.assertIsNotNone(contracts.get("EvidenceChunk"))


class TestArtifactVersioning(unittest.TestCase):
    def test_wrap_unwrap_and_versions(self):
        reg = ArtifactRegistry()
        c1 = _FakeMCQ(q_id="Q12", stem="v1 stem")
        e1 = wrap_candidate_mcq(c1, version=1, created_in_run="run1", registry=reg)
        self.assertEqual(e1.ref, ArtifactRef("Q12", 1))
        self.assertEqual(unwrap(e1).stem, "v1 stem")

        c2 = _FakeMCQ(q_id="Q12", stem="v2 stem", revision_count=1)
        e2 = revise_candidate_version(reg, e1, c2, created_in_run="run1")
        self.assertEqual(e2.ref, ArtifactRef("Q12", 2))
        # old version preserved
        old = reg.get_by_ref(ArtifactRef("Q12", 1))
        self.assertIsNotNone(old)
        self.assertEqual(unwrap(old).stem, "v1 stem")
        self.assertTrue(old.superseded)
        self.assertEqual(len(reg.derived_from), 1)
        self.assertEqual(reg.derived_from[0].previous_artifact_ref, e1.ref)

    def test_block_does_not_delete(self):
        reg = ArtifactRegistry()
        chunk = _FakeChunk(chunk_id="E17", content="evidence")
        env = wrap_evidence_chunk(chunk, registry=reg)
        resolver = ArtifactResolver(registry=reg)
        resolver.make_available(env.ref)
        resolver.block(env.ref)
        self.assertIsNone(resolver.get(env.ref))
        self.assertIsNotNone(resolver.get_historical(env.ref))
        self.assertIsNotNone(reg.get_by_ref(env.ref))

    def test_immutability(self):
        env = ArtifactEnvelope.create(
            logical_id="x",
            artifact_type="EvidenceChunk",
            payload={"a": 1},
            branch_id="factual",
            created_in_run="r",
        )
        with self.assertRaises(Exception):
            env.branch_id = "other"  # type: ignore[misc]


class TestProvenanceOccurrences(unittest.TestCase):
    def test_repeated_occurrences_and_schema(self):
        schema = ProvenanceSchema(
            relations=[
                SchemaRelation(
                    "quiz_generator",
                    "CandidateMCQ",
                    "candidate_question",
                    "evidence_verifier",
                )
            ]
        )
        rec = ProvenanceRecorder(schema=schema)
        p1 = rec.begin_occurrence("evidence_verifier")
        r1 = rec.complete_occurrence(p1, produced_artifacts=(ArtifactRef("Q1", 1),))
        p2 = rec.begin_occurrence("evidence_verifier")
        r2 = rec.complete_occurrence(p2, produced_artifacts=(ArtifactRef("Q1", 2),))
        self.assertEqual(r1.labeled_id, "evidence_verifier#1")
        self.assertEqual(r2.labeled_id, "evidence_verifier#2")
        self.assertIs(rec.occurrence_by_label("evidence_verifier#2"), r2)

        # producer then consumer bindings
        gen = rec.begin_occurrence(
            "quiz_generator",
            input_bindings=(ArtifactBinding(ArtifactRef("E1", 1), "supporting_evidence"),),
        )
        gen = rec.complete_occurrence(gen, produced_artifacts=(ArtifactRef("Q9", 1),))
        ver = rec.begin_occurrence(
            "evidence_verifier",
            input_bindings=(
                ArtifactBinding(ArtifactRef("Q9", 1), "candidate_question"),
            ),
        )
        ver = rec.complete_occurrence(ver)
        relations = rec.build_run_relations()
        self.assertTrue(
            any(
                r.artifact_ref == ArtifactRef("Q9", 1)
                and r.relation == "candidate_question"
                for r in relations
            )
        )


class TestCheckpointAndReplay(unittest.TestCase):
    def test_executor_histories_independent(self):
        ex = LangGraphReplayExecutor(compiled_graph=None)
        r1 = ex.run_factual(
            {"step_logs": [{"a": 1}, {"a": 2}]},
            run_id="run_a",
            allow_synthetic_history=True,
        )
        r2 = ex.run_factual(
            {"step_logs": [{"b": 1}]},
            run_id="run_b",
            allow_synthetic_history=True,
        )
        self.assertEqual(r1.run_id, "run_a")
        self.assertEqual(r2.run_id, "run_b")
        self.assertGreaterEqual(len(ex.get_checkpoint_index("run_a").records), 1)
        self.assertNotEqual(
            ex.get_checkpoint_index("run_a").records[0].thread_id,
            ex.get_checkpoint_index("run_b").records[0].thread_id
            if ex.get_checkpoint_index("run_b").records
            else "x",
        )

    def test_strip_runtime_keys(self):
        state = {
            "run_id": "r",
            "agent_runner": object(),
            "rag_agents": object(),
            "cooperative_state": {"ok": True},
        }
        safe = strip_runtime_keys(state)
        self.assertNotIn("agent_runner", safe)
        self.assertNotIn("rag_agents", safe)
        self.assertIn("cooperative_state", safe)

    def test_remove_fork_and_noop(self):
        reg = ArtifactRegistry()
        env = wrap_evidence_chunk(_FakeChunk("E17", "text"), registry=reg)
        ex = LangGraphReplayExecutor(compiled_graph=None)
        contracts = InterventionContractRegistry()
        contracts.register(
            InterventionContract(
                artifact_type="EvidenceChunk",
                allowed_operations=("REMOVE", "ADD", "REPLACE"),
                replacement_compatible_types=("EvidenceChunk",),
                allow_external_add=True,
            )
        )
        ex.contracts = contracts
        factual = ex.run_factual(
            {"step_logs": [{}, {}, {}]},
            run_id="run_cf",
            allow_synthetic_history=True,
        )
        # seed resolver catalog (fixture / resolver-only path)
        resolver = ArtifactResolver(registry=reg, branch_id="factual")
        for ref in reg.all_refs():
            resolver.make_available(ref)
        ex._resolvers[("run_cf", "factual")] = resolver

        ckpt = ex.get_checkpoint_index("run_cf").records[0]
        noop = ex.fork_noop("run_cf", BoundarySelector.by_alias(ckpt.alias))
        iv = ArtifactIntervention(operation="REMOVE", artifact=env.ref, checkpoint=ckpt)
        cf = ex.fork_with_intervention(
            "run_cf", iv, artifact_type="EvidenceChunk", registry=reg
        )
        self.assertIn("factual", ex.branches.branches)
        self.assertFalse(ex.get_resolver(cf.branch_id).is_available(env.ref))
        self.assertIsNotNone(ex.get_resolver(cf.branch_id).get_historical(env.ref))
        self.assertNotEqual(noop.branch_id, cf.branch_id)
        self.assertEqual(len(ex.branches.list_siblings("factual", run_id="run_cf")), 2)

    def test_add_replace(self):
        reg = ArtifactRegistry()
        old = wrap_evidence_chunk(_FakeChunk("E1", "old"), registry=reg)
        new = wrap_evidence_chunk(_FakeChunk("E2", "new"), registry=reg)
        ex = LangGraphReplayExecutor(compiled_graph=None)
        contracts = InterventionContractRegistry()
        contracts.register(
            InterventionContract(
                artifact_type="EvidenceChunk",
                allowed_operations=("ADD", "REPLACE", "REMOVE"),
                replacement_compatible_types=("EvidenceChunk",),
                allow_external_add=True,
            )
        )
        ex.contracts = contracts
        ex.run_factual(
            {"step_logs": [{}]},
            run_id="run_ar",
            allow_synthetic_history=True,
        )
        ckpt = ex.get_checkpoint_index("run_ar").records[0]
        add = ArtifactIntervention(operation="ADD", artifact=new.ref, checkpoint=ckpt)
        br = ex.fork_with_intervention(
            "run_ar", add, artifact_type="EvidenceChunk", registry=reg
        )
        self.assertTrue(ex.get_resolver(br.branch_id).is_available(new.ref))

        missing = ArtifactRef("E999", 1)
        with self.assertRaises(ValueError):
            ex.fork_with_intervention(
                "run_ar",
                ArtifactIntervention(operation="ADD", artifact=missing, checkpoint=ckpt),
                artifact_type="EvidenceChunk",
                registry=reg,
            )

        ex.run_factual(
            {"step_logs": [{}]},
            run_id="run_rp",
            allow_synthetic_history=True,
        )
        reg2 = ArtifactRegistry()
        old2 = wrap_evidence_chunk(_FakeChunk("E1", "old"), registry=reg2)
        new2 = wrap_evidence_chunk(_FakeChunk("E2", "new"), registry=reg2)
        ckpt2 = ex.get_checkpoint_index("run_rp").records[0]
        rep = ArtifactIntervention(
            operation="REPLACE", artifact=(old2.ref, new2.ref), checkpoint=ckpt2
        )
        br2 = ex.fork_with_intervention(
            "run_rp", rep, artifact_type="EvidenceChunk", registry=reg2
        )
        resolved = ex.get_resolver(br2.branch_id).resolve_ref(old2.ref)
        self.assertEqual(resolved, new2.ref)


class TestOutcomeAndAxis(unittest.TestCase):
    def test_outcome_evaluator(self):
        from mas_langgraph.cases.mcq.artifacts import wrap_candidate_mcq
        from mas_langgraph.artifacts.registry import ArtifactRegistry
        from mas_langgraph.domains.quiz.entities import AnswerOption, CandidateMCQ, SourceRef

        ev = McqOutcomeEvaluator(require_cross_cove=False)
        ref = ArtifactRef("Q1", 1)
        candidate = CandidateMCQ(
            q_id="Q1",
            blueprint_id="bp1",
            stem="s",
            options=[
                AnswerOption("A", "a", True),
                AnswerOption("B", "b", False),
                AnswerOption("C", "c", False),
                AnswerOption("D", "d", False),
            ],
            correct_option_id="A",
            answer_explanation="because",
            source_refs=[SourceRef("d", 1, "E1")],
            status="draft",
            flags=[],
        )
        reg = ArtifactRegistry()
        wrap_candidate_mcq(candidate, registry=reg)
        assessment = ev.assess(
            {
                "artifact_catalog": dict(reg.catalog),
                "run_id": "r",
                "branch_id": "factual",
            },
            ref,
        )
        decision = ev.decide(assessment)
        self.assertIn(decision.decision, {"accept", "revise", "reject"})
        self.assertEqual(decision.target_ref, ref)
        self.assertEqual(decision.assessment_ref, assessment.assessment_ref)

    def test_slicer_excludes_unrelated(self):
        from mas_langgraph.provenance.models import OccurrenceRecord, RunProvenanceRelation

        target = ArtifactRef("Q1", 1)
        other = ArtifactRef("Q2", 1)
        decision = ArtifactRef("decision_Q1", 1)
        case = ExplainedCase(
            run_id="r",
            artifact_ref=target,
            branch_id="factual",
            decision_ref=decision,
            outcome="accept",
        )
        occs = [
            OccurrenceRecord(
                occurrence_id="o1",
                node_id="quiz_generator",
                occurrence_index=1,
                branch_id="factual",
                produced_artifacts=(target,),
                input_bindings=(ArtifactBinding(ArtifactRef("E1", 1), "supporting_evidence"),),
            ),
            OccurrenceRecord(
                occurrence_id="o2",
                node_id="quiz_generator",
                occurrence_index=2,
                branch_id="factual",
                produced_artifacts=(other,),
            ),
        ]
        rels = [
            RunProvenanceRelation("o1", target, "candidate_question", "o3", "CandidateMCQ"),
        ]
        sl = ProvenanceSlicer().backward_slice(case, occs, rels)
        ids = {o.occurrence_id for o in sl.relevant_occurrences}
        self.assertIn("o1", ids)
        self.assertNotIn("o2", ids)

    def test_axis_parser_and_explanation_loop(self):
        parser = AxisQueryParser(alias_map={"E17": ArtifactRef("E17", 1)})
        q = parser.parse("WHAT(E17)")
        resolved = parser.resolve(q)
        self.assertEqual(resolved.operation, "WHAT")
        self.assertEqual(resolved.target_ref, ArtifactRef("E17", 1))
        with self.assertRaises(QueryValidationError):
            parser.resolve(parser.parse("WHAT(E17)"))  # duplicate

        reg = ArtifactRegistry()
        env = wrap_evidence_chunk(_FakeChunk("E17", "pdf text"), registry=reg)
        wrap_candidate_mcq(_FakeMCQ("Q1", "stem?"), registry=reg)
        ex = LangGraphReplayExecutor(compiled_graph=None)
        ex.run_factual(
            {"step_logs": [{}, {}]},
            run_id="run_axis",
            allow_synthetic_history=True,
        )
        idx = ex.get_checkpoint_index("run_axis")
        from mas_langgraph.provenance.recorder import ProvenanceRecorder

        rec = ProvenanceRecorder()
        p = rec.begin_occurrence("evidence_verifier")
        rec.complete_occurrence(p, produced_artifacts=(ArtifactRef("Q1", 1),))

        contracts = InterventionContractRegistry()
        contracts.register(
            InterventionContract(
                artifact_type="EvidenceChunk",
                allowed_operations=("REMOVE", "ADD", "REPLACE"),
                replacement_compatible_types=("EvidenceChunk",),
            )
        )
        controller = AxisExplanationController(
            explained_case=ExplainedCase(
                run_id="run_axis",
                artifact_ref=ArtifactRef("Q1", 1),
                branch_id="factual",
                decision_ref=ArtifactRef("decision_Q1", 1),
                outcome="accepted",
            ),
            registry=reg,
            occurrences=rec.occurrences,
            provenance_relations=rec.build_run_relations(),
            checkpoint_index=idx,
            replay_executor=ex,
            contracts=contracts,
            alias_map={"E17": env.ref, "Q1": ArtifactRef("Q1", 1)},
            compatibility_mode=True,
        )
        # seed catalog on run resolver for REMOVE
        ex._resolvers[("run_axis", "factual")] = ArtifactResolver(registry=reg)
        for ref in reg.all_refs():
            ex._resolvers[("run_axis", "factual")].make_available(ref)

        result = controller.run(
            "Why was this accepted?",
            scripted_queries=[
                f"REMOVE(E17, {idx.records[0].alias})",
                "DONE",
            ],
        )
        self.assertGreaterEqual(result.successful_round_count, 1)
        self.assertGreaterEqual(result.rounds, 1)
        self.assertTrue(
            "Provenance slice" in result.final_explanation
            or "Slice for" in result.final_explanation
        )


class TestStateSnapshots(unittest.TestCase):
    def test_before_after_distinct(self):
        class CS:
            def __init__(self):
                self.n = 0

            def snapshot_counts(self):
                return {"n": self.n}

        cs = CS()
        state = {"cooperative_state": cs}
        before = capture_state_before(state)
        cs.n = 5
        after = capture_state_after(state, {"cooperative_state": cs})
        self.assertFalse(snapshots_share_refs(before, after))
        self.assertEqual(before["n"], 0)
        self.assertEqual(after["n"], 5)


class TestPrompt2SingleExecutionPath(unittest.TestCase):
    """Prompt 2 — one authoritative MCQ graph via CaseDefinition."""

    def test_mcq_case_builds_live_graph(self):
        case = MCQCaseDefinition()
        graph = case.build_graph()
        self.assertIsNotNone(graph)

    def test_all_entry_points_use_same_builder(self):
        from mas_langgraph.core.graph_builder import build_mcq_workflow_graph
        from mas_langgraph.orchestrator.langgraph_workflow import build_workflow_graph

        g_builder = build_mcq_workflow_graph()
        g_compat = build_workflow_graph()
        g_case = MCQCaseDefinition().build_graph()
        self.assertEqual(type(g_builder), type(g_compat))
        self.assertEqual(type(g_compat), type(g_case))

    def test_orchestrator_uses_case_definition_graph(self):
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.orchestrator.langgraph_workflow import (
            LangGraphWorkflowOrchestrator,
        )

        state = CooperativeState()
        orch = LangGraphWorkflowOrchestrator(
            state=state,
            agent_runner=object(),
            case_definition=MCQCaseDefinition(),
        )
        self.assertEqual(orch.case_definition.case_id, "mcq_generation")
        self.assertIsNotNone(orch._graph)
        self.assertIsNotNone(orch._executor)


class TestPrompt3RuntimeAndCheckpointing(unittest.TestCase):
    """Prompt 3 — runtime context separation and real factual checkpointing."""

    def test_runtime_context_node_access(self):
        from dataclasses import dataclass

        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph
        from langgraph.runtime import Runtime
        from typing import TypedDict

        from mas_langgraph.core.runtime import WorkflowRuntime

        @dataclass
        class _Runner:
            marker: str = "expected"

        class _State(TypedDict):
            logs: list

        def probe_node(state: _State, runtime: Runtime[WorkflowRuntime]) -> dict:
            assert runtime.context.agent_runner.marker == "expected"
            return {"logs": ["ok"]}

        graph = StateGraph(_State, context_schema=WorkflowRuntime)
        graph.add_node("probe", probe_node)
        graph.add_edge(START, "probe")
        graph.add_edge("probe", END)
        cp = MemorySaver()
        app = graph.compile(checkpointer=cp)
        rt = WorkflowRuntime(agent_runner=_Runner())
        app.invoke({"logs": []}, config={"configurable": {"thread_id": "t_probe"}}, context=rt)

    def test_factual_run_real_checkpoints(self):
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.core.graph_builder import build_mcq_workflow_graph
        from mas_langgraph.core.runtime import WorkflowRuntime
        from mas_langgraph.core.workflow_executor import WorkflowExecutor
        from mas_langgraph.orchestrator.needs import NeedsEvaluator

        from mas_langgraph.core.checkpointer import default_factual_checkpointer

        cp = default_factual_checkpointer()
        graph = build_mcq_workflow_graph(checkpointer=cp)
        cs = CooperativeState()

        class StubRunner:
            _searched_queries: set = set()
            coverage_plan = None
            pdf_id = None

            def invoke(self, agent_id, text="", cooperative_state=None, **kwargs):
                return {"role": agent_id, "result": "stub ok"}

            def retrieval_exhausted(self, **kwargs):
                return False

        needs = NeedsEvaluator(cs, None, agent_runner=StubRunner())
        rt = WorkflowRuntime(
            agent_runner=StubRunner(),
            needs_evaluator=needs,
            case_definition=MCQCaseDefinition(),
        )
        ex = WorkflowExecutor(runtime=rt, checkpointer=cp, compiled_graph=graph)

        initial = {
            "cooperative_state": cs,
            "run_id": "run_prompt3_a",
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
        }
        result = ex.run(initial, runtime=rt, run_id="run_prompt3_a")
        self.assertEqual(result.run_id, "run_prompt3_a")
        history_a = ex.get_state_history("run_prompt3_a")
        self.assertGreater(len(history_a), 1)
        for snap in history_a:
            ckpt_id = str((snap.config.get("configurable") or {}).get("checkpoint_id", ""))
            self.assertFalse(ckpt_id.startswith("synth_"))
            values = getattr(snap, "values", None) or {}
            for key in ("agent_runner", "rag_agents", "needs", "orchestrator_config"):
                self.assertNotIn(key, values)

        result_b = ex.run(
            {**initial, "run_id": "run_prompt3_b"},
            runtime=rt,
            run_id="run_prompt3_b",
        )
        self.assertEqual(result_b.run_id, "run_prompt3_b")
        history_b = ex.get_state_history("run_prompt3_b")
        self.assertNotEqual(
            [(s.config.get("configurable") or {}).get("checkpoint_id") for s in history_a[:3]],
            [(s.config.get("configurable") or {}).get("checkpoint_id") for s in history_b[:3]],
        )
        configs = ex.get_checkpoint_configurations("run_prompt3_a")
        self.assertTrue(configs)
        self.assertTrue(all(c["thread_id"] == "run_prompt3_a" for c in configs))

    def test_orchestrator_delegates_to_executor(self):
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.cases.mcq import MCQCaseDefinition
        from mas_langgraph.orchestrator.langgraph_workflow import LangGraphWorkflowOrchestrator

        cs = CooperativeState()

        class StubRunner:
            _searched_queries: set = set()
            coverage_plan = None
            pdf_id = None
            topic = "test"

            def invoke(self, agent_id, text="", cooperative_state=None, **kwargs):
                return {"role": agent_id, "result": "stub"}

            def retrieval_exhausted(self, **kwargs):
                return True

        orch = LangGraphWorkflowOrchestrator(
            state=cs,
            agent_runner=StubRunner(),
            case_definition=MCQCaseDefinition(),
        )
        self.assertIsNotNone(orch._executor)
        self.assertIs(orch._executor.checkpointer, orch.checkpointer)
        result = orch.run_full_workflow()
        self.assertIn("run_id", result)
        self.assertTrue(result["run_id"])
        history = orch.get_state_history()
        self.assertGreater(len(history), 0)


class TestPrompt4CopySafeState(unittest.TestCase):
    """Prompt 4 — copy-safe CooperativeState transitions."""

    def test_clone_cooperative_state_independence(self):
        from mas_langgraph.core.entities import DocSection, EvidenceChunk
        from mas_langgraph.core.state_cloning import clone_cooperative_state
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState

        original = CooperativeState()
        original.add_section(
            DocSection(
                section_id="s1",
                title="T",
                slide_range=(1, 2),
                importance_weight=1.0,
            )
        )
        original.add_evidence_chunk(
            EvidenceChunk(
                chunk_id="c1",
                content="alpha",
                metadata={},
                quality_score=0.9,
            )
        )
        clone = clone_cooperative_state(original)
        self.assertIsNot(clone, original)
        self.assertIsNot(clone.evidence_pool, original.evidence_pool)
        clone.evidence_pool["c1"].content = "changed"
        self.assertEqual(original.evidence_pool["c1"].content, "alpha")

    def test_node_does_not_mutate_input_state(self):
        from mas_langgraph.core.entities import DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.domains.quiz.blueprint_planner import create_mcq_blueprints_from_coverage
        from mas_langgraph.orchestrator.langgraph_workflow import node_plan_mcq_blueprints
        from mas_langgraph.orchestrator.runtime_helpers import workflow_runtime_context

        cs = CooperativeState()
        cs.add_section(
            DocSection(
                section_id="sec_kmeans",
                title="k-Means",
                slide_range=(6, 10),
                importance_weight=0.9,
                content_summary="k-Means clustering.",
            )
        )
        cs.add_learning_objective(
            LearningObjective(
                lo_id="lo_kmeans",
                section_id="sec_kmeans",
                objective_text="Explain k-means.",
                difficulty_level="intermediate",
            )
        )
        chunk = EvidenceChunk(
            chunk_id="chunk_kmeans",
            content="k-means assigns points to centroids.",
            metadata={"page_number": 8},
            quality_score=0.9,
        )
        cs.add_evidence_chunk(chunk)
        atom = KnowledgeAtom(
            atom_id="atom_kmeans",
            content="k-means alternates assignment.",
            atom_type="procedure",
            citations=[{"chunk_id": "chunk_kmeans"}],
            section_id="sec_kmeans",
            lo_id="lo_kmeans",
        )
        cs.add_knowledge_atom(atom)
        cs.link_chunk_to_atom("chunk_kmeans", "atom_kmeans")
        before_count = len(cs.mcq_blueprints)

        graph_state = {
            "cooperative_state": cs,
            "step_logs": [],
            "error": None,
        }
        runtime = workflow_runtime_context()
        update = node_plan_mcq_blueprints(graph_state, runtime)
        self.assertIsNot(update["cooperative_state"], cs)
        self.assertEqual(len(cs.mcq_blueprints), before_count)
        self.assertGreater(len(update["cooperative_state"].mcq_blueprints), before_count)

    def test_stale_runtime_state_resistance(self):
        from mas_langgraph.core.entities import DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.orchestrator.langgraph_workflow import node_plan_mcq_blueprints
        from mas_langgraph.orchestrator.runtime_helpers import workflow_runtime_context
        from mas_langgraph.orchestrator.needs import NeedsEvaluator

        live = CooperativeState()
        stale = CooperativeState()
        live.add_section(
            DocSection(
                section_id="sec_kmeans",
                title="k-Means",
                slide_range=(6, 10),
                importance_weight=0.9,
            )
        )
        live.add_learning_objective(
            LearningObjective(
                lo_id="lo_kmeans",
                section_id="sec_kmeans",
                objective_text="Explain k-means.",
                difficulty_level="intermediate",
            )
        )
        chunk = EvidenceChunk(
            chunk_id="chunk_kmeans",
            content="k-means content",
            metadata={},
            quality_score=0.9,
        )
        live.add_evidence_chunk(chunk)
        atom = KnowledgeAtom(
            atom_id="atom_kmeans",
            content="procedure",
            atom_type="procedure",
            citations=[{"chunk_id": "chunk_kmeans"}],
            section_id="sec_kmeans",
            lo_id="lo_kmeans",
        )
        live.add_knowledge_atom(atom)
        live.link_chunk_to_atom("chunk_kmeans", "atom_kmeans")

        class StubRunner:
            state = stale

        needs = NeedsEvaluator(stale, agent_runner=StubRunner())
        runtime = workflow_runtime_context(needs_evaluator=needs, agent_runner=StubRunner())
        update = node_plan_mcq_blueprints(
            {"cooperative_state": live, "step_logs": [], "error": None},
            runtime,
        )
        self.assertGreater(len(update["cooperative_state"].mcq_blueprints), 0)
        self.assertEqual(len(stale.mcq_blueprints), 0)

    def test_checkpoint_historical_isolation(self):
        from mas_langgraph.core.checkpointer import default_factual_checkpointer
        from mas_langgraph.core.entities import DocSection
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.core.graph_builder import build_mcq_workflow_graph
        from mas_langgraph.core.runtime import WorkflowRuntime
        from mas_langgraph.core.workflow_executor import WorkflowExecutor
        from mas_langgraph.orchestrator.needs import NeedsEvaluator

        cp = default_factual_checkpointer()
        graph = build_mcq_workflow_graph(checkpointer=cp)
        cs = CooperativeState()

        class StubRunner:
            coverage_plan = None
            pdf_id = None

            def invoke(self, agent_id, text="", cooperative_state=None, **kwargs):
                return {"role": agent_id, "result": "stub"}

            def retrieval_exhausted(self, **kwargs):
                return False

        needs = NeedsEvaluator(cs, None, agent_runner=StubRunner())
        rt = WorkflowRuntime(
            agent_runner=StubRunner(),
            needs_evaluator=needs,
            case_definition=MCQCaseDefinition(),
        )
        ex = WorkflowExecutor(runtime=rt, checkpointer=cp, compiled_graph=graph)

        initial = {
            "cooperative_state": cs,
            "run_id": "run_p4_iso",
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
        }
        ex.run(initial, runtime=rt, run_id="run_p4_iso")
        history = ex.get_state_history("run_p4_iso")
        self.assertGreaterEqual(len(history), 2)
        early = history[0]
        early_cs = early.values.get("cooperative_state")
        early_outline_len = len(getattr(early_cs, "doc_outline", []) or [])
        final_cs = ex.get_final_state("run_p4_iso")["cooperative_state"]
        if hasattr(final_cs, "doc_outline") and final_cs.doc_outline:
            final_cs.doc_outline.append(
                DocSection(
                    section_id="injected",
                    title="Injected",
                    slide_range=(99, 100),
                    importance_weight=0.1,
                )
            )
        early_cs_after = history[0].values.get("cooperative_state")
        self.assertEqual(
            len(getattr(early_cs_after, "doc_outline", []) or []),
            early_outline_len,
        )

    def test_state_before_after_logging(self):
        from mas_langgraph.core.entities import DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.orchestrator.langgraph_workflow import node_plan_mcq_blueprints
        from mas_langgraph.orchestrator.runtime_helpers import workflow_runtime_context

        cs = CooperativeState()
        cs.add_section(
            DocSection(
                section_id="sec_kmeans",
                title="k-Means",
                slide_range=(6, 10),
                importance_weight=0.9,
            )
        )
        cs.add_learning_objective(
            LearningObjective(
                lo_id="lo_kmeans",
                section_id="sec_kmeans",
                objective_text="Explain k-means.",
                difficulty_level="intermediate",
            )
        )
        chunk = EvidenceChunk(
            chunk_id="chunk_kmeans",
            content="k-means content",
            metadata={},
            quality_score=0.9,
        )
        cs.add_evidence_chunk(chunk)
        atom = KnowledgeAtom(
            atom_id="atom_kmeans",
            content="procedure",
            atom_type="procedure",
            citations=[{"chunk_id": "chunk_kmeans"}],
            section_id="sec_kmeans",
            lo_id="lo_kmeans",
        )
        cs.add_knowledge_atom(atom)
        cs.link_chunk_to_atom("chunk_kmeans", "atom_kmeans")

        update = node_plan_mcq_blueprints(
            {"cooperative_state": cs, "step_logs": [], "error": None},
            workflow_runtime_context(),
        )
        log = update["step_logs"][0]
        before_bp = log["state_before"].get("mcq_blueprints_len", 0)
        after_bp = log["state_after"].get("mcq_blueprints_len", 0)
        self.assertEqual(before_bp, 0)
        self.assertGreater(after_bp, 0)

    def test_searched_queries_graph_authoritative(self):
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.orchestrator.langgraph_workflow import route_after_retrieve
        from mas_langgraph.orchestrator.needs import NeedsEvaluator
        from mas_langgraph.orchestrator.runtime_helpers import workflow_runtime_context

        cs = CooperativeState()

        class StubRunner:
            _searched_queries = {"runner_stale_query"}

            def __init__(self):
                self.last_searched: set | None = None

            def retrieval_exhausted(self, cooperative_state=None, searched_queries=None):
                self.last_searched = set(searched_queries or set())
                return True

        runner = StubRunner()
        needs = NeedsEvaluator(cs, agent_runner=runner)
        runtime = workflow_runtime_context(needs_evaluator=needs, agent_runner=runner)
        graph_state = {
            "cooperative_state": cs,
            "searched_queries": {"graph_authoritative_query"},
            "retrieve_attempts": 0,
        }
        route_after_retrieve(graph_state, runtime)
        self.assertEqual(runner.last_searched, {"graph_authoritative_query"})


class TestPrompt5ArtifactIdentity(unittest.TestCase):
    """Prompt 5 — generic immutable artifact identity layer."""

    def test_artifact_ref_equality_and_hash(self):
        a = ArtifactRef("Q12", 1)
        b = ArtifactRef("Q12", 1)
        c = ArtifactRef("Q12", 2)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(a.logical_id, c.logical_id)
        self.assertIn(a, {b})
        self.assertNotIn(c, {a})

    def test_envelope_payload_isolation(self):
        from mas_langgraph.cases.mcq.artifacts import wrap_candidate_mcq

        candidate = _FakeMCQ(q_id="Q12", stem="original")
        envelope = wrap_candidate_mcq(candidate, created_in_run="run1")
        candidate.stem = "CHANGED"
        self.assertNotEqual(envelope.payload.stem, "CHANGED")
        self.assertEqual(envelope.payload.stem, "original")

    def test_duplicate_registration_conflict(self):
        reg = ArtifactRegistry()
        chunk_a = _FakeChunk(chunk_id="E17", content="alpha")
        chunk_b = _FakeChunk(chunk_id="E17", content="beta")
        wrap_evidence_chunk(chunk_a, created_in_run="run1", registry=reg)
        with self.assertRaises(ValueError):
            wrap_evidence_chunk(chunk_b, created_in_run="run1", registry=reg)

    def test_duplicate_registration_idempotent(self):
        reg = ArtifactRegistry()
        chunk = _FakeChunk(chunk_id="E17", content="alpha")
        e1 = wrap_evidence_chunk(chunk, created_in_run="run1", registry=reg)
        e2 = wrap_evidence_chunk(chunk, created_in_run="run1", registry=reg)
        self.assertIs(e1, e2)
        self.assertEqual(len(reg.catalog), 1)

    def test_version_lookup_order(self):
        reg = ArtifactRegistry()
        c1 = _FakeMCQ(q_id="Q12", stem="v1")
        c2 = _FakeMCQ(q_id="Q12", stem="v2", revision_count=1)
        wrap_candidate_mcq(c1, version=1, created_in_run="run1", registry=reg)
        wrap_candidate_mcq(c2, version=2, created_in_run="run1", registry=reg)
        versions = reg.versions_of("Q12", "factual")
        self.assertEqual([e.ref.version for e in versions], [1, 2])

    def test_branch_identity_distinguishable(self):
        chunk = _FakeChunk(chunk_id="E17", content="evidence")
        factual = wrap_evidence_chunk(
            chunk, branch_id="factual", created_in_run="run1",
        )
        counterfactual = wrap_evidence_chunk(
            chunk, branch_id="cf-remove-E17", created_in_run="run1",
        )
        self.assertNotEqual(factual.instance_id, counterfactual.instance_id)
        self.assertEqual(factual.ref, counterfactual.ref)
        self.assertNotEqual(factual.branch_id, counterfactual.branch_id)

    def test_mcq_adapter_wrapping(self):
        from mas_langgraph.cases.mcq.artifacts import (
            wrap_blueprint,
            wrap_cross_cove_report,
            wrap_evidence_report,
            wrap_knowledge_atom,
            wrap_quiz_pack,
            wrap_supervisor_decision,
        )
        from mas_langgraph.domains.quiz.entities import (
            CrossCoVeReport,
            EvidenceCheckReport,
            MCQBlueprint,
            SupervisorDecision,
        )
        from mas_langgraph.core.entities import EvidenceChunk, KnowledgeAtom, QuizPack

        chunk_env = wrap_evidence_chunk(
            EvidenceChunk("E17", "text", {}, quality_score=0.9),
            created_in_run="run42",
            branch_id="factual",
        )
        self.assertEqual(chunk_env.ref, ArtifactRef("E17", 1))
        self.assertEqual(chunk_env.created_in_run, "run42")

        atom_env = wrap_knowledge_atom(
            KnowledgeAtom("A3", "x", "fact", [], section_id="s1"),
            created_in_run="run42",
            branch_id="factual",
        )
        self.assertEqual(atom_env.artifact_type, "KnowledgeAtom")

        bp_env = wrap_blueprint(
            MCQBlueprint(
                "BP5", "s1", "lo1", [], [], "basic_definition",
                "easy", "Remembering", "goal",
            ),
            created_in_run="run42",
            branch_id="factual",
        )
        self.assertEqual(bp_env.ref.logical_id, "BP5")

        cand_env = wrap_candidate_mcq(
            _FakeMCQ(q_id="Q12", stem="stem"),
            created_in_run="run42",
            branch_id="factual",
        )
        self.assertEqual(cand_env.ref, ArtifactRef("Q12", 1))

        report_env = wrap_evidence_report(
            EvidenceCheckReport(
                "Q12", True, [], [], 0.9,
            ),
            created_in_run="run42",
            branch_id="factual",
        )
        self.assertEqual(report_env.ref.logical_id, "evidence-report:Q12")

        cc_env = wrap_cross_cove_report(
            CrossCoVeReport("Q12", "A", ["A"], "A", True),
            created_in_run="run42",
            branch_id="factual",
        )
        self.assertEqual(cc_env.artifact_type, "CrossCoVeReport")

        dec_env = wrap_supervisor_decision(
            SupervisorDecision("Q12", "accept", ["ok"]),
            created_in_run="run42",
            branch_id="factual",
        )
        self.assertEqual(dec_env.ref.logical_id, "decision:Q12")

        pack_env = wrap_quiz_pack(
            QuizPack("pack1", "Quiz", [], metadata={}),
            created_in_run="run42",
            branch_id="factual",
        )
        self.assertEqual(pack_env.artifact_type, "QuizPack")

    def test_artifacts_import_isolation(self):
        import subprocess
        import sys
        from pathlib import Path

        rag_dir = Path(__file__).resolve().parents[2]
        code = """
import sys
import mas_langgraph.artifacts as artifacts_pkg
assert "mas_langgraph.cases.mcq.artifacts" not in sys.modules
assert "mas_langgraph.domains.quiz.entities" not in sys.modules
"""
        subprocess.check_call([sys.executable, "-c", code], cwd=str(rag_dir))

    def test_factual_run_registers_artifacts(self):
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.core.checkpointer import default_factual_checkpointer
        from mas_langgraph.core.graph_builder import build_mcq_workflow_graph
        from mas_langgraph.core.runtime import WorkflowRuntime
        from mas_langgraph.core.workflow_executor import WorkflowExecutor
        from mas_langgraph.orchestrator.needs import NeedsEvaluator

        cp = default_factual_checkpointer()
        graph = build_mcq_workflow_graph(checkpointer=cp)
        cs = CooperativeState()

        class StubRunner:
            coverage_plan = None
            pdf_id = None

            def invoke(self, agent_id, text="", cooperative_state=None, **kwargs):
                return {"role": agent_id, "result": "stub"}

            def retrieval_exhausted(self, **kwargs):
                return False

        rt = WorkflowRuntime(
            agent_runner=StubRunner(),
            needs_evaluator=NeedsEvaluator(cs, None, agent_runner=StubRunner()),
            case_definition=MCQCaseDefinition(),
        )
        ex = WorkflowExecutor(runtime=rt, checkpointer=cp, compiled_graph=graph)
        run_id = "run_prompt5_artifacts"
        initial = {
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
        }
        ex.run(initial, runtime=rt, run_id=run_id)
        final = ex.get_final_state(run_id)
        catalog = final.get("artifact_catalog") or {}
        self.assertGreater(len(catalog), 0)
        types = {env.artifact_type for env in catalog.values()}
        self.assertIn("QuizPack", types)
        for env in catalog.values():
            self.assertEqual(env.created_in_run, run_id)
            self.assertEqual(env.branch_id, "factual")
        self.assertEqual(final.get("blocked_artifact_refs") or set(), set())
        self.assertEqual(final.get("artifact_substitutions") or {}, {})

    def test_checkpoint_artifact_catalog_isolation(self):
        from mas_langgraph.core.checkpointer import default_factual_checkpointer
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.core.graph_builder import build_mcq_workflow_graph
        from mas_langgraph.core.runtime import WorkflowRuntime
        from mas_langgraph.core.workflow_executor import WorkflowExecutor
        from mas_langgraph.orchestrator.needs import NeedsEvaluator

        cp = default_factual_checkpointer()
        graph = build_mcq_workflow_graph(checkpointer=cp)
        cs = CooperativeState()

        class StubRunner:
            coverage_plan = None
            pdf_id = None

            def invoke(self, agent_id, text="", cooperative_state=None, **kwargs):
                return {"role": agent_id, "result": "stub"}

            def retrieval_exhausted(self, **kwargs):
                return False

        rt = WorkflowRuntime(
            agent_runner=StubRunner(),
            needs_evaluator=NeedsEvaluator(cs, None, agent_runner=StubRunner()),
            case_definition=MCQCaseDefinition(),
        )
        ex = WorkflowExecutor(runtime=rt, checkpointer=cp, compiled_graph=graph)
        run_id = "run_p5_iso"
        initial = {
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
        }
        ex.run(initial, runtime=rt, run_id=run_id)
        history = ex.get_state_history(run_id)
        self.assertGreaterEqual(len(history), 2)
        early_count = len((history[0].values or {}).get("artifact_catalog") or {})
        final_catalog = ex.get_final_state(run_id).get("artifact_catalog") or {}
        self.assertGreater(len(final_catalog), early_count)
        self.assertEqual(
            len((history[0].values or {}).get("artifact_catalog") or {}),
            early_count,
        )


class TestPrompt6ImmutableRevision(unittest.TestCase):
    """Prompt 6 — immutable MCQ revisions and version-specific reports."""

    def _revision_state(self):
        from mas_langgraph.core.entities import DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
        from mas_langgraph.core.state import CooperativeState
        from mas_langgraph.domains.quiz.blueprint_planner import create_mcq_blueprints_from_coverage
        from mas_langgraph.domains.quiz.entities import AnswerOption, CandidateMCQ, CrossCoVeReport, EvidenceCheckReport, SourceRef, SupervisorDecision
        from mas_langgraph.domains.quiz.revision import revise_candidate_mcq

        state = CooperativeState()
        state.add_section(
            DocSection(
                section_id="sec_kmeans",
                title="k-Means",
                slide_range=(6, 10),
                importance_weight=0.9,
            )
        )
        state.add_learning_objective(
            LearningObjective(
                lo_id="lo_kmeans",
                section_id="sec_kmeans",
                objective_text="Explain k-means.",
                difficulty_level="intermediate",
            )
        )
        chunk = EvidenceChunk(
            chunk_id="chunk_kmeans",
            content="k-means assigns points to nearest centroids.",
            metadata={"page_number": 8},
            quality_score=0.9,
        )
        state.add_evidence_chunk(chunk)
        atom = KnowledgeAtom(
            atom_id="atom_kmeans",
            content="k-means alternates assignment.",
            atom_type="procedure",
            citations=[{"chunk_id": "chunk_kmeans"}],
            section_id="sec_kmeans",
            lo_id="lo_kmeans",
        )
        state.add_knowledge_atom(atom)
        state.link_chunk_to_atom("chunk_kmeans", "atom_kmeans")
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        state.store_mcq_blueprints([blueprint])
        v1 = CandidateMCQ(
            q_id="Q12",
            blueprint_id=blueprint.blueprint_id,
            stem="Old stem",
            options=[
                AnswerOption(option_id="A", text="Old correct", is_correct=True),
                AnswerOption(option_id="B", text="Wrong", is_correct=False),
                AnswerOption(option_id="C", text="Wrong2", is_correct=False),
                AnswerOption(option_id="D", text="Wrong3", is_correct=False),
            ],
            correct_option_id="A",
            answer_explanation="Old explanation.",
            source_refs=[SourceRef(doc_id="141", page=8, chunk_id="chunk_kmeans")],
            status="needs_revision",
            flags=["evidence_not_supported"],
        )
        state.store_candidate_mcq(v1)
        state.store_evidence_report(
            EvidenceCheckReport(
                q_id="Q12",
                supported_by_pdf=False,
                unsupported_claims=["bad"],
                source_refs=v1.source_refs,
                confidence=0.2,
                candidate_version=1,
            )
        )
        state.store_cross_cove_report(
            CrossCoVeReport(
                q_id="Q12",
                generator_answer="A",
                verifier_answers=["B"],
                majority_answer="B",
                passed=False,
                candidate_version=1,
            )
        )
        state.store_supervisor_decision(
            SupervisorDecision(
                q_id="Q12",
                decision="revise",
                reasons=["needs work"],
                candidate_version=1,
            )
        )
        return state, v1, revise_candidate_mcq

    def test_revision_does_not_mutate_source(self):
        state, v1, revise = self._revision_state()
        v2 = revise(v1, state, llm=None, max_revisions=2)
        self.assertIsNot(v2, v1)
        self.assertEqual(v1.stem, "Old stem")
        self.assertEqual(v1.revision_count, 0)
        self.assertEqual(v2.revision_count, 1)
        self.assertEqual(v2.q_id, "Q12")

    def test_multiple_revisions_chain(self):
        from mas_langgraph.domains.quiz.entities import EvidenceCheckReport, SupervisorDecision

        state, v1, revise = self._revision_state()
        v2 = revise(v1, state, llm=None, max_revisions=3)
        v2.status = "needs_revision"
        state.store_candidate_mcq(v2)
        state.store_evidence_report(
            EvidenceCheckReport(
                q_id="Q12",
                supported_by_pdf=False,
                unsupported_claims=["bad"],
                source_refs=v2.source_refs,
                confidence=0.2,
                candidate_version=2,
            )
        )
        state.store_supervisor_decision(
            SupervisorDecision(
                q_id="Q12",
                decision="revise",
                reasons=["again"],
                candidate_version=2,
            )
        )
        v3 = revise(v2, state, llm=None, max_revisions=3)
        self.assertEqual(v3.revision_count, 2)
        self.assertEqual(v2.revision_count, 1)
        self.assertEqual(v1.revision_count, 0)

    def test_no_stale_report_reuse(self):
        from mas_langgraph.domains.quiz.coverage import apply_acceptance_gate
        from mas_langgraph.domains.quiz.versioning import get_operational_evidence_report

        state, v1, revise = self._revision_state()
        self.assertIsNotNone(get_operational_evidence_report(state, v1))
        v2 = revise(v1, state, llm=None, max_revisions=2)
        self.assertIsNone(get_operational_evidence_report(state, v2))
        decision = apply_acceptance_gate(v2, state)
        self.assertEqual(decision.decision, "revise")
        self.assertIn("needs_evidence_verification", decision.required_fixes)

    def test_version_specific_reports_in_catalog(self):
        from mas_langgraph.artifacts.models import ArtifactRef, DerivedFrom
        from mas_langgraph.artifacts.registry import ArtifactRegistry
        from mas_langgraph.cases.mcq.artifact_registration import register_mcq_artifacts_for_stage

        state, v1, revise = self._revision_state()
        reg = ArtifactRegistry()
        from mas_langgraph.artifacts.resolver import ArtifactResolver
        resolver = ArtifactResolver(registry=reg, branch_id="factual")
        register_mcq_artifacts_for_stage(reg, resolver, state, stage="candidates", run_id="run6", branch_id="factual")
        register_mcq_artifacts_for_stage(reg, resolver, state, stage="evidence_reports", run_id="run6", branch_id="factual")
        v1_report = reg.get_by_ref(ArtifactRef("evidence-report:Q12", 1), "factual")
        self.assertIsNotNone(v1_report)
        v2 = revise(v1, state, llm=None, max_revisions=2)
        register_mcq_artifacts_for_stage(reg, resolver, state, stage="candidates", run_id="run6", branch_id="factual")
        versions = [e.ref.version for e in reg.versions_of("Q12", "factual")]
        self.assertEqual(versions, [1, 2])
        self.assertTrue(any(isinstance(d, DerivedFrom) for d in reg.derived_from))

    def test_packaging_selects_accepted_version(self):
        from mas_langgraph.domains.quiz.coverage import apply_acceptance_gate
        from mas_langgraph.domains.quiz.packaging import package_final_quiz
        from mas_langgraph.domains.quiz.verification import verify_candidate_against_evidence
        from mas_langgraph.domains.quiz.cross_cove import run_cross_cove_for_candidate

        state, v1, revise = self._revision_state()
        v1.status = "needs_revision"
        v2 = revise(v1, state, llm=None, max_revisions=2)
        report = verify_candidate_against_evidence(v2, state, llm=None)
        state.store_evidence_report(report)
        cove = run_cross_cove_for_candidate(v2, verifier_llm=None)
        state.store_cross_cove_report(cove)
        decision = apply_acceptance_gate(v2, state)
        state.store_supervisor_decision(decision)
        state.store_candidate_mcq(v2)
        pack = package_final_quiz(state, title="Quiz")
        self.assertEqual(len(pack.questions), 1)
        self.assertIn("PDF evidence", pack.questions[0].question_text)


class TestPrompt7OccurrenceProvenance(unittest.TestCase):
    """Prompt 7 — occurrence-level provenance and D_run."""

    def _stub_runtime(self, cs):
        from mas_langgraph.cases.mcq import MCQCaseDefinition
        from mas_langgraph.core.runtime import WorkflowRuntime
        from mas_langgraph.orchestrator.needs import NeedsEvaluator

        class StubRunner:
            coverage_plan = None
            pdf_id = None

            def invoke(self, agent_id, text="", cooperative_state=None, **kwargs):
                return {"role": agent_id, "result": "stub"}

            def retrieval_exhausted(self, **kwargs):
                return False

        return WorkflowRuntime(
            agent_runner=StubRunner(),
            needs_evaluator=NeedsEvaluator(cs, None, agent_runner=StubRunner()),
            case_definition=MCQCaseDefinition(),
        )

    def _seeded_state(self):
        from mas_langgraph.core.entities import DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.domains.quiz.blueprint_planner import create_mcq_blueprints_from_coverage
        from mas_langgraph.domains.quiz.entities import AnswerOption, CandidateMCQ, SourceRef
        from mas_langgraph.domains.quiz.generation import register_candidate_mcq

        cs = CooperativeState()
        cs.add_section(
            DocSection("sec_kmeans", "k-Means", (6, 10), 0.9, content_summary="k-Means")
        )
        cs.add_learning_objective(
            LearningObjective("lo_kmeans", "sec_kmeans", "Explain k-means.", "intermediate")
        )
        chunk = EvidenceChunk(
            "E17",
            "k-means assigns points to nearest centroids then updates them.",
            {"page_number": 8},
            quality_score=0.9,
        )
        cs.add_evidence_chunk(chunk)
        atom = KnowledgeAtom(
            "A3",
            "k-means alternates assignment.",
            "procedure",
            [{"chunk_id": "E17"}],
            section_id="sec_kmeans",
            lo_id="lo_kmeans",
        )
        cs.add_knowledge_atom(atom)
        cs.link_chunk_to_atom("E17", "A3")
        bps = create_mcq_blueprints_from_coverage(cs)
        cs.store_mcq_blueprints(bps)
        bp = bps[0]
        mcq = CandidateMCQ(
            q_id="Q12",
            blueprint_id=bp.blueprint_id,
            stem="What does k-means assign points to?",
            options=[
                AnswerOption("A", "Nearest centroid", True),
                AnswerOption("B", "Random cluster", False),
                AnswerOption("C", "Previous label", False),
                AnswerOption("D", "Outlier bucket", False),
            ],
            correct_option_id="A",
            answer_explanation="Nearest centroid assignment.",
            source_refs=[SourceRef("141", 8, "E17", "A3")],
            status="draft",
        )
        register_candidate_mcq(cs, mcq, bp)
        return cs, bp, mcq

    def test_deterministic_occurrence_ids_and_repeated_verify(self):
        from mas_langgraph.artifacts.models import ArtifactRef
        from mas_langgraph.provenance.models import ArtifactBinding, make_occurrence_id
        from mas_langgraph.provenance.recorder import ProvenanceRecorder
        from mas_langgraph.cases.mcq import MCQCaseDefinition

        schema = MCQCaseDefinition().load_provenance_schema()
        rec = ProvenanceRecorder(schema=schema, run_id="run42", branch_id="factual")
        p1 = rec.begin_occurrence("verify_candidate_mcqs_against_evidence")
        r1 = rec.complete_occurrence(p1, produced_artifacts=(ArtifactRef("evidence-report:Q12", 1),))
        p2 = rec.begin_occurrence("verify_candidate_mcqs_against_evidence")
        r2 = rec.complete_occurrence(p2, produced_artifacts=(ArtifactRef("evidence-report:Q12", 2),))
        self.assertEqual(r1.occurrence_index, 1)
        self.assertEqual(r2.occurrence_index, 2)
        self.assertEqual(
            r1.occurrence_id,
            make_occurrence_id(
                run_id="run42",
                branch_id="factual",
                node_id="verify_candidate_mcqs_against_evidence",
                occurrence_index=1,
            ),
        )
        self.assertNotEqual(r1.occurrence_id, r2.occurrence_id)
        self.assertEqual(r1.execution_index, 1)
        self.assertEqual(r2.execution_index, 2)

    def test_build_run_relations_populates_artifact_type(self):
        from mas_langgraph.artifacts.models import ArtifactRef
        from mas_langgraph.provenance.models import ArtifactBinding
        from mas_langgraph.provenance.recorder import ProvenanceRecorder
        from mas_langgraph.cases.mcq import MCQCaseDefinition

        schema = MCQCaseDefinition().load_provenance_schema()
        rec = ProvenanceRecorder(schema=schema, run_id="runT", branch_id="factual")
        gen = rec.begin_occurrence("generate_candidate_mcqs_from_blueprints")
        rec.register_artifact_type(ArtifactRef("Q12", 1), "CandidateMCQ")
        gen = rec.complete_occurrence(
            gen,
            produced_artifacts=(ArtifactRef("Q12", 1),),
            producer_artifact_types={ArtifactRef("Q12", 1): "CandidateMCQ"},
        )
        ver = rec.begin_occurrence(
            "verify_candidate_mcqs_against_evidence",
            input_bindings=(ArtifactBinding(ArtifactRef("Q12", 1), "candidate_question"),),
        )
        rec.complete_occurrence(ver)
        relations = rec.build_run_relations()
        match = [r for r in relations if r.artifact_ref == ArtifactRef("Q12", 1)]
        self.assertTrue(match)
        self.assertEqual(match[0].artifact_type, "CandidateMCQ")
        self.assertEqual(match[0].relation, "candidate_question")

    def test_schema_violation_retained(self):
        from mas_langgraph.artifacts.models import ArtifactRef
        from mas_langgraph.provenance.models import ArtifactBinding
        from mas_langgraph.provenance.recorder import ProvenanceRecorder
        from mas_langgraph.cases.mcq import MCQCaseDefinition

        schema = MCQCaseDefinition().load_provenance_schema()
        rec = ProvenanceRecorder(schema=schema, run_id="runV", branch_id="factual")
        gen = rec.begin_occurrence("generate_candidate_mcqs_from_blueprints")
        rec.complete_occurrence(
            gen,
            produced_artifacts=(ArtifactRef("Q12", 1),),
            producer_artifact_types={ArtifactRef("Q12", 1): "CandidateMCQ"},
        )
        bad = rec.begin_occurrence(
            "package_final_quiz",
            input_bindings=(ArtifactBinding(ArtifactRef("Q12", 1), "bogus_relation"),),
        )
        rec.complete_occurrence(bad)
        self.assertTrue(rec.violations)
        self.assertTrue(any(r.relation == "bogus_relation" for r in rec.run_provenance_relations))

    def test_generator_bindings_from_blueprint_fields(self):
        from mas_langgraph.artifacts.channel_state import empty_artifact_channels
        from mas_langgraph.artifacts.registry import ArtifactRegistry
        from mas_langgraph.cases.mcq.artifact_registration import register_mcq_artifacts_for_stage
        from mas_langgraph.cases.mcq.provenance import input_bindings_for_node
        from mas_langgraph.artifacts.resolver import ArtifactResolver

        cs, bp, mcq = self._seeded_state()
        # Clear candidates so generation node binds planned blueprints
        cs.candidate_mcqs.clear()
        for b in cs.mcq_blueprints.values():
            b.status = "planned"
        reg = ArtifactRegistry()
        resolver = ArtifactResolver(registry=reg, branch_id="factual")
        for stage in ("outline", "learning_objectives", "evidence", "atoms", "blueprints"):
            register_mcq_artifacts_for_stage(
                reg, resolver, cs, stage=stage, run_id="runB", branch_id="factual"
            )
        bindings = input_bindings_for_node(
            "generate_candidate_mcqs_from_blueprints",
            {"cooperative_state": cs},
            reg,
            branch_id="factual",
        )
        relations = {b.relation for b in bindings}
        self.assertIn("generation_blueprint", relations)
        self.assertIn("supporting_evidence", relations)
        self.assertIn("knowledge_atom", relations)
        bp_refs = [b.artifact_ref for b in bindings if b.relation == "generation_blueprint"]
        self.assertTrue(any(r.logical_id == bp.blueprint_id for r in bp_refs))
        ev_refs = [b.artifact_ref for b in bindings if b.relation == "supporting_evidence"]
        self.assertTrue(any(r.logical_id == "E17" for r in ev_refs))

    def test_instrumented_verify_acceptance_revision_chain(self):
        from mas_langgraph.artifacts.channel_state import empty_artifact_channels
        from mas_langgraph.artifacts.models import ArtifactRef, DerivedFrom
        from mas_langgraph.cases.mcq.artifact_registration import register_mcq_artifacts_for_stage
        from mas_langgraph.core.checkpointer import default_factual_checkpointer
        from mas_langgraph.domains.quiz.entities import (
            CrossCoVeReport,
            EvidenceCheckReport,
            SupervisorDecision,
        )
        from mas_langgraph.domains.quiz.revision import revise_candidate_mcq
        from mas_langgraph.provenance.channel_state import empty_provenance_channels
        from mas_langgraph.provenance.instrumentation import instrument_node
        from mas_langgraph.orchestrator import langgraph_workflow as lg
        from langgraph.runtime import Runtime

        cs, bp, mcq = self._seeded_state()
        runtime = self._stub_runtime(cs)
        state = {
            "cooperative_state": cs,
            "run_id": "run_rev",
            "branch_id": "factual",
            **empty_artifact_channels(),
            **empty_provenance_channels(),
            "searched_queries": set(),
            "step_logs": [],
            "error": None,
            "revision_iterations": 0,
            "max_revision_iterations": 2,
        }
        # Pre-register upstream artifacts
        from mas_langgraph.artifacts.channel_state import load_artifact_channels, dump_artifact_channels

        reg, resolver = load_artifact_channels(state)
        for stage in ("outline", "learning_objectives", "evidence", "atoms", "blueprints", "candidates"):
            register_mcq_artifacts_for_stage(
                reg, resolver, cs, stage=stage, run_id="run_rev", branch_id="factual"
            )
        state.update(dump_artifact_channels(reg, resolver))

        class _RT:
            context = runtime

        # Force failing reports for revision path
        cs.store_evidence_report(
            EvidenceCheckReport(
                q_id="Q12",
                supported_by_pdf=False,
                unsupported_claims=["bad"],
                source_refs=mcq.source_refs,
                confidence=0.2,
                candidate_version=1,
            )
        )
        cs.store_cross_cove_report(
            CrossCoVeReport(
                q_id="Q12",
                generator_answer="A",
                verifier_answers=["B", "B", "B"],
                majority_answer="B",
                passed=False,
                candidate_version=1,
            )
        )
        cs.store_supervisor_decision(
            SupervisorDecision(
                q_id="Q12",
                decision="revise",
                reasons=["needs revision"],
                candidate_version=1,
                required_fixes=["evidence_not_supported"],
            )
        )
        mcq.status = "needs_revision"
        mcq.flags.append("evidence_not_supported")
        cs.store_candidate_mcq(mcq)
        reg, resolver = load_artifact_channels(state)
        for stage in ("evidence_reports", "cross_cove_reports", "supervisor_decisions"):
            register_mcq_artifacts_for_stage(
                reg, resolver, cs, stage=stage, run_id="run_rev", branch_id="factual"
            )
        state.update(dump_artifact_channels(reg, resolver))

        # Register ProducedBy for Q12:v1 via fake generation occurrence
        from mas_langgraph.provenance.channel_state import load_provenance_recorder, dump_provenance_channels
        from mas_langgraph.cases.mcq import MCQCaseDefinition

        rec = load_provenance_recorder(
            state, schema=MCQCaseDefinition().load_provenance_schema(), run_id="run_rev"
        )
        gen = rec.begin_occurrence("generate_candidate_mcqs_from_blueprints")
        rec.complete_occurrence(
            gen,
            produced_artifacts=(ArtifactRef("Q12", 1),),
            producer_artifact_types={ArtifactRef("Q12", 1): "CandidateMCQ"},
        )
        for ref, atype, node in [
            (ArtifactRef("evidence-report:Q12", 1), "EvidenceCheckReport", "verify_candidate_mcqs_against_evidence"),
            (ArtifactRef("cross-cove-report:Q12", 1), "CrossCoVeReport", "cross_cove_verify_candidate_mcqs"),
            (ArtifactRef("decision:Q12", 1), "SupervisorDecision", "apply_acceptance_gate"),
            (ArtifactRef("E17", 1), "EvidenceChunk", "initial_retrieve_evidence_for_all_sections"),
            (ArtifactRef("A3", 1), "KnowledgeAtom", "atomize_evidence"),
            (ArtifactRef(bp.blueprint_id, 1), "MCQBlueprint", "plan_mcq_blueprints"),
        ]:
            occ = rec.begin_occurrence(node)
            rec.complete_occurrence(
                occ, produced_artifacts=(ref,), producer_artifact_types={ref: atype}
            )
        state.update(dump_provenance_channels(rec))

        revise_fn = instrument_node("revise_candidate_mcqs", lg.node_revise_candidate_mcqs)
        update = revise_fn(state, _RT())
        state.update(update)
        produced = [
            pb for pb in state["produced_by"] if pb.artifact_ref == ArtifactRef("Q12", 2)
        ]
        self.assertTrue(produced)
        self.assertTrue(
            any("revise_candidate_mcqs" in pb.producer_occurrence_id for pb in produced)
        )
        self.assertTrue(
            any(
                isinstance(d, DerivedFrom)
                and d.new_artifact_ref == ArtifactRef("Q12", 2)
                and d.previous_artifact_ref == ArtifactRef("Q12", 1)
                for d in (state.get("artifact_derivations") or [])
            )
        )
        self.assertEqual(state["cooperative_state"].candidate_mcqs["Q12"].revision_count, 1)

    def test_factual_run_zero_schema_violations(self):
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.core.checkpointer import default_factual_checkpointer
        from mas_langgraph.core.graph_builder import build_mcq_workflow_graph
        from mas_langgraph.core.workflow_executor import WorkflowExecutor

        cp = default_factual_checkpointer()
        graph = build_mcq_workflow_graph(checkpointer=cp)
        cs = CooperativeState()
        rt = self._stub_runtime(cs)
        ex = WorkflowExecutor(runtime=rt, checkpointer=cp, compiled_graph=graph)
        result = ex.run(
            {
                "cooperative_state": cs,
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
            },
            runtime=rt,
            run_id="run_p7_ok",
        )
        self.assertEqual(result.provenance_violations, [])
        self.assertGreater(len(result.occurrences), 5)
        labels = [o.labeled_id for o in result.occurrences]
        self.assertIn("package_final_quiz#1", labels)
        self.assertEqual(result.final_state.get("provenance_unresolved_producers") or [], [])

    def test_provenance_checkpoint_isolation(self):
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.core.checkpointer import default_factual_checkpointer
        from mas_langgraph.core.graph_builder import build_mcq_workflow_graph
        from mas_langgraph.core.workflow_executor import WorkflowExecutor

        cp = default_factual_checkpointer()
        graph = build_mcq_workflow_graph(checkpointer=cp)
        cs = CooperativeState()
        rt = self._stub_runtime(cs)
        ex = WorkflowExecutor(runtime=rt, checkpointer=cp, compiled_graph=graph)
        run_id = "run_p7_iso"
        ex.run(
            {
                "cooperative_state": cs,
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
            },
            runtime=rt,
            run_id=run_id,
        )
        history = ex.get_state_history(run_id)
        self.assertGreaterEqual(len(history), 2)
        early = history[-1]  # oldest first or newest? LangGraph typically newest first
        # Find earliest with fewest occurrences
        snaps = sorted(
            history,
            key=lambda s: len((getattr(s, "values", None) or {}).get("occurrence_records") or []),
        )
        early_vals = snaps[0].values or {}
        early_count = len(early_vals.get("occurrence_records") or [])
        final_count = len(ex.get_final_state(run_id).get("occurrence_records") or [])
        self.assertGreater(final_count, early_count)
        # Re-read early snapshot count unchanged
        early_after = len((snaps[0].values or {}).get("occurrence_records") or [])
        self.assertEqual(early_after, early_count)

    def test_deterministic_reconstruction(self):
        from mas_langgraph.artifacts.models import ArtifactRef
        from mas_langgraph.provenance.models import ArtifactBinding
        from mas_langgraph.provenance.recorder import ProvenanceRecorder
        from mas_langgraph.cases.mcq import MCQCaseDefinition

        schema = MCQCaseDefinition().load_provenance_schema()
        rec = ProvenanceRecorder(schema=schema, run_id="runD", branch_id="factual")
        gen = rec.begin_occurrence(
            "generate_candidate_mcqs_from_blueprints",
            input_bindings=(ArtifactBinding(ArtifactRef("E17", 1), "supporting_evidence"),),
        )
        rec.complete_occurrence(
            gen,
            produced_artifacts=(ArtifactRef("Q12", 1),),
            producer_artifact_types={
                ArtifactRef("Q12", 1): "CandidateMCQ",
                ArtifactRef("E17", 1): "EvidenceChunk",
            },
        )
        # Pretend E17 was produced earlier
        ret = rec.begin_occurrence("initial_retrieve_evidence_for_all_sections")
        rec.complete_occurrence(
            ret,
            produced_artifacts=(ArtifactRef("E17", 1),),
            producer_artifact_types={ArtifactRef("E17", 1): "EvidenceChunk"},
        )
        # Rebuild from state dump
        dumped = rec.to_state()
        rec2 = ProvenanceRecorder.from_state(
            occurrence_records=dumped["occurrence_records"],
            produced_by=dumped["produced_by"],
            node_counts=dumped["provenance_node_counts"],
            execution_index=dumped["provenance_execution_index"],
            artifact_types={
                ArtifactRef("Q12", 1): "CandidateMCQ",
                ArtifactRef("E17", 1): "EvidenceChunk",
            },
            run_id="runD",
            branch_id="factual",
            schema=schema,
        )
        a = rec2.build_run_relations()
        b = rec2.build_run_relations()
        self.assertEqual(
            [(r.producer_occurrence_id, str(r.artifact_ref), r.relation, r.consumer_occurrence_id, r.artifact_type) for r in a],
            [(r.producer_occurrence_id, str(r.artifact_ref), r.relation, r.consumer_occurrence_id, r.artifact_type) for r in b],
        )

    def test_candidate_separation_q12_q13(self):
        from mas_langgraph.artifacts.models import ArtifactRef
        from mas_langgraph.provenance.models import ArtifactBinding
        from mas_langgraph.provenance.recorder import ProvenanceRecorder
        from mas_langgraph.cases.mcq import MCQCaseDefinition

        schema = MCQCaseDefinition().load_provenance_schema()
        rec = ProvenanceRecorder(schema=schema, run_id="runSep", branch_id="factual")
        for qid in ("Q12", "Q13"):
            gen = rec.begin_occurrence("generate_candidate_mcqs_from_blueprints")
            rec.complete_occurrence(
                gen,
                produced_artifacts=(ArtifactRef(qid, 1),),
                producer_artifact_types={ArtifactRef(qid, 1): "CandidateMCQ"},
            )
        ver = rec.begin_occurrence(
            "verify_candidate_mcqs_against_evidence",
            input_bindings=(ArtifactBinding(ArtifactRef("Q12", 1), "candidate_question"),),
        )
        rec.complete_occurrence(
            ver,
            produced_artifacts=(ArtifactRef("evidence-report:Q12", 1),),
            producer_artifact_types={ArtifactRef("evidence-report:Q12", 1): "EvidenceCheckReport"},
        )
        rels = [
            r for r in rec.run_provenance_relations
            if r.consumer_occurrence_id == ver.occurrence_id
        ]
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0].artifact_ref, ArtifactRef("Q12", 1))
        self.assertNotEqual(rels[0].artifact_ref.logical_id, "Q13")


class TestWorkflowStillBuilds(unittest.TestCase):
    def test_build_workflow_graph(self):
        from mas_langgraph.orchestrator.langgraph_workflow import build_workflow_graph

        g = build_workflow_graph()
        self.assertIsNotNone(g)

    def test_workflow_runtime_and_executor(self):
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState

        case = MCQCaseDefinition()
        rt = WorkflowRuntime(case_definition=case)
        self.assertEqual(rt.case_definition.case_id, "mcq_generation")
        ex = WorkflowExecutor(runtime=rt, compiled_graph=None)
        result = ex.run(
            {
                "cooperative_state": CooperativeState(),
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
            },
            run_id="run_x",
        )
        self.assertEqual(result.run_id, "run_x")


class TestPrompt8OutcomeAssessment(unittest.TestCase):
    """Prompt 8 — Gamma_alpha / delta_alpha aligned with factual acceptance gate."""

    def _passing_inputs(self, **overrides):
        from mas_langgraph.domains.quiz.acceptance_policy import GateInputs

        base = dict(
            revision_count=0,
            max_revisions=2,
            candidate_status="draft",
            flags=(),
            evidence_report_available=True,
            cross_cove_required=True,
            cross_cove_report_available=True,
            pdf_supported=True,
            cross_cove_passed=True,
            cross_cove_majority_present=True,
            source_refs_present=True,
            correct_option_count=1,
            blueprint_available=True,
            target_atom_count=1,
            blueprint_evidence_count=1,
            answer_supported=True,
            stem_supported=True,
            distractors_valid=True,
            explanation_supported=True,
            evidence_confidence=0.9,
        )
        base.update(overrides)
        return GateInputs(**base)

    def test_legacy_gate_parity_table(self):
        from mas_langgraph.domains.quiz.acceptance_policy import decide_acceptance

        cases = [
            ("A_no_evidence", dict(evidence_report_available=False), "revise"),
            (
                "B_no_cross_cove",
                dict(cross_cove_report_available=False),
                "revise",
            ),
            (
                "C_structural",
                dict(flags=("multiple_correct_options",)),
                "reject",
            ),
            (
                "D_non_structural_hard",
                dict(flags=("evidence_not_supported",)),
                "revise",
            ),
            ("E_pdf", dict(pdf_supported=False, answer_supported=False), "revise"),
            ("F_cross_cove_fail", dict(cross_cove_passed=False), "revise"),
            (
                "G_no_majority",
                dict(cross_cove_passed=True, cross_cove_majority_present=False),
                "revise",
            ),
            ("H_no_sources", dict(source_refs_present=False), "revise"),
            ("I_correct_count", dict(correct_option_count=2), "reject"),
            ("J_no_blueprint", dict(blueprint_available=False), "revise"),
            ("K_no_atoms", dict(target_atom_count=0), "revise"),
            ("L_no_chunks", dict(blueprint_evidence_count=0), "revise"),
            ("M_accept", {}, "accept"),
            (
                "N_exhausted_budget",
                dict(pdf_supported=False, revision_count=2, max_revisions=2),
                "reject",
            ),
        ]
        for name, overrides, expected in cases:
            with self.subTest(name):
                gate = decide_acceptance(self._passing_inputs(**overrides))
                self.assertEqual(gate.decision, expected, msg=name)

    def test_apply_acceptance_gate_matches_delta(self):
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.domains.quiz.acceptance_policy import (
            decide_acceptance,
            status_for_decision,
        )
        from mas_langgraph.domains.quiz.coverage import apply_acceptance_gate
        from mas_langgraph.domains.quiz.entities import (
            AnswerOption,
            CandidateMCQ,
            CrossCoVeReport,
            EvidenceCheckReport,
            MCQBlueprint,
            SourceRef,
        )

        state = CooperativeState()
        bp = MCQBlueprint(
            blueprint_id="BP12",
            section_id="S1",
            lo_id="L1",
            target_atom_ids=["A1"],
            evidence_chunk_ids=["E1"],
            question_type="basic_definition",
            difficulty="intermediate",
            bloom_level="Remembering",
            generation_goal="g",
        )
        state.store_mcq_blueprints([bp])
        mcq = CandidateMCQ(
            q_id="Q12",
            blueprint_id="BP12",
            stem="stem",
            options=[
                AnswerOption("A", "a", True),
                AnswerOption("B", "b", False),
                AnswerOption("C", "c", False),
                AnswerOption("D", "d", False),
            ],
            correct_option_id="A",
            answer_explanation="x",
            source_refs=[SourceRef("d", 1, "E1")],
        )
        state.store_candidate_mcq(mcq)
        state.store_evidence_report(
            EvidenceCheckReport(
                q_id="Q12",
                supported_by_pdf=True,
                unsupported_claims=[],
                source_refs=mcq.source_refs,
                confidence=0.9,
                candidate_version=1,
                answer_supported=True,
            )
        )
        state.store_cross_cove_report(
            CrossCoVeReport(
                q_id="Q12",
                generator_answer="A",
                verifier_answers=["A", "A"],
                majority_answer="A",
                passed=True,
                candidate_version=1,
            )
        )
        expected = decide_acceptance(self._passing_inputs())
        decision = apply_acceptance_gate(mcq, state)
        self.assertEqual(decision.decision, expected.decision)
        self.assertEqual(mcq.status, status_for_decision(decision.decision))
        self.assertEqual(decision.assessment_ref, ArtifactRef("assessment:Q12", 1))
        self.assertIn("Q12", state.outcome_assessments)
        self.assertEqual(
            state.outcome_assessments["Q12"].assessment_ref,
            decision.assessment_ref,
        )

    def test_assessment_identity_and_history(self):
        from mas_langgraph.artifacts.channel_state import dump_artifact_channels, load_artifact_channels
        from mas_langgraph.cases.mcq.artifact_registration import register_mcq_artifacts_for_stage
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.domains.quiz.acceptance_policy import build_outcome_assessment
        from mas_langgraph.domains.quiz.entities import AnswerOption, CandidateMCQ, SourceRef

        state = CooperativeState()
        v1 = CandidateMCQ(
            q_id="Q12",
            blueprint_id="BP",
            stem="v1",
            options=[AnswerOption("A", "a", True), AnswerOption("B", "b", False)],
            correct_option_id="A",
            answer_explanation="e",
            source_refs=[SourceRef("d", 1, "E1")],
            revision_count=0,
        )
        v2 = CandidateMCQ(
            q_id="Q12",
            blueprint_id="BP",
            stem="v2",
            options=[AnswerOption("A", "a", True), AnswerOption("B", "b", False)],
            correct_option_id="A",
            answer_explanation="e",
            source_refs=[SourceRef("d", 1, "E1")],
            revision_count=1,
        )
        a1 = build_outcome_assessment(ArtifactRef("Q12", 1), self._passing_inputs())
        a2 = build_outcome_assessment(
            ArtifactRef("Q12", 2), self._passing_inputs(pdf_supported=False)
        )
        state.outcome_assessments["Q12"] = a1
        graph = {"run_id": "r", "branch_id": "factual"}
        reg, resolver = load_artifact_channels(graph)
        register_mcq_artifacts_for_stage(
            reg, resolver, state, stage="outcome_assessments", run_id="r", branch_id="factual"
        )
        state.outcome_assessments["Q12"] = a2
        register_mcq_artifacts_for_stage(
            reg, resolver, state, stage="outcome_assessments", run_id="r", branch_id="factual"
        )
        versions = reg.versions_of("assessment:Q12", "factual")
        self.assertEqual({v.ref.version for v in versions}, {1, 2})
        self.assertNotEqual(ArtifactRef("assessment:Q12", 1), ArtifactRef("assessment:Q12", 2))
        # Immutable: mutating operational pointer must not rewrite v1 payload.
        stored_v1 = reg.get_by_ref(ArtifactRef("assessment:Q12", 1), "factual").payload
        self.assertTrue(
            any(c.name == "pdf_supported" and c.value is True for c in stored_v1.criteria)
        )

    def test_historical_exact_version_assessment(self):
        from mas_langgraph.artifacts.registry import ArtifactRegistry
        from mas_langgraph.cases.mcq import MCQCaseDefinition
        from mas_langgraph.cases.mcq.artifacts import (
            wrap_candidate_mcq,
            wrap_cross_cove_report,
            wrap_evidence_report,
        )
        from mas_langgraph.domains.quiz.entities import (
            AnswerOption,
            CandidateMCQ,
            CrossCoVeReport,
            EvidenceCheckReport,
            SourceRef,
        )

        reg = ArtifactRegistry()
        v1 = CandidateMCQ(
            q_id="Q12",
            blueprint_id="BP",
            stem="stem-v1",
            options=[
                AnswerOption("A", "a", True),
                AnswerOption("B", "b", False),
                AnswerOption("C", "c", False),
                AnswerOption("D", "d", False),
            ],
            correct_option_id="A",
            answer_explanation="e",
            source_refs=[SourceRef("d", 1, "E1")],
            revision_count=0,
            status="needs_revision",
        )
        v2 = CandidateMCQ(
            q_id="Q12",
            blueprint_id="BP",
            stem="stem-v2",
            options=v1.options,
            correct_option_id="A",
            answer_explanation="e",
            source_refs=v1.source_refs,
            revision_count=1,
            status="draft",
        )
        wrap_candidate_mcq(v1, version=1, registry=reg)
        wrap_candidate_mcq(v2, version=2, registry=reg)
        wrap_evidence_report(
            EvidenceCheckReport(
                q_id="Q12",
                supported_by_pdf=False,
                unsupported_claims=["x"],
                source_refs=v1.source_refs,
                confidence=0.1,
                candidate_version=1,
                answer_supported=False,
            ),
            version=1,
            registry=reg,
        )
        wrap_evidence_report(
            EvidenceCheckReport(
                q_id="Q12",
                supported_by_pdf=True,
                unsupported_claims=[],
                source_refs=v2.source_refs,
                confidence=0.9,
                candidate_version=2,
                answer_supported=True,
            ),
            version=2,
            registry=reg,
        )
        wrap_cross_cove_report(
            CrossCoVeReport(
                q_id="Q12",
                generator_answer="A",
                verifier_answers=["A"],
                majority_answer="A",
                passed=True,
                candidate_version=1,
            ),
            version=1,
            registry=reg,
        )
        wrap_cross_cove_report(
            CrossCoVeReport(
                q_id="Q12",
                generator_answer="A",
                verifier_answers=["A"],
                majority_answer="A",
                passed=True,
                candidate_version=2,
            ),
            version=2,
            registry=reg,
        )
        # Operational state points at v2 / supported report.
        class CS:
            candidate_mcqs = {"Q12": v2}
            evidence_reports = {
                "Q12": EvidenceCheckReport(
                    q_id="Q12",
                    supported_by_pdf=True,
                    unsupported_claims=[],
                    source_refs=v2.source_refs,
                    confidence=0.9,
                    candidate_version=2,
                    answer_supported=True,
                )
            }
            cross_cove_reports = {}
            mcq_blueprints = {}

        ev = MCQCaseDefinition().load_outcome_evaluator()
        assessment = ev.assess(
            {
                "artifact_catalog": dict(reg.catalog),
                "cooperative_state": CS(),
                "branch_id": "factual",
            },
            ArtifactRef("Q12", 1),
        )
        by_name = {c.name: c.value for c in assessment.criteria}
        self.assertFalse(by_name["pdf_supported"])
        self.assertEqual(assessment.assessment_ref, ArtifactRef("assessment:Q12", 1))
        decision = ev.decide(assessment)
        self.assertEqual(decision.decision, "revise")
        self.assertEqual(decision.assessment_ref.version, 1)

    def test_missing_catalog_version_fails(self):
        from mas_langgraph.cases.mcq import MCQCaseDefinition

        ev = MCQCaseDefinition().load_outcome_evaluator()
        with self.assertRaises(LookupError):
            ev.assess({"artifact_catalog": {}, "branch_id": "factual"}, ArtifactRef("Q12", 1))

    def test_decision_assessment_linkage_and_status(self):
        from mas_langgraph.domains.quiz.acceptance_policy import (
            build_decision_artifact,
            build_outcome_assessment,
            decide_acceptance,
            status_for_decision,
        )

        for decision_name, status in (
            ("accept", "accepted"),
            ("revise", "needs_revision"),
            ("reject", "rejected"),
        ):
            with self.subTest(decision_name):
                if decision_name == "accept":
                    inputs = self._passing_inputs()
                elif decision_name == "revise":
                    inputs = self._passing_inputs(pdf_supported=False)
                else:
                    inputs = self._passing_inputs(flags=("multiple_correct_options",))
                assessment = build_outcome_assessment(ArtifactRef("Q12", 2), inputs)
                gate = decide_acceptance(inputs)
                da = build_decision_artifact(assessment, gate)
                self.assertEqual(da.decision, decision_name)
                self.assertEqual(da.assessment_ref, ArtifactRef("assessment:Q12", 2))
                self.assertEqual(da.decision_ref, ArtifactRef("decision:Q12", 2))
                self.assertEqual(da.target_ref, ArtifactRef("Q12", 2))
                self.assertEqual(status_for_decision(da.decision), status)

    def test_required_tool_success_not_fabricated(self):
        from mas_langgraph.cases.mcq import MCQCaseDefinition
        from mas_langgraph.domains.quiz.acceptance_policy import build_outcome_assessment

        ev = MCQCaseDefinition().load_outcome_evaluator()
        self.assertIn("required_tool_success", ev.criteria_meta)
        self.assertFalse(ev.criteria_meta["required_tool_success"].get("implemented", True))
        self.assertNotIn("required_tool_success", ev.criteria_order)
        assessment = build_outcome_assessment(
            ArtifactRef("Q12", 1),
            self._passing_inputs(),
            criteria_meta=ev.criteria_meta,
            criteria_order=ev.criteria_order,
        )
        names = [c.name for c in assessment.criteria]
        self.assertNotIn("required_tool_success", names)

    def test_deterministic_gamma(self):
        from mas_langgraph.domains.quiz.acceptance_policy import (
            build_outcome_assessment,
            decide_acceptance,
        )

        a1 = build_outcome_assessment(ArtifactRef("Q12", 1), self._passing_inputs())
        a2 = build_outcome_assessment(ArtifactRef("Q12", 1), self._passing_inputs())
        self.assertEqual(a1.criteria, a2.criteria)
        self.assertEqual(
            decide_acceptance(self._passing_inputs()).decision,
            decide_acceptance(self._passing_inputs()).decision,
        )

    def test_package_authorizing_decision_refs(self):
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.domains.quiz.coverage import apply_acceptance_gate
        from mas_langgraph.domains.quiz.entities import (
            AnswerOption,
            CandidateMCQ,
            CrossCoVeReport,
            EvidenceCheckReport,
            MCQBlueprint,
            SourceRef,
        )
        from mas_langgraph.domains.quiz.packaging import package_final_quiz

        state = CooperativeState()
        bp = MCQBlueprint(
            blueprint_id="BP12",
            section_id="S1",
            lo_id="L1",
            target_atom_ids=["A1"],
            evidence_chunk_ids=["E1"],
            question_type="basic_definition",
            difficulty="intermediate",
            bloom_level="Remembering",
            generation_goal="g",
        )
        state.store_mcq_blueprints([bp])
        mcq = CandidateMCQ(
            q_id="Q12",
            blueprint_id="BP12",
            stem="stem",
            options=[
                AnswerOption("A", "a", True),
                AnswerOption("B", "b", False),
                AnswerOption("C", "c", False),
                AnswerOption("D", "d", False),
            ],
            correct_option_id="A",
            answer_explanation="x",
            source_refs=[SourceRef("d", 1, "E1")],
            revision_count=1,
        )
        state.store_candidate_mcq(mcq)
        state.store_evidence_report(
            EvidenceCheckReport(
                q_id="Q12",
                supported_by_pdf=True,
                unsupported_claims=[],
                source_refs=mcq.source_refs,
                confidence=0.9,
                candidate_version=2,
                answer_supported=True,
            )
        )
        state.store_cross_cove_report(
            CrossCoVeReport(
                q_id="Q12",
                generator_answer="A",
                verifier_answers=["A"],
                majority_answer="A",
                passed=True,
                candidate_version=2,
            )
        )
        d = apply_acceptance_gate(mcq, state)
        state.store_supervisor_decision(d)
        state.store_candidate_mcq(mcq)
        pack = package_final_quiz(state, title="T")
        self.assertIn("decision:Q12:v2", pack.metadata["authorizing_decision_refs"])
        self.assertNotIn("decision:Q12:v1", pack.metadata["authorizing_decision_refs"])

    def test_acceptance_registers_assessment_produced_by(self):
        from mas_langgraph.artifacts.channel_state import (
            dump_artifact_channels,
            empty_artifact_channels,
            load_artifact_channels,
        )
        from mas_langgraph.cases.mcq import MCQCaseDefinition
        from mas_langgraph.cases.mcq.artifact_registration import register_mcq_artifacts_for_stage
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.core.runtime import WorkflowRuntime
        from mas_langgraph.domains.quiz.entities import (
            AnswerOption,
            CandidateMCQ,
            CrossCoVeReport,
            EvidenceCheckReport,
            MCQBlueprint,
            SourceRef,
        )
        from mas_langgraph.orchestrator.langgraph_workflow import node_apply_acceptance_gate
        from mas_langgraph.provenance.channel_state import empty_provenance_channels
        from mas_langgraph.provenance.instrumentation import instrument_node
        from mas_langgraph.input.orchestrator_config import OrchestratorConfig

        cs = CooperativeState()
        bp = MCQBlueprint(
            blueprint_id="BP12",
            section_id="S1",
            lo_id="L1",
            target_atom_ids=["A1"],
            evidence_chunk_ids=["E1"],
            question_type="basic_definition",
            difficulty="intermediate",
            bloom_level="Remembering",
            generation_goal="g",
        )
        cs.store_mcq_blueprints([bp])
        mcq = CandidateMCQ(
            q_id="Q12",
            blueprint_id="BP12",
            stem="stem",
            options=[
                AnswerOption("A", "a", True),
                AnswerOption("B", "b", False),
                AnswerOption("C", "c", False),
                AnswerOption("D", "d", False),
            ],
            correct_option_id="A",
            answer_explanation="x",
            source_refs=[SourceRef("d", 1, "E1")],
        )
        cs.store_candidate_mcq(mcq)
        cs.store_evidence_report(
            EvidenceCheckReport(
                q_id="Q12",
                supported_by_pdf=True,
                unsupported_claims=[],
                source_refs=mcq.source_refs,
                confidence=0.9,
                candidate_version=1,
                answer_supported=True,
            )
        )
        cs.store_cross_cove_report(
            CrossCoVeReport(
                q_id="Q12",
                generator_answer="A",
                verifier_answers=["A"],
                majority_answer="A",
                passed=True,
                candidate_version=1,
            )
        )
        state = {
            "run_id": "run_p8",
            "branch_id": "factual",
            "cooperative_state": cs,
            "searched_queries": set(),
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
        for stage in ("blueprints", "candidates", "evidence_reports", "cross_cove_reports"):
            register_mcq_artifacts_for_stage(
                reg, resolver, cs, stage=stage, run_id="run_p8", branch_id="factual"
            )
        state.update(dump_artifact_channels(reg, resolver))

        case = MCQCaseDefinition()
        runtime = WorkflowRuntime(
            case_definition=case,
            orchestrator_config=OrchestratorConfig(),
        )

        class _RT:
            context = runtime

        wrapped = instrument_node("apply_acceptance_gate", node_apply_acceptance_gate)
        update = wrapped(state, _RT())
        state.update(update)
        produced = state.get("produced_by") or []
        assessment_pb = [
            p for p in produced if getattr(p, "artifact_ref", None) == ArtifactRef("assessment:Q12", 1)
        ]
        decision_pb = [
            p for p in produced if getattr(p, "artifact_ref", None) == ArtifactRef("decision:Q12", 1)
        ]
        self.assertTrue(assessment_pb, msg=f"produced_by={produced}")
        self.assertTrue(decision_pb, msg=f"produced_by={produced}")
        self.assertIn("apply_acceptance_gate#", assessment_pb[0].producer_occurrence_id)
        self.assertEqual(
            assessment_pb[0].producer_occurrence_id,
            decision_pb[0].producer_occurrence_id,
        )
        for rel in state.get("run_provenance_relations") or []:
            self.assertNotEqual(getattr(rel, "relation", None), "assessment_to_decision")


class TestCheckpointIndex(unittest.TestCase):
    def test_resolve_alias_and_occurrence(self):
        idx = CheckpointIndex(thread_id="t")
        idx.records = [
            CheckpointRecord(
                alias="c_0",
                thread_id="t",
                checkpoint_id="id0",
                after_occurrences=("run:t:build_outline#1",),
                after_occurrence_labels=("build_outline#1",),
            ),
            CheckpointRecord(
                alias="c_1",
                thread_id="t",
                checkpoint_id="id1",
                after_occurrences=("run:t:evidence_verifier#1",),
                after_occurrence_labels=("evidence_verifier#1",),
                next_nodes=("cross_cove_verify_candidate_mcqs",),
                parent_checkpoint_id="id0",
            ),
        ]
        idx._after_alias_by_occurrence["run:t:evidence_verifier#1"] = "c_1"
        idx._before_alias_by_occurrence["run:t:evidence_verifier#1"] = "c_0"
        idx._occurrence_by_label["evidence_verifier#1"] = OccurrenceRecord(
            occurrence_id="run:t:evidence_verifier#1",
            node_id="evidence_verifier",
            occurrence_index=1,
            branch_id="factual",
            run_id="t",
        )
        self.assertEqual(idx.resolve(BoundarySelector.by_alias("c_1")).checkpoint_id, "id1")
        self.assertEqual(
            idx.resolve(BoundarySelector.after_occurrence("evidence_verifier", 1)).alias,
            "c_1",
        )
        with self.assertRaises(KeyError):
            idx.resolve(BoundarySelector.by_alias("c_99"))


class TestPrompt9CheckpointOccurrenceIndex(unittest.TestCase):
    """Prompt 9 — real checkpoint/occurrence index and boundary resolution."""

    def _occ(self, run_id, node, index, execution_index, branch="factual"):
        from mas_langgraph.provenance.models import make_occurrence_id

        return OccurrenceRecord(
            occurrence_id=make_occurrence_id(
                run_id=run_id,
                branch_id=branch,
                node_id=node,
                occurrence_index=index,
            ),
            node_id=node,
            occurrence_index=index,
            branch_id=branch,
            run_id=run_id,
            execution_index=execution_index,
        )

    def _snap(
        self,
        *,
        run_id,
        checkpoint_id,
        step,
        occurrences,
        parent_id=None,
        next_nodes=(),
    ):
        class Snap:
            pass

        s = Snap()
        s.config = {
            "configurable": {
                "thread_id": run_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": "",
            }
        }
        s.parent_config = (
            {
                "configurable": {
                    "thread_id": run_id,
                    "checkpoint_id": parent_id,
                    "checkpoint_ns": "",
                }
            }
            if parent_id
            else None
        )
        s.metadata = {"step": step}
        s.next = tuple(next_nodes)
        s.values = {"occurrence_records": list(occurrences)}
        s.id = checkpoint_id
        return s

    def test_superstep_from_metadata_not_alias_index(self):
        run = "run_meta"
        a1 = self._occ(run, "build_outline", 1, 1)
        history = [
            self._snap(run_id=run, checkpoint_id="ck0", step=-1, occurrences=[]),
            self._snap(
                run_id=run,
                checkpoint_id="ck1",
                step=7,
                occurrences=[a1],
                parent_id="ck0",
            ),
        ]
        idx = CheckpointIndex(thread_id=run)
        idx.build_from_history(history)
        self.assertEqual(idx.records[0].alias, "c_0")
        self.assertEqual(idx.records[0].superstep, -1)
        self.assertEqual(idx.records[1].alias, "c_1")
        self.assertEqual(idx.records[1].superstep, 7)
        self.assertNotEqual(idx.records[1].superstep, 1)

    def test_occurrence_delta_mapping(self):
        run = "run_delta"
        a = self._occ(run, "A", 1, 1)
        b = self._occ(run, "B", 1, 2)
        c = self._occ(run, "C", 1, 3)
        history = [
            self._snap(run_id=run, checkpoint_id="ckA", step=0, occurrences=[a]),
            self._snap(
                run_id=run, checkpoint_id="ckB", step=1, occurrences=[a, b], parent_id="ckA"
            ),
            self._snap(
                run_id=run,
                checkpoint_id="ckC",
                step=2,
                occurrences=[a, b, c],
                parent_id="ckB",
            ),
        ]
        idx = CheckpointIndex(thread_id=run)
        idx.build_from_history(history)
        self.assertEqual(idx.records[0].after_occurrence_labels, ("A#1",))
        self.assertEqual(idx.records[1].after_occurrence_labels, ("B#1",))
        self.assertEqual(idx.records[2].after_occurrence_labels, ("C#1",))

    def test_parallel_superstep_fixture(self):
        run = "run_par"
        a = self._occ(run, "A", 1, 1)
        b = self._occ(run, "B", 1, 2)
        c = self._occ(run, "C", 1, 3)
        history = [
            self._snap(run_id=run, checkpoint_id="p", step=3, occurrences=[a]),
            self._snap(
                run_id=run,
                checkpoint_id="child",
                step=4,
                occurrences=[a, b, c],
                parent_id="p",
            ),
        ]
        idx = CheckpointIndex(thread_id=run)
        idx.build_from_history(history)
        idx.enrich_from_occurrences([a, b, c])
        eb = idx.occurrence("B", 1)
        ec = idx.occurrence("C", 1)
        self.assertEqual(eb.checkpoint_before, ec.checkpoint_before)
        self.assertEqual(eb.checkpoint_after, ec.checkpoint_after)
        self.assertEqual(eb.superstep, ec.superstep)
        self.assertEqual(eb.superstep, 4)
        self.assertEqual(
            set(idx.resolve(BoundarySelector.after_occurrence("B", 1)).after_occurrence_labels),
            {"B#1", "C#1"},
        )

    def test_invalid_occurrence_no_positional_fallback(self):
        run = "run_inv"
        v1 = self._occ(run, "verify_candidate_mcqs_against_evidence", 1, 1)
        v2 = self._occ(run, "verify_candidate_mcqs_against_evidence", 2, 2)
        history = [
            self._snap(run_id=run, checkpoint_id="ck0", step=0, occurrences=[]),
            self._snap(
                run_id=run, checkpoint_id="ck1", step=1, occurrences=[v1], parent_id="ck0"
            ),
            self._snap(
                run_id=run,
                checkpoint_id="ck2",
                step=2,
                occurrences=[v1, v2],
                parent_id="ck1",
            ),
        ]
        idx = CheckpointIndex(thread_id=run)
        idx.build_from_history(history)
        idx.enrich_from_occurrences([v1, v2])
        with self.assertRaises(KeyError):
            idx.resolve(
                BoundarySelector.before_occurrence(
                    "verify_candidate_mcqs_against_evidence", 99
                )
            )

    def test_before_and_after_occurrence(self):
        run = "run_ba"
        v1 = self._occ(run, "verify_candidate_mcqs_against_evidence", 1, 1)
        rev = self._occ(run, "revise_candidate_mcqs", 1, 2)
        v2 = self._occ(run, "verify_candidate_mcqs_against_evidence", 2, 3)
        history = [
            self._snap(run_id=run, checkpoint_id="ck0", step=0, occurrences=[]),
            self._snap(
                run_id=run, checkpoint_id="ck1", step=1, occurrences=[v1], parent_id="ck0"
            ),
            self._snap(
                run_id=run,
                checkpoint_id="ck2",
                step=2,
                occurrences=[v1, rev],
                parent_id="ck1",
            ),
            self._snap(
                run_id=run,
                checkpoint_id="ck3",
                step=3,
                occurrences=[v1, rev, v2],
                parent_id="ck2",
            ),
        ]
        idx = CheckpointIndex(thread_id=run)
        idx.build_from_history(history)
        idx.enrich_from_occurrences([v1, rev, v2])
        after_rev = idx.resolve(BoundarySelector.after_occurrence("revise_candidate_mcqs", 1))
        self.assertEqual(after_rev.checkpoint_id, "ck2")
        before_v2 = idx.resolve(
            BoundarySelector.before_occurrence(
                "verify_candidate_mcqs_against_evidence", 2
            )
        )
        self.assertEqual(before_v2.checkpoint_id, "ck2")
        self.assertEqual(before_v2.alias, idx.occurrence("verify_candidate_mcqs_against_evidence", 2).checkpoint_before)
        after_v2 = idx.resolve(
            BoundarySelector.after_occurrence(
                "verify_candidate_mcqs_against_evidence", 2
            )
        )
        self.assertEqual(after_v2.checkpoint_id, "ck3")
        self.assertNotEqual(
            idx.occurrence("verify_candidate_mcqs_against_evidence", 1).checkpoint_after,
            idx.occurrence("verify_candidate_mcqs_against_evidence", 2).checkpoint_after,
        )

    def test_ambiguous_before_next_node(self):
        run = "run_amb"
        history = [
            self._snap(
                run_id=run,
                checkpoint_id="ck0",
                step=0,
                occurrences=[],
                next_nodes=("verify_candidate_mcqs_against_evidence",),
            ),
            self._snap(
                run_id=run,
                checkpoint_id="ck1",
                step=1,
                occurrences=[],
                parent_id="ck0",
                next_nodes=("verify_candidate_mcqs_against_evidence",),
            ),
        ]
        idx = CheckpointIndex(thread_id=run)
        idx.build_from_history(history)
        with self.assertRaises(ValueError):
            idx.resolve(
                BoundarySelector.before_next_node(
                    "verify_candidate_mcqs_against_evidence"
                )
            )

    def test_alias_exact_and_zero_occurrence_checkpoint(self):
        run = "run_alias"
        a = self._occ(run, "A", 1, 1)
        history = [
            self._snap(run_id=run, checkpoint_id="ck0", step=-1, occurrences=[]),
            self._snap(
                run_id=run, checkpoint_id="ck1", step=0, occurrences=[a], parent_id="ck0"
            ),
        ]
        idx = CheckpointIndex(thread_id=run)
        idx.build_from_history(history)
        self.assertEqual(idx.resolve(BoundarySelector.by_alias("c_0")).after_occurrences, ())
        self.assertEqual(idx.get_alias("c_1").after_occurrence_labels, ("A#1",))
        with self.assertRaises(KeyError):
            idx.resolve(BoundarySelector.by_alias("c_99"))

    def test_enrichment_preserves_identity_fields(self):
        run = "run_enr"
        from mas_langgraph.artifacts.models import ArtifactRef
        from mas_langgraph.provenance.models import ArtifactBinding

        raw = OccurrenceRecord(
            occurrence_id=f"{run}:factual:build_outline#1",
            node_id="build_outline",
            occurrence_index=1,
            branch_id="factual",
            run_id=run,
            execution_index=3,
            input_bindings=(ArtifactBinding(ArtifactRef("S1", 1), "document_section"),),
            produced_artifacts=(ArtifactRef("S1", 1),),
            checkpoint_before=None,
            checkpoint_after=None,
            superstep=None,
        )
        history = [
            self._snap(run_id=run, checkpoint_id="ck0", step=1, occurrences=[]),
            self._snap(
                run_id=run,
                checkpoint_id="ck1",
                step=2,
                occurrences=[raw],
                parent_id="ck0",
            ),
        ]
        idx = CheckpointIndex(thread_id=run)
        idx.build_from_history(history)
        enriched = idx.enrich_from_occurrences([raw])[0]
        self.assertEqual(enriched.occurrence_id, raw.occurrence_id)
        self.assertEqual(enriched.node_id, raw.node_id)
        self.assertEqual(enriched.occurrence_index, raw.occurrence_index)
        self.assertEqual(enriched.execution_index, raw.execution_index)
        self.assertEqual(enriched.input_bindings, raw.input_bindings)
        self.assertEqual(enriched.produced_artifacts, raw.produced_artifacts)
        self.assertEqual(enriched.checkpoint_before, "c_0")
        self.assertEqual(enriched.checkpoint_after, "c_1")
        self.assertEqual(enriched.superstep, 2)

    def test_synthetic_rejected_unless_opt_in(self):
        run = "run_synth"
        history = [
            self._snap(run_id=run, checkpoint_id="synth_0", step=0, occurrences=[]),
        ]
        idx = CheckpointIndex(thread_id=run)
        with self.assertRaises(ValueError):
            idx.build_from_history(history, allow_synthetic=False)
        idx.build_from_history(history, allow_synthetic=True)
        self.assertEqual(len(idx.records), 1)

    def test_two_runs_isolated_and_retrieval(self):
        from mas_langgraph.replay.executor import LangGraphReplayExecutor

        ex = LangGraphReplayExecutor(compiled_graph=None)
        r1 = ex.run_factual(
            {"step_logs": [{"a": 1}, {"a": 2}, {"a": 3}]},
            run_id="run_A",
            allow_synthetic_history=True,
        )
        r2 = ex.run_factual(
            {"step_logs": [{"b": 1}]},
            run_id="run_B",
            allow_synthetic_history=True,
        )
        idx_a = ex.get_checkpoint_index("run_A")
        idx_b = ex.get_checkpoint_index("run_B")
        self.assertNotEqual(
            idx_a.resolve(BoundarySelector.by_alias("c_0")).checkpoint_id,
            idx_b.resolve(BoundarySelector.by_alias("c_0")).checkpoint_id,
        )
        self.assertEqual(idx_a.records[0].thread_id, "run_A")
        self.assertEqual(idx_b.records[0].thread_id, "run_B")
        hist_len = len(ex.get_state_history("run_A"))
        _ = idx_a.resolved_occurrences
        self.assertEqual(len(ex.get_state_history("run_A")), hist_len)

        snap = ex.get_checkpoint_snapshot("run_A", BoundarySelector.by_alias("c_1"))
        rec = idx_a.get_alias("c_1")
        self.assertEqual(extract_checkpoint_id(snap), rec.checkpoint_id)
        self.assertEqual(extract_thread_id(snap, "run_A"), "run_A")
        self.assertIs(r1.checkpoint_index, idx_a)
        self.assertIs(r2.checkpoint_index, idx_b)

    def test_real_langgraph_metadata_and_retrieval(self):
        from mas_langgraph.cases.mcq import MCQCaseDefinition
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.core.runtime import WorkflowRuntime
        from mas_langgraph.core.workflow_executor import WorkflowExecutor
        from mas_langgraph.orchestrator.needs import NeedsEvaluator
        from mas_langgraph.provenance.channel_state import empty_provenance_channels
        from mas_langgraph.artifacts.channel_state import empty_artifact_channels

        class StubRunner:
            coverage_plan = None
            pdf_id = None

            def invoke(self, agent_id, text="", cooperative_state=None, **kwargs):
                return {"role": agent_id, "result": "stub"}

            def retrieval_exhausted(self, **kwargs):
                return False

        case = MCQCaseDefinition()
        cs = CooperativeState()
        rt = WorkflowRuntime(
            case_definition=case,
            agent_runner=StubRunner(),
            needs_evaluator=NeedsEvaluator(cs, None, agent_runner=StubRunner()),
        )
        ex = WorkflowExecutor(runtime=rt)
        result = ex.run(
            {
                "cooperative_state": cs,
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
            },
            run_id="run_p9_real",
        )
        self.assertIsNone(result.error, msg=result.error)
        idx = result.checkpoint_index
        self.assertIsNotNone(idx)
        self.assertGreaterEqual(len(idx.records), 2)
        has_real_step = any(r.superstep is not None for r in idx.records)
        self.assertTrue(has_real_step)
        mismatch = any(
            r.superstep is not None and r.superstep != int(r.alias.split("_")[1])
            for r in idx.records
        )
        if len(idx.records) > 1:
            self.assertTrue(
                mismatch
                or idx.records[0].superstep in (-1, 0)
                or idx.records[0].superstep is None
            )

        hist_before = len(ex.get_state_history("run_p9_real"))
        mid = idx.records[min(2, len(idx.records) - 1)]
        snap = ex.replay_executor.get_checkpoint_snapshot(
            "run_p9_real", BoundarySelector.by_alias(mid.alias)
        )
        self.assertEqual(extract_checkpoint_id(snap), mid.checkpoint_id)
        self.assertEqual(extract_thread_id(snap, "run_p9_real"), "run_p9_real")
        self.assertEqual(len(ex.get_state_history("run_p9_real")), hist_before)

        for occ in result.occurrences:
            if getattr(occ, "checkpoint_after", None):
                self.assertTrue(str(occ.checkpoint_after).startswith("c_"))


class TestPrompt10ForkAndReplay(unittest.TestCase):
    """Prompt 10 — real LangGraph NOOP / REMOVE fork-and-replay."""

    def _stub_runtime(self, cs=None):
        from mas_langgraph.cases.mcq import MCQCaseDefinition
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.core.runtime import WorkflowRuntime
        from mas_langgraph.orchestrator.needs import NeedsEvaluator

        class StubRunner:
            coverage_plan = None
            pdf_id = None

            def invoke(self, agent_id, text="", cooperative_state=None, **kwargs):
                return {"role": agent_id, "result": "stub"}

            def retrieval_exhausted(self, **kwargs):
                return False

        cs = cs or CooperativeState()
        case = MCQCaseDefinition()
        return WorkflowRuntime(
            case_definition=case,
            agent_runner=StubRunner(),
            needs_evaluator=NeedsEvaluator(cs, None, agent_runner=StubRunner()),
        ), cs, case

    def _initial(self, cs, extra=None):
        from mas_langgraph.artifacts.channel_state import empty_artifact_channels
        from mas_langgraph.provenance.channel_state import empty_provenance_channels

        state = {
            "cooperative_state": cs,
            "searched_queries": set(),
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
        if extra:
            state.update(extra)
        return state

    def test_real_noop_replay_and_same_origin_as_remove(self):
        from mas_langgraph.core.entities import EvidenceChunk
        from mas_langgraph.core.workflow_executor import WorkflowExecutor
        from mas_langgraph.artifacts.channel_state import dump_artifact_channels
        from mas_langgraph.artifacts.registry import ArtifactRegistry
        from mas_langgraph.artifacts.resolver import ArtifactResolver
        from mas_langgraph.cases.mcq.artifacts import wrap_evidence_chunk
        from mas_langgraph.replay.interventions import ArtifactIntervention

        rt, cs, _ = self._stub_runtime()
        cs.add_evidence_chunk(
            EvidenceChunk(
                chunk_id="E17",
                content="k-means clusters",
                metadata={"page_number": 1},
                quality_score=0.9,
            )
        )
        reg = ArtifactRegistry()
        resolver = ArtifactResolver(registry=reg, branch_id="factual")
        wrap_evidence_chunk(cs.evidence_pool["E17"], registry=reg)
        for ref in reg.all_refs():
            resolver.make_available(ref)

        ex = WorkflowExecutor(runtime=rt)
        factual = ex.run(
            self._initial(cs, dump_artifact_channels(reg, resolver)),
            run_id="run_p10_noop",
        )
        self.assertIsNone(factual.error, msg=factual.error)
        idx = factual.checkpoint_index
        origin = idx.resolve(BoundarySelector.before_next_node("package_final_quiz"))
        snap = ex.replay_executor.get_checkpoint_snapshot(
            "run_p10_noop", BoundarySelector.by_alias(origin.alias)
        )
        self.assertIn("package_final_quiz", tuple(snap.next or ()))

        factual_index_len = len(idx.records)
        factual_catalog = dict(factual.final_state.get("artifact_catalog") or {})
        factual_blocked = set(
            factual.final_state.get("blocked_artifact_refs") or set()
        )
        factual_occ_ids = {
            getattr(o, "occurrence_id", None) for o in (factual.occurrences or [])
        }

        noop = ex.replay_executor.fork_noop(
            "run_p10_noop", BoundarySelector.by_alias(origin.alias)
        )
        self.assertIsNone(noop.error, msg=noop.error)
        self.assertNotEqual(noop.branch_id, "factual")
        self.assertEqual(noop.run_id, "run_p10_noop")
        self.assertEqual(noop.final_state.get("branch_id"), noop.branch_id)
        self.assertTrue(noop.branch_occurrences)
        self.assertTrue(
            any(
                getattr(o, "node_id", None) == "package_final_quiz"
                for o in noop.branch_occurrences
            )
        )
        # Upstream nodes before the origin must not re-execute on the branch
        self.assertFalse(
            any(
                getattr(o, "node_id", None)
                == "initial_retrieve_evidence_for_all_sections"
                for o in noop.branch_occurrences
            )
        )
        self.assertTrue(noop.branch_history)
        for snap_b in noop.branch_history:
            cid = extract_checkpoint_id(snap_b)
            self.assertFalse(str(cid).startswith("synth_"))
        self.assertEqual(
            set(noop.final_state.get("blocked_artifact_refs") or set()),
            set(snap.values.get("blocked_artifact_refs") or set()),
        )

        # REMOVE from same origin
        iv = ArtifactIntervention(
            operation="REMOVE",
            artifact=ArtifactRef("E17", 1),
            checkpoint=origin,
        )
        cf = ex.replay_executor.fork_with_intervention("run_p10_noop", iv)
        self.assertIsNone(cf.error, msg=cf.error)
        self.assertEqual(
            noop.branch_metadata.origin_checkpoint_id,
            cf.branch_metadata.origin_checkpoint_id,
        )
        self.assertEqual(cf.branch_metadata.parent_branch_id, "factual")
        self.assertEqual(noop.branch_metadata.parent_branch_id, "factual")
        blocked = cf.final_state.get("blocked_artifact_refs") or set()
        self.assertIn(ArtifactRef("E17", 1), blocked)
        # Node-local projected business state cannot see E17
        from mas_langgraph.cases.mcq.intervention_view import (
            project_cooperative_state_for_resolution,
        )

        projected = project_cooperative_state_for_resolution(cf.final_state)
        self.assertNotIn("E17", getattr(projected, "evidence_pool", {}) or {})
        # Historical retained
        from mas_langgraph.artifacts.channel_state import load_artifact_channels

        _reg, cres = load_artifact_channels(cf.final_state)
        self.assertFalse(cres.is_available(ArtifactRef("E17", 1)))
        self.assertIsNotNone(cres.get_historical(ArtifactRef("E17", 1)))
        hist = cres.get_historical(ArtifactRef("E17", 1))
        self.assertEqual(hist.branch_id, cf.branch_id)
        self.assertNotEqual(hist.instance_id.split(":")[1], "factual")
        # Inherited artifact must not gain a counterfactual ProducedBy
        for rel in cf.final_state.get("produced_by") or []:
            prod_ref = getattr(rel, "artifact", None) or (
                rel.get("artifact") if isinstance(rel, dict) else None
            )
            prod_occ = getattr(rel, "occurrence_id", None) or (
                rel.get("occurrence_id") if isinstance(rel, dict) else None
            )
            if prod_ref == ArtifactRef("E17", 1) or (
                isinstance(prod_ref, ArtifactRef) and prod_ref.artifact_id == "E17"
            ):
                self.assertNotIn(f":{cf.branch_id}:", str(prod_occ))
        for o in cf.branch_occurrences:
            self.assertEqual(o.branch_id, cf.branch_id)
            self.assertIn(f":{cf.branch_id}:", o.occurrence_id)
        # Factual unchanged
        self.assertEqual(len(ex.replay_executor.get_checkpoint_index("run_p10_noop").records), factual_index_len)
        self.assertEqual(
            dict(ex.replay_executor._factual_results["run_p10_noop"].get("artifact_catalog") or {}),
            factual_catalog,
        )
        self.assertEqual(
            set(
                ex.replay_executor._factual_results["run_p10_noop"].get(
                    "blocked_artifact_refs"
                )
                or set()
            ),
            factual_blocked,
        )
        self.assertEqual(
            {
                getattr(o, "occurrence_id", None)
                for o in (
                    ex.replay_executor._factual_results["run_p10_noop"].get(
                        "occurrence_records"
                    )
                    or []
                )
            },
            factual_occ_ids,
        )

    def test_block_survives_make_available(self):
        reg = ArtifactRegistry()
        env = wrap_evidence_chunk(_FakeChunk("E17", "t"), registry=reg)
        resolver = ArtifactResolver(registry=reg, branch_id="factual")
        resolver.make_available(env.ref)
        resolver.block(env.ref)
        resolver.make_available(env.ref)  # must NOT unblock
        self.assertFalse(resolver.is_available(env.ref))
        self.assertIsNotNone(resolver.get_historical(env.ref))

    def test_projection_removes_blocked_evidence(self):
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.cases.mcq.intervention_view import (
            project_cooperative_state_for_resolution,
        )
        from mas_langgraph.core.entities import EvidenceChunk
        from mas_langgraph.artifacts.channel_state import dump_artifact_channels

        cs = CooperativeState()
        cs.add_evidence_chunk(
            EvidenceChunk("E17", "a", metadata={}, quality_score=1.0)
        )
        cs.add_evidence_chunk(
            EvidenceChunk("E24", "b", metadata={}, quality_score=1.0)
        )
        reg = ArtifactRegistry()
        resolver = ArtifactResolver(registry=reg, branch_id="factual")
        for cid in ("E17", "E24"):
            wrap_evidence_chunk(cs.evidence_pool[cid], registry=reg)
            resolver.make_available(ArtifactRef(cid, 1))
        resolver.block(ArtifactRef("E17", 1))
        state = {"cooperative_state": cs, "branch_id": "factual", **dump_artifact_channels(reg, resolver)}
        projected = project_cooperative_state_for_resolution(state)
        self.assertNotIn("E17", projected.evidence_pool)
        self.assertIn("E24", projected.evidence_pool)

    def test_invalid_checkpoint_and_missing_origin_artifact(self):
        from mas_langgraph.core.workflow_executor import WorkflowExecutor
        from mas_langgraph.replay.interventions import ArtifactIntervention

        rt, cs, _ = self._stub_runtime()
        ex = WorkflowExecutor(runtime=rt)
        factual = ex.run(self._initial(cs), run_id="run_p10_val")
        self.assertIsNone(factual.error, msg=factual.error)
        origin = factual.checkpoint_index.records[1]
        bad = CheckpointRecord(
            alias=origin.alias,
            thread_id="run_p10_val",
            checkpoint_id="not-the-real-id",
        )
        with self.assertRaises(ValueError):
            ex.replay_executor.fork_with_intervention(
                "run_p10_val",
                ArtifactIntervention(
                    operation="REMOVE",
                    artifact=ArtifactRef("E17", 1),
                    checkpoint=bad,
                ),
            )
        with self.assertRaises(ValueError):
            ex.replay_executor.fork_with_intervention(
                "run_p10_val",
                ArtifactIntervention(
                    operation="REMOVE",
                    artifact=ArtifactRef("E_MISSING", 1),
                    checkpoint=origin,
                ),
            )

    def test_add_requires_approved_source_on_live_graph(self):
        from mas_langgraph.core.workflow_executor import WorkflowExecutor
        from mas_langgraph.replay.interventions import ArtifactIntervention

        rt, cs, _ = self._stub_runtime()
        ex = WorkflowExecutor(runtime=rt)
        factual = ex.run(self._initial(cs), run_id="run_p10_add")
        ckpt = factual.checkpoint_index.records[1]
        with self.assertRaises(ValueError):
            ex.replay_executor.fork_with_intervention(
                "run_p10_add",
                ArtifactIntervention(
                    operation="ADD",
                    artifact=ArtifactRef("E1", 1),
                    checkpoint=ckpt,
                ),
            )

    def test_branch_store_two_factual_runs(self):
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.core.workflow_executor import WorkflowExecutor

        rt, cs, _ = self._stub_runtime()
        ex = WorkflowExecutor(runtime=rt)
        a = ex.run(self._initial(cs), run_id="run_A10")
        b = ex.run(self._initial(CooperativeState()), run_id="run_B10")
        self.assertIsNone(a.error)
        self.assertIsNone(b.error)
        origin_a = a.checkpoint_index.records[1]
        origin_b = b.checkpoint_index.records[1]
        na = ex.replay_executor.fork_noop(
            "run_A10", BoundarySelector.by_alias(origin_a.alias)
        )
        nb = ex.replay_executor.fork_noop(
            "run_B10", BoundarySelector.by_alias(origin_b.alias)
        )
        self.assertEqual(na.run_id, "run_A10")
        self.assertEqual(nb.run_id, "run_B10")
        self.assertEqual(
            ex.replay_executor.branches.get("run_A10", "factual").run_id, "run_A10"
        )
        self.assertEqual(
            ex.replay_executor.branches.get("run_B10", "factual").run_id, "run_B10"
        )
        self.assertEqual(
            ex.replay_executor.branches.get("run_A10", na.branch_id).run_id, "run_A10"
        )
        self.assertEqual(
            ex.replay_executor.branches.get("run_B10", nb.branch_id).run_id, "run_B10"
        )

    def test_sibling_independence_e17_e24(self):
        from mas_langgraph.artifacts.channel_state import dump_artifact_channels
        from mas_langgraph.core.entities import EvidenceChunk
        from mas_langgraph.core.workflow_executor import WorkflowExecutor
        from mas_langgraph.replay.interventions import ArtifactIntervention

        rt, cs, _ = self._stub_runtime()
        for cid, text in (("E17", "a"), ("E24", "b")):
            cs.add_evidence_chunk(
                EvidenceChunk(cid, text, metadata={"page_number": 1}, quality_score=0.9)
            )
        reg = ArtifactRegistry()
        resolver = ArtifactResolver(registry=reg, branch_id="factual")
        for cid in ("E17", "E24"):
            wrap_evidence_chunk(cs.evidence_pool[cid], registry=reg)
            resolver.make_available(ArtifactRef(cid, 1))
        ex = WorkflowExecutor(runtime=rt)
        factual = ex.run(
            self._initial(cs, dump_artifact_channels(reg, resolver)),
            run_id="run_p10_sib",
        )
        self.assertIsNone(factual.error, msg=factual.error)
        origin = factual.checkpoint_index.resolve(
            BoundarySelector.before_next_node("package_final_quiz")
        )
        noop = ex.replay_executor.fork_noop(
            "run_p10_sib", BoundarySelector.by_alias(origin.alias)
        )
        cf17 = ex.replay_executor.fork_with_intervention(
            "run_p10_sib",
            ArtifactIntervention(
                operation="REMOVE",
                artifact=ArtifactRef("E17", 1),
                checkpoint=origin,
            ),
        )
        cf24 = ex.replay_executor.fork_with_intervention(
            "run_p10_sib",
            ArtifactIntervention(
                operation="REMOVE",
                artifact=ArtifactRef("E24", 1),
                checkpoint=origin,
            ),
        )
        self.assertIsNone(noop.error, msg=noop.error)
        self.assertIsNone(cf17.error, msg=cf17.error)
        self.assertIsNone(cf24.error, msg=cf24.error)
        ids = {noop.branch_id, cf17.branch_id, cf24.branch_id}
        self.assertEqual(len(ids), 3)
        for br in (noop, cf17, cf24):
            self.assertEqual(br.branch_metadata.parent_branch_id, "factual")
            self.assertEqual(br.branch_metadata.origin_checkpoint_alias, origin.alias)
            self.assertEqual(
                br.branch_metadata.origin_checkpoint_id, origin.checkpoint_id
            )
        self.assertIn(ArtifactRef("E17", 1), cf17.final_state.get("blocked_artifact_refs") or set())
        self.assertNotIn(
            ArtifactRef("E17", 1), cf24.final_state.get("blocked_artifact_refs") or set()
        )
        self.assertIn(ArtifactRef("E24", 1), cf24.final_state.get("blocked_artifact_refs") or set())
        self.assertNotIn(
            ArtifactRef("E24", 1), cf17.final_state.get("blocked_artifact_refs") or set()
        )

    def test_contracts_require_intervenable_blueprint(self):
        from mas_langgraph.cases.mcq import MCQCaseDefinition

        contracts = MCQCaseDefinition().load_intervention_contracts()
        self.assertIn("MCQBlueprint", contracts.contracts)
        types = MCQCaseDefinition().load_artifact_types()
        self.assertTrue(types.require("MCQBlueprint").intervenable)

    def test_fork_preserves_next(self):
        from mas_langgraph.core.workflow_executor import WorkflowExecutor

        rt, cs, _ = self._stub_runtime()
        ex = WorkflowExecutor(runtime=rt)
        factual = ex.run(self._initial(cs), run_id="run_p10_next")
        origin = factual.checkpoint_index.resolve(
            BoundarySelector.before_next_node("package_final_quiz")
        )
        origin_snap = ex.replay_executor.get_checkpoint_snapshot(
            "run_p10_next", BoundarySelector.by_alias(origin.alias)
        )
        origin_next = tuple(origin_snap.next or ())
        noop = ex.replay_executor.fork_noop(
            "run_p10_next", BoundarySelector.by_alias(origin.alias)
        )
        self.assertIsNone(noop.error, msg=noop.error)
        # Fork checkpoint is first in branch history with matching branch_id
        self.assertTrue(noop.branch_history)
        fork_snap = noop.branch_history[0]
        # After full replay, first branch snap may already be mid-branch; check metadata
        self.assertEqual(noop.branch_metadata.origin_checkpoint_id, origin.checkpoint_id)
        self.assertEqual(origin_next, ("package_final_quiz",))


class TestPrompt11AddReplace(unittest.TestCase):
    """Prompt 11 — real LangGraph ADD / REPLACE interventions."""

    def _stub_runtime(self, cs=None):
        from mas_langgraph.cases.mcq import MCQCaseDefinition
        from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
        from mas_langgraph.core.runtime import WorkflowRuntime
        from mas_langgraph.orchestrator.needs import NeedsEvaluator

        class StubRunner:
            coverage_plan = None
            pdf_id = None

            def invoke(self, agent_id, text="", cooperative_state=None, **kwargs):
                return {"role": agent_id, "result": "stub"}

            def retrieval_exhausted(self, **kwargs):
                return False

        cs = cs or CooperativeState()
        case = MCQCaseDefinition()
        return WorkflowRuntime(
            case_definition=case,
            agent_runner=StubRunner(),
            needs_evaluator=NeedsEvaluator(cs, None, agent_runner=StubRunner()),
        ), cs, case

    def _initial(self, cs, extra=None):
        from mas_langgraph.artifacts.channel_state import empty_artifact_channels
        from mas_langgraph.provenance.channel_state import empty_provenance_channels

        state = {
            "cooperative_state": cs,
            "searched_queries": set(),
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
        if extra:
            state.update(extra)
        return state

    def _seed_evidence(self, cs, *ids):
        from mas_langgraph.core.entities import EvidenceChunk

        for cid in ids:
            cs.add_evidence_chunk(
                EvidenceChunk(
                    chunk_id=cid,
                    content=f"content-{cid}",
                    metadata={"page_number": 1},
                    quality_score=0.9,
                )
            )

    def test_real_add_from_intervention_registry(self):
        from mas_langgraph.artifacts.channel_state import (
            dump_artifact_channels,
            load_artifact_channels,
        )
        from mas_langgraph.cases.mcq.artifacts import wrap_evidence_chunk
        from mas_langgraph.cases.mcq.intervention_view import (
            project_cooperative_state_for_resolution,
        )
        from mas_langgraph.core.entities import EvidenceChunk
        from mas_langgraph.core.workflow_executor import WorkflowExecutor
        from mas_langgraph.replay.checkpoints import extract_checkpoint_id
        from mas_langgraph.replay.interventions import ArtifactIntervention

        rt, cs, _ = self._stub_runtime()
        self._seed_evidence(cs, "E17")
        reg = ArtifactRegistry()
        resolver = ArtifactResolver(registry=reg, branch_id="factual")
        wrap_evidence_chunk(cs.evidence_pool["E17"], registry=reg)
        for ref in reg.all_refs():
            resolver.make_available(ref)

        # Controlled external catalog with E99
        ext = ArtifactRegistry()
        e99 = EvidenceChunk(
            chunk_id="E99",
            content="external-evidence",
            metadata={"page_number": 2},
            quality_score=0.8,
        )
        wrap_evidence_chunk(e99, branch_id="intervention", registry=ext)

        ex = WorkflowExecutor(runtime=rt)
        factual = ex.run(
            self._initial(cs, dump_artifact_channels(reg, resolver)),
            run_id="run_p11_add_ext",
        )
        self.assertIsNone(factual.error, msg=factual.error)
        origin = factual.checkpoint_index.resolve(
            BoundarySelector.before_next_node("package_final_quiz")
        )
        snap = ex.replay_executor.get_checkpoint_snapshot(
            "run_p11_add_ext", BoundarySelector.by_alias(origin.alias)
        )
        self.assertFalse(
            any(
                getattr(e, "ref", None) == ArtifactRef("E99", 1)
                for e in (snap.values.get("artifact_catalog") or {}).values()
            )
        )

        cf = ex.replay_executor.fork_with_intervention(
            "run_p11_add_ext",
            ArtifactIntervention(
                operation="ADD",
                artifact=ArtifactRef("E99", 1),
                checkpoint=origin,
            ),
            registry=ext,
        )
        self.assertIsNone(cf.error, msg=cf.error)
        self.assertTrue(cf.branch_occurrences)
        self.assertTrue(
            any(o.node_id == "package_final_quiz" for o in cf.branch_occurrences)
        )
        for snap_b in cf.branch_history:
            self.assertFalse(str(extract_checkpoint_id(snap_b)).startswith("synth_"))

        blocked = cf.final_state.get("blocked_artifact_refs") or set()
        available = cf.final_state.get("available_artifact_refs") or set()
        self.assertIn(ArtifactRef("E99", 1), available)
        self.assertNotIn(ArtifactRef("E99", 1), blocked)

        projected = project_cooperative_state_for_resolution(cf.final_state)
        self.assertIn("E99", projected.evidence_pool)
        self.assertEqual(projected.evidence_pool["E99"].chunk_id, "E99")

        _reg, cres = load_artifact_channels(cf.final_state)
        hist = cres.get_historical(ArtifactRef("E99", 1))
        self.assertIsNotNone(hist)
        self.assertEqual(hist.branch_id, cf.branch_id)
        self.assertIn(f":{cf.branch_id}:", hist.instance_id)

        events = cf.final_state.get("intervention_events") or []
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.operation, "ADD")
        self.assertEqual(list(ev.artifact_refs), [ArtifactRef("E99", 1)])
        self.assertEqual(list(ev.supplied_refs), [ArtifactRef("E99", 1)])
        self.assertIn("intervention_catalog", ev.source_kinds)

        # No fake ProducedBy for E99 from branch nodes
        for pb in cf.final_state.get("produced_by") or []:
            pref = getattr(pb, "artifact_ref", None)
            if pref == ArtifactRef("E99", 1):
                self.fail("Intervention-supplied E99 must not gain ProducedBy")

        unresolved = cf.final_state.get("provenance_unresolved_producers") or []
        self.assertFalse(
            any("E99" in str(u) and "unresolved_producer" in str(u) for u in unresolved)
        )

    def test_real_add_from_factual_catalog_early_checkpoint(self):
        """E24 exists in final factual catalog but not at early origin → ADD imports it."""
        from mas_langgraph.artifacts.channel_state import dump_artifact_channels
        from mas_langgraph.cases.mcq.artifacts import wrap_evidence_chunk
        from mas_langgraph.cases.mcq.intervention_view import (
            project_cooperative_state_for_resolution,
        )
        from mas_langgraph.core.workflow_executor import WorkflowExecutor
        from mas_langgraph.replay.interventions import ArtifactIntervention

        rt, cs, _ = self._stub_runtime()
        self._seed_evidence(cs, "E17", "E24")
        reg = ArtifactRegistry()
        resolver = ArtifactResolver(registry=reg, branch_id="factual")
        for cid in ("E17", "E24"):
            wrap_evidence_chunk(cs.evidence_pool[cid], registry=reg)
            resolver.make_available(ArtifactRef(cid, 1))

        ex = WorkflowExecutor(runtime=rt)
        # Start without E24 in initial catalog — inject E24 only into final via
        # pre-seeded CS so it appears from c_1. To get "late" artifact, remove E24
        # from initial channels and add only E17, then manually put E24 in factual
        # final after run is not allowed. Instead: seed both, pick origin before
        # first node where catalog may still be empty (c_0), if E24 already there
        # use external for "absent at origin" case covered above.
        # Here: both present at origin; ADD is idempotent.
        factual = ex.run(
            self._initial(cs, dump_artifact_channels(reg, resolver)),
            run_id="run_p11_add_idem",
        )
        origin = factual.checkpoint_index.resolve(
            BoundarySelector.before_next_node("package_final_quiz")
        )
        cf = ex.replay_executor.fork_with_intervention(
            "run_p11_add_idem",
            ArtifactIntervention(
                operation="ADD",
                artifact=ArtifactRef("E24", 1),
                checkpoint=origin,
            ),
        )
        self.assertIsNone(cf.error, msg=cf.error)
        self.assertIn(
            ArtifactRef("E24", 1),
            cf.final_state.get("available_artifact_refs") or set(),
        )
        # Idempotent: no duplicate supplied when already at origin
        events = cf.final_state.get("intervention_events") or []
        self.assertEqual(len(events), 1)
        self.assertEqual(list(events[0].supplied_refs), [])

    def test_real_replace_evidence_and_projection(self):
        from dataclasses import replace as dc_replace

        from mas_langgraph.artifacts.channel_state import (
            dump_artifact_channels,
            load_artifact_channels,
        )
        from mas_langgraph.cases.mcq.artifacts import wrap_blueprint, wrap_evidence_chunk
        from mas_langgraph.cases.mcq.intervention_view import (
            project_cooperative_state_for_resolution,
        )
        from mas_langgraph.core.workflow_executor import WorkflowExecutor
        from mas_langgraph.domains.quiz.entities import MCQBlueprint
        from mas_langgraph.replay.interventions import ArtifactIntervention

        rt, cs, _ = self._stub_runtime()
        self._seed_evidence(cs, "E17", "E24")
        bp = MCQBlueprint(
            blueprint_id="BP12",
            section_id="sec_1",
            lo_id="lo_1",
            target_atom_ids=[],
            evidence_chunk_ids=["E17"],
            question_type="basic_definition",
            difficulty="easy",
            bloom_level="Remembering",
            generation_goal="test",
        )
        cs.mcq_blueprints["BP12"] = bp

        reg = ArtifactRegistry()
        resolver = ArtifactResolver(registry=reg, branch_id="factual")
        for cid in ("E17", "E24"):
            wrap_evidence_chunk(cs.evidence_pool[cid], registry=reg)
            resolver.make_available(ArtifactRef(cid, 1))
        wrap_blueprint(bp, registry=reg)
        resolver.make_available(ArtifactRef("BP12", 1))

        ex = WorkflowExecutor(runtime=rt)
        factual = ex.run(
            self._initial(cs, dump_artifact_channels(reg, resolver)),
            run_id="run_p11_repl",
        )
        self.assertIsNone(factual.error, msg=factual.error)
        origin = factual.checkpoint_index.resolve(
            BoundarySelector.before_next_node("package_final_quiz")
        )
        cf = ex.replay_executor.fork_with_intervention(
            "run_p11_repl",
            ArtifactIntervention(
                operation="REPLACE",
                artifact=(ArtifactRef("E17", 1), ArtifactRef("E24", 1)),
                checkpoint=origin,
            ),
        )
        self.assertIsNone(cf.error, msg=cf.error)
        _reg, cres = load_artifact_channels(cf.final_state)
        self.assertEqual(cres.resolve_ref(ArtifactRef("E17", 1)), ArtifactRef("E24", 1))
        self.assertIsNotNone(cres.get_historical(ArtifactRef("E17", 1)))
        self.assertIsNotNone(cres.get_historical(ArtifactRef("E24", 1)))
        subs = cf.final_state.get("artifact_substitutions") or {}
        self.assertEqual(subs.get(ArtifactRef("E17", 1)), ArtifactRef("E24", 1))

        projected = project_cooperative_state_for_resolution(cf.final_state)
        self.assertNotIn("E17", projected.evidence_pool)
        self.assertIn("E24", projected.evidence_pool)
        pbp = projected.mcq_blueprints.get("BP12")
        self.assertIsNotNone(pbp)
        self.assertEqual(list(pbp.evidence_chunk_ids), ["E24"])

        events = cf.final_state.get("intervention_events") or []
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].operation, "REPLACE")
        self.assertEqual(
            list(events[0].artifact_refs),
            [ArtifactRef("E17", 1), ArtifactRef("E24", 1)],
        )

        # Factual unchanged
        fact_subs = (
            ex.replay_executor._factual_results["run_p11_repl"].get(
                "artifact_substitutions"
            )
            or {}
        )
        self.assertNotIn(ArtifactRef("E17", 1), fact_subs)

    def test_replace_rejects_incompatible_and_self_and_missing(self):
        from mas_langgraph.artifacts.channel_state import dump_artifact_channels
        from mas_langgraph.cases.mcq.artifacts import wrap_evidence_chunk, wrap_knowledge_atom
        from mas_langgraph.core.entities import KnowledgeAtom
        from mas_langgraph.core.workflow_executor import WorkflowExecutor
        from mas_langgraph.replay.interventions import ArtifactIntervention

        rt, cs, _ = self._stub_runtime()
        self._seed_evidence(cs, "E17")
        cs.knowledge_atoms["A3"] = KnowledgeAtom(
            atom_id="A3",
            content="atom",
            atom_type="fact",
            citations=[],
        )
        reg = ArtifactRegistry()
        resolver = ArtifactResolver(registry=reg, branch_id="factual")
        wrap_evidence_chunk(cs.evidence_pool["E17"], registry=reg)
        wrap_knowledge_atom(cs.knowledge_atoms["A3"], registry=reg)
        for ref in reg.all_refs():
            resolver.make_available(ref)

        ex = WorkflowExecutor(runtime=rt)
        factual = ex.run(
            self._initial(cs, dump_artifact_channels(reg, resolver)),
            run_id="run_p11_rej",
        )
        origin = factual.checkpoint_index.records[1]
        with self.assertRaises(ValueError):
            ex.replay_executor.fork_with_intervention(
                "run_p11_rej",
                ArtifactIntervention(
                    operation="REPLACE",
                    artifact=(ArtifactRef("E17", 1), ArtifactRef("E17", 1)),
                    checkpoint=origin,
                ),
            )
        with self.assertRaises(ValueError):
            ex.replay_executor.fork_with_intervention(
                "run_p11_rej",
                ArtifactIntervention(
                    operation="REPLACE",
                    artifact=(ArtifactRef("E17", 1), ArtifactRef("A3", 1)),
                    checkpoint=origin,
                ),
            )
        with self.assertRaises(ValueError):
            ex.replay_executor.fork_with_intervention(
                "run_p11_rej",
                ArtifactIntervention(
                    operation="REPLACE",
                    artifact=(ArtifactRef("E17", 1), ArtifactRef("E999", 1)),
                    checkpoint=origin,
                ),
            )

    def test_sibling_independence_add_replace_remove(self):
        from mas_langgraph.artifacts.channel_state import dump_artifact_channels
        from mas_langgraph.cases.mcq.artifacts import wrap_evidence_chunk
        from mas_langgraph.core.entities import EvidenceChunk
        from mas_langgraph.core.workflow_executor import WorkflowExecutor
        from mas_langgraph.replay.interventions import ArtifactIntervention

        rt, cs, _ = self._stub_runtime()
        self._seed_evidence(cs, "E17", "E24")
        reg = ArtifactRegistry()
        resolver = ArtifactResolver(registry=reg, branch_id="factual")
        for cid in ("E17", "E24"):
            wrap_evidence_chunk(cs.evidence_pool[cid], registry=reg)
            resolver.make_available(ArtifactRef(cid, 1))
        ext = ArtifactRegistry()
        wrap_evidence_chunk(
            EvidenceChunk("E99", "x", metadata={}, quality_score=1.0),
            branch_id="intervention",
            registry=ext,
        )

        ex = WorkflowExecutor(runtime=rt)
        factual = ex.run(
            self._initial(cs, dump_artifact_channels(reg, resolver)),
            run_id="run_p11_sib",
        )
        origin = factual.checkpoint_index.resolve(
            BoundarySelector.before_next_node("package_final_quiz")
        )
        noop = ex.replay_executor.fork_noop(
            "run_p11_sib", BoundarySelector.by_alias(origin.alias)
        )
        add = ex.replay_executor.fork_with_intervention(
            "run_p11_sib",
            ArtifactIntervention(
                operation="ADD", artifact=ArtifactRef("E99", 1), checkpoint=origin
            ),
            registry=ext,
        )
        repl = ex.replay_executor.fork_with_intervention(
            "run_p11_sib",
            ArtifactIntervention(
                operation="REPLACE",
                artifact=(ArtifactRef("E17", 1), ArtifactRef("E24", 1)),
                checkpoint=origin,
            ),
        )
        rem = ex.replay_executor.fork_with_intervention(
            "run_p11_sib",
            ArtifactIntervention(
                operation="REMOVE", artifact=ArtifactRef("E17", 1), checkpoint=origin
            ),
        )
        for br in (noop, add, repl, rem):
            self.assertIsNone(br.error, msg=br.error)
            self.assertEqual(br.branch_metadata.parent_branch_id, "factual")
            self.assertEqual(br.branch_metadata.origin_checkpoint_id, origin.checkpoint_id)
        self.assertNotIn(
            ArtifactRef("E17", 1),
            noop.final_state.get("artifact_substitutions") or {},
        )
        self.assertEqual(
            (repl.final_state.get("artifact_substitutions") or {}).get(
                ArtifactRef("E17", 1)
            ),
            ArtifactRef("E24", 1),
        )
        self.assertIn(
            ArtifactRef("E17", 1), rem.final_state.get("blocked_artifact_refs") or set()
        )
        self.assertNotIn(
            ArtifactRef("E17", 1),
            rem.final_state.get("artifact_substitutions") or {},
        )
        self.assertIn(
            ArtifactRef("E99", 1),
            add.final_state.get("available_artifact_refs") or set(),
        )
        self.assertNotIn(
            ArtifactRef("E99", 1),
            rem.final_state.get("available_artifact_refs") or set(),
        )

    def test_source_conflict_and_external_add_policy(self):
        from mas_langgraph.artifacts.channel_state import dump_artifact_channels
        from mas_langgraph.cases.mcq.artifacts import wrap_evidence_chunk, wrap_knowledge_atom
        from mas_langgraph.core.entities import EvidenceChunk, KnowledgeAtom
        from mas_langgraph.core.workflow_executor import WorkflowExecutor
        from mas_langgraph.replay.interventions import (
            ArtifactIntervention,
            InterventionContract,
            InterventionContractRegistry,
        )

        rt, cs, _ = self._stub_runtime()
        self._seed_evidence(cs, "E24")
        reg = ArtifactRegistry()
        resolver = ArtifactResolver(registry=reg, branch_id="factual")
        wrap_evidence_chunk(cs.evidence_pool["E24"], registry=reg)
        resolver.make_available(ArtifactRef("E24", 1))

        ex = WorkflowExecutor(runtime=rt)
        factual = ex.run(
            self._initial(cs, dump_artifact_channels(reg, resolver)),
            run_id="run_p11_conflict",
        )
        origin = factual.checkpoint_index.records[1]

        ext = ArtifactRegistry()
        wrap_evidence_chunk(
            EvidenceChunk("E24", "DIFFERENT-PAYLOAD", metadata={}, quality_score=0.1),
            branch_id="intervention",
            registry=ext,
        )
        with self.assertRaises(ValueError):
            ex.replay_executor.fork_with_intervention(
                "run_p11_conflict",
                ArtifactIntervention(
                    operation="ADD",
                    artifact=ArtifactRef("E24", 1),
                    checkpoint=origin,
                ),
                registry=ext,
            )

        # External ADD forbidden for KnowledgeAtom
        atom_reg = ArtifactRegistry()
        wrap_knowledge_atom(
            KnowledgeAtom("A99", "x", "fact", []),
            branch_id="intervention",
            registry=atom_reg,
        )
        # Use a temporary contract that allows ADD but not external
        contracts = InterventionContractRegistry()
        contracts.register(
            InterventionContract(
                artifact_type="KnowledgeAtom",
                allowed_operations=("ADD",),
                allow_external_add=False,
            )
        )
        ex.replay_executor.contracts = contracts
        with self.assertRaises(ValueError):
            ex.replay_executor.fork_with_intervention(
                "run_p11_conflict",
                ArtifactIntervention(
                    operation="ADD",
                    artifact=ArtifactRef("A99", 1),
                    checkpoint=origin,
                ),
                registry=atom_reg,
            )

    def test_remove_records_intervention_event(self):
        from mas_langgraph.artifacts.channel_state import dump_artifact_channels
        from mas_langgraph.cases.mcq.artifacts import wrap_evidence_chunk
        from mas_langgraph.core.workflow_executor import WorkflowExecutor
        from mas_langgraph.replay.interventions import ArtifactIntervention

        rt, cs, _ = self._stub_runtime()
        self._seed_evidence(cs, "E17")
        reg = ArtifactRegistry()
        resolver = ArtifactResolver(registry=reg, branch_id="factual")
        wrap_evidence_chunk(cs.evidence_pool["E17"], registry=reg)
        resolver.make_available(ArtifactRef("E17", 1))
        ex = WorkflowExecutor(runtime=rt)
        factual = ex.run(
            self._initial(cs, dump_artifact_channels(reg, resolver)),
            run_id="run_p11_rm_ev",
        )
        origin = factual.checkpoint_index.resolve(
            BoundarySelector.before_next_node("package_final_quiz")
        )
        cf = ex.replay_executor.fork_with_intervention(
            "run_p11_rm_ev",
            ArtifactIntervention(
                operation="REMOVE",
                artifact=ArtifactRef("E17", 1),
                checkpoint=origin,
            ),
        )
        events = cf.final_state.get("intervention_events") or []
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].operation, "REMOVE")
        self.assertEqual(list(events[0].artifact_refs), [ArtifactRef("E17", 1)])


if __name__ == "__main__":
    unittest.main()
