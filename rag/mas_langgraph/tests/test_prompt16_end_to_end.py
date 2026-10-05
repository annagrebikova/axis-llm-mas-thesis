"""Prompt 16 — public AXIS service, export, and end-to-end integration."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Tuple

from mas_langgraph.artifacts.channel_state import (
    dump_artifact_channels,
    empty_artifact_channels,
    load_artifact_channels,
)
from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.artifacts.registry import ArtifactRegistry
from mas_langgraph.axis.export import (
    AXIS_EXPLANATION_SCHEMA_VERSION,
    axis_report_to_dict,
    dump_axis_explanation,
    explanation_result_to_report,
)
from mas_langgraph.axis.service import AxisExplanationService
from mas_langgraph.cases.mcq import MCQCaseDefinition
from mas_langgraph.cases.mcq.artifact_registration import register_mcq_artifacts_for_stage
from mas_langgraph.cases.mcq.cooperative_state import CooperativeState
from mas_langgraph.cases.mcq.explained_case import MCQExplainedCaseResolver
from mas_langgraph.cases.mcq.verbalizer import McqVerbalizer
from mas_langgraph.core.checkpointer import default_factual_checkpointer
from mas_langgraph.core.entities import (
    DocSection,
    EvidenceChunk,
    KnowledgeAtom,
    LearningObjective,
)
from mas_langgraph.core.graph_builder import build_mcq_workflow_graph
from mas_langgraph.core.runtime import WorkflowRuntime
from mas_langgraph.core.workflow_executor import WorkflowExecutor
from mas_langgraph.orchestrator.needs import NeedsEvaluator
from mas_langgraph.provenance.channel_state import (
    dump_provenance_channels,
    empty_provenance_channels,
    load_provenance_recorder,
)
from mas_langgraph.provenance.models import OccurrenceRecord
from mas_langgraph.replay.branch_artifacts import artifact_exists_at_checkpoint
from mas_langgraph.replay.checkpoints import (
    BoundarySelector,
    CheckpointIndex,
    CheckpointRecord,
)
from mas_langgraph.replay.executor import LangGraphReplayExecutor, WorkflowRunResult


RAG_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = RAG_DIR / "docs"


def _stub_runner():
    class StubRunner:
        coverage_plan = None
        pdf_id = None

        def invoke(self, agent_id, text="", cooperative_state=None, **kwargs):
            return {"role": agent_id, "result": "stub"}

        def retrieval_exhausted(self, **kwargs):
            return False

    return StubRunner()


def _seeded_cooperative_state() -> CooperativeState:
    cs = CooperativeState()
    cs.add_section(
        DocSection(
            "sec_kmeans",
            "k-Means",
            (6, 10),
            0.9,
            content_summary="k-Means clustering",
        )
    )
    cs.add_learning_objective(
        LearningObjective(
            "lo_kmeans", "sec_kmeans", "Explain k-means.", "intermediate"
        )
    )
    for cid, text in [
        (
            "E17",
            "k-means assigns points to nearest centroids then updates them.",
        ),
        (
            "E18",
            "Objective minimizes within-cluster sum of squares.",
        ),
    ]:
        cs.add_evidence_chunk(
            EvidenceChunk(
                cid,
                text,
                {"page_number": 8, "section_id": "sec_kmeans"},
                quality_score=0.9,
            )
        )
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
    cs.update_section_evidence_status()
    return cs


def _build_seeded_factual_run(
    run_id: str = "run_p16_e2e",
    *,
    cooperative_state: Any = None,
) -> Tuple[MCQCaseDefinition, WorkflowExecutor, WorkflowRunResult]:
    """Deterministic seeded LangGraph factual fixture for AXIS E2E.

    Pre-seeds outline/evidence/atoms, then resumes the real 22-node MCQ graph
    after ``update_section_evidence_status`` so blueprint → generate → verify →
    Cross-CoVe → acceptance → package execute with real MemorySaver checkpoints.
    External RAG/LLM services are stubbed; graph nodes and instrumentation are real.
    """
    case = MCQCaseDefinition()
    cs = cooperative_state if cooperative_state is not None else _seeded_cooperative_state()
    cp = default_factual_checkpointer()
    graph = build_mcq_workflow_graph(checkpointer=cp)
    runner = _stub_runner()
    rt = WorkflowRuntime(
        agent_runner=runner,
        needs_evaluator=NeedsEvaluator(cs, None, agent_runner=runner),
        case_definition=case,
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
            occ,
            produced_artifacts=(ref,),
            producer_artifact_types={ref: atype},
        )
    state.update(dump_provenance_channels(rec))
    result = ex.run(
        state,
        runtime=rt,
        run_id=run_id,
        resume_after_node="update_section_evidence_status",
    )
    return case, ex, result


class TestDocsConsistency(unittest.TestCase):
    def test_checkpoint_replay_docs_not_stale(self):
        text = (DOCS_DIR / "checkpoint_replay.md").read_text(encoding="utf-8")
        self.assertNotIn("ADD / REPLACE are not real", text)

    def test_architecture_docs_not_mid_migration(self):
        text = (DOCS_DIR / "architecture.md").read_text(encoding="utf-8")
        self.assertNotIn("Incremental migration state (Prompt 2–7)", text)
        self.assertNotIn("This repository is mid-migration", text)


class TestBoundaryOccurrenceIndex(unittest.TestCase):
    def test_repeated_verifier_uses_occurrence_index(self):
        idx = CheckpointIndex(thread_id="t")
        idx.add(
            CheckpointRecord(
                alias="c_0",
                thread_id="t",
                checkpoint_id="id0",
            )
        )
        idx.add(
            CheckpointRecord(
                alias="c_1",
                thread_id="t",
                checkpoint_id="id1",
                after_occurrences=("run:t:verify_candidate_mcqs_against_evidence#1",),
                after_occurrence_labels=("verify_candidate_mcqs_against_evidence#1",),
                parent_checkpoint_id="id0",
            )
        )
        idx.add(
            CheckpointRecord(
                alias="c_2",
                thread_id="t",
                checkpoint_id="id2",
                after_occurrences=("run:t:verify_candidate_mcqs_against_evidence#2",),
                after_occurrence_labels=("verify_candidate_mcqs_against_evidence#2",),
                parent_checkpoint_id="id1",
            )
        )
        from mas_langgraph.axis.slicing import CaseSlice
        from mas_langgraph.axis.models import ExplainedCase

        occ1 = OccurrenceRecord(
            occurrence_id="run:t:verify_candidate_mcqs_against_evidence#1",
            node_id="verify_candidate_mcqs_against_evidence",
            occurrence_index=1,
            branch_id="factual",
            run_id="run",
            execution_index=10,
        )
        occ2 = OccurrenceRecord(
            occurrence_id="run:t:verify_candidate_mcqs_against_evidence#2",
            node_id="verify_candidate_mcqs_against_evidence",
            occurrence_index=2,
            branch_id="factual",
            run_id="run",
            execution_index=20,
        )
        case_slice = CaseSlice(
            explained_case=ExplainedCase(
                run_id="run",
                artifact_ref=ArtifactRef("Q", 1),
                branch_id="factual",
                decision_ref=ArtifactRef("decision:Q", 1),
                outcome="accept",
            ),
            relevant_occurrences=(occ1, occ2),
        )
        text = McqVerbalizer().verbalize_boundaries(case_slice, idx, strict=True)
        self.assertIn("before verify_candidate_mcqs_against_evidence#1", text)
        self.assertIn("before verify_candidate_mcqs_against_evidence#2", text)
        b1 = idx.resolve(
            BoundarySelector.before_occurrence(
                "verify_candidate_mcqs_against_evidence", 1
            )
        )
        b2 = idx.resolve(
            BoundarySelector.before_occurrence(
                "verify_candidate_mcqs_against_evidence", 2
            )
        )
        self.assertNotEqual(b1.checkpoint_id, b2.checkpoint_id)


class TestExplainedCaseResolver(unittest.TestCase):
    def test_list_and_historical_versions(self):
        from mas_langgraph.artifacts.models import ArtifactEnvelope
        from mas_langgraph.core.outcomes import OutcomeAssessment
        from mas_langgraph.domains.quiz.entities import SupervisorDecision

        reg = ArtifactRegistry()

        class _Cand:
            q_id = "Q12"
            revision_count = 0

        for ver in (1, 2):
            aref = ArtifactRef("assessment:Q12", ver)
            dref = ArtifactRef("decision:Q12", ver)
            cref = ArtifactRef("Q12", ver)
            reg.store(
                ArtifactEnvelope.create(
                    logical_id="Q12",
                    version=ver,
                    artifact_type="CandidateMCQ",
                    payload=_Cand(),
                    branch_id="factual",
                    created_in_run="r",
                )
            )
            assessment = OutcomeAssessment(
                assessment_ref=aref,
                target_ref=cref,
                criteria=(),
            )
            decision = SupervisorDecision(
                q_id="Q12",
                decision="accept" if ver == 2 else "revise",
                reasons=["ok"],
                candidate_version=ver,
                assessment_ref=aref,
            )
            reg.store(
                ArtifactEnvelope.create(
                    logical_id="assessment:Q12",
                    version=ver,
                    artifact_type="OutcomeAssessment",
                    payload=assessment,
                    branch_id="factual",
                    created_in_run="r",
                )
            )
            reg.store(
                ArtifactEnvelope.create(
                    logical_id="decision:Q12",
                    version=ver,
                    artifact_type="SupervisorDecision",
                    payload=decision,
                    branch_id="factual",
                    created_in_run="r",
                )
            )
        # Incomplete candidate (no decision)
        reg.store(
            ArtifactEnvelope.create(
                logical_id="Q999",
                version=1,
                artifact_type="CandidateMCQ",
                payload=_Cand(),
                branch_id="factual",
                created_in_run="r",
            )
        )
        fr = WorkflowRunResult(
            run_id="r",
            branch_id="factual",
            final_state={"artifact_catalog": dict(reg.catalog)},
        )
        resolver = MCQExplainedCaseResolver()
        refs = resolver.list_explainable_artifacts(factual_run_result=fr)
        self.assertEqual([str(r) for r in refs], ["Q12:v1", "Q12:v2"])
        c1 = resolver.resolve(factual_run_result=fr, artifact_ref=ArtifactRef("Q12", 1))
        c2 = resolver.resolve(factual_run_result=fr, artifact_ref=ArtifactRef("Q12", 2))
        self.assertEqual(c1.artifact_ref.version, 1)
        self.assertEqual(c2.artifact_ref.version, 2)
        self.assertNotEqual(c1.outcome, c2.outcome)
        with self.assertRaises(ValueError):
            resolver.resolve(factual_run_result=fr, artifact_ref=ArtifactRef("Q999", 1))
        with self.assertRaises(ValueError):
            resolver.resolve(factual_run_result=fr, artifact_ref=ArtifactRef("Q999", 9))


class TestPublicServiceValidation(unittest.TestCase):
    def test_wrong_branch_rejected(self):
        case = MCQCaseDefinition()
        fr = WorkflowRunResult(run_id="r", branch_id="cf_x", final_state={})
        ex = LangGraphReplayExecutor(compiled_graph=None)
        ex.run_factual({"step_logs": [{}]}, run_id="r", allow_synthetic_history=True)
        with self.assertRaises(ValueError):
            AxisExplanationService(
                case_definition=case,
                factual_run_result=fr,
                replay_executor=ex,
            )

    def test_run_index_mismatch_rejected(self):
        case = MCQCaseDefinition()
        fr = WorkflowRunResult(
            run_id="r1",
            branch_id="factual",
            final_state={},
            checkpoint_index=CheckpointIndex(thread_id="other"),
        )
        ex = LangGraphReplayExecutor(compiled_graph=None)
        ex.run_factual({"step_logs": [{}]}, run_id="r1", allow_synthetic_history=True)
        with self.assertRaises(ValueError):
            AxisExplanationService(
                case_definition=case,
                factual_run_result=fr,
                replay_executor=ex,
                checkpoint_index=fr.checkpoint_index,
            )


class TestExport(unittest.TestCase):
    def test_deterministic_report_and_json(self):
        from mas_langgraph.axis.explainer import ExplanationResult
        from mas_langgraph.axis.models import ExplainedCase
        from mas_langgraph.axis.slicing import CaseSlice

        explained = ExplainedCase(
            run_id="r",
            artifact_ref=ArtifactRef("Q12", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("decision:Q12", 1),
            outcome="accept",
            assessment_ref=ArtifactRef("assessment:Q12", 1),
        )
        result = ExplanationResult(
            explained_case=explained,
            case_slice=CaseSlice(
                explained_case=explained,
                relevant_artifact_refs=(ArtifactRef("Q12", 1), ArtifactRef("E17", 1)),
            ),
            final_explanation="Decision accept for Q12.",
            termination_reason="done",
            rounds=1,
        )
        case = MCQCaseDefinition()
        r1 = explanation_result_to_report(result, question="Why?", case_definition=case)
        r2 = explanation_result_to_report(result, question="Why?", case_definition=case)
        d1 = axis_report_to_dict(r1)
        d2 = axis_report_to_dict(r2)
        self.assertEqual(d1, d2)
        self.assertEqual(d1["schema_version"], AXIS_EXPLANATION_SCHEMA_VERSION)
        self.assertNotIn("WorkflowRunResult", json.dumps(d1))
        with tempfile.TemporaryDirectory() as td:
            path = dump_axis_explanation(
                result, str(Path(td) / "axis.json"), question="Why?", case_definition=case
            )
            loaded = json.loads(Path(path).read_text(encoding="utf-8"))
            for key in (
                "schema_version",
                "case_id",
                "run_id",
                "artifact_ref",
                "factual_decision",
                "final_explanation",
                "query_rounds",
                "replay_comparisons",
            ):
                self.assertIn(key, loaded)


class TestImportIsolation(unittest.TestCase):
    def test_generic_imports_clean(self):
        code = (
            "import mas_langgraph; import mas_langgraph.core; "
            "import mas_langgraph.artifacts; import mas_langgraph.provenance; "
            "import mas_langgraph.replay; import mas_langgraph.axis"
        )
        subprocess.check_call(
            [sys.executable, "-c", code],
            cwd=str(RAG_DIR),
        )


class TestBackbonePublicAPI(unittest.TestCase):
    def test_classic_mode_rejects_axis(self):
        from mas_langgraph import CooperativeBackbone

        bb = CooperativeBackbone(orchestrator_mode="classic")
        with self.assertRaises(RuntimeError):
            bb.factual_run_result()

    def test_langgraph_run_id_and_factual_access(self):
        from mas_langgraph import CooperativeBackbone

        bb = CooperativeBackbone(orchestrator_mode="langgraph")
        with self.assertRaises(RuntimeError):
            bb.explain_artifact(ArtifactRef("Q", 1), "why?")
        summary = bb.run_full_workflow()
        factual = bb.factual_run_result()
        self.assertEqual(summary["run_id"], factual.run_id)
        self.assertEqual(factual.branch_id, "factual")


class TestPrompt16EndToEnd(unittest.TestCase):
    def test_full_mcq_axis_pipeline(self):
        case, ex, factual = _build_seeded_factual_run("run_p16_full")
        self.assertIsNone(factual.error)
        self.assertTrue(factual.checkpoint_index)
        self.assertFalse(
            any(
                str(r.checkpoint_id).startswith("synth_")
                for r in factual.checkpoint_index.records
            )
        )
        labels = [o.labeled_id for o in factual.occurrences]
        self.assertIn("verify_candidate_mcqs_against_evidence#1", labels)
        self.assertIn("verify_candidate_mcqs_against_evidence#2", labels)
        self.assertIn("apply_acceptance_gate#1", labels)

        idx_before = list(factual.checkpoint_index.records)
        reg_before = {
            k: str(v.ref)
            for k, v in (factual.artifact_registry.catalog if factual.artifact_registry else {}).items()
        }
        produced_before = list(factual.produced_by)
        relations_before = list(factual.provenance_relations)
        packs_before = list(
            getattr(factual.final_state.get("cooperative_state"), "final_quiz_packs", [])
            or []
        )

        service = AxisExplanationService(
            case_definition=case,
            factual_run_result=factual,
            replay_executor=ex.replay_executor,
            checkpoint_index=factual.checkpoint_index,
        )
        refs = service.list_explainable_artifacts()
        self.assertGreaterEqual(len(refs), 1)
        # Prefer latest historical version when present
        artifact_ref = refs[-1]
        explained = service.resolve_case(artifact_ref)
        self.assertEqual(explained.artifact_ref, artifact_ref)
        self.assertEqual(explained.assessment_ref.version, artifact_ref.version)
        self.assertEqual(explained.decision_ref.version, artifact_ref.version)
        if len(refs) >= 2:
            older = service.resolve_case(refs[0])
            self.assertEqual(older.artifact_ref.version, refs[0].version)
            self.assertNotEqual(older.artifact_ref, explained.artifact_ref)

        from mas_langgraph.axis.slicing import ProvenanceSlicer

        case_slice = ProvenanceSlicer().slice_run_result(
            explained,
            factual,
            adapter=case.load_provenance_slice_adapter(),
            strict=True,
        )
        self.assertGreater(len(case_slice.relevant_artifact_refs), 0)
        self.assertGreater(len(case_slice.relevant_occurrences), 0)

        # Dynamic evidence target + real boundary
        ev_refs = [
            r
            for r in case_slice.relevant_artifact_refs
            if r.logical_id.startswith("E")
        ]
        self.assertTrue(ev_refs)
        target = ev_refs[0]
        boundary_label = None
        origin = None
        for occ in case_slice.relevant_occurrences:
            if occ.node_id != "verify_candidate_mcqs_against_evidence":
                continue
            sel = BoundarySelector.before_occurrence(
                occ.node_id, int(occ.occurrence_index)
            )
            try:
                ck = factual.checkpoint_index.resolve(sel)
            except Exception:
                continue
            snap = ex.replay_executor.get_checkpoint_snapshot(
                factual.run_id, BoundarySelector.by_alias(ck.alias)
            )
            vals = getattr(snap, "values", snap)
            if artifact_exists_at_checkpoint(vals, target):
                boundary_label = f"before {occ.labeled_id}"
                origin = ck
                break
        self.assertIsNotNone(boundary_label)
        self.assertIsNotNone(origin)
        self.assertFalse(str(origin.checkpoint_id).startswith("synth_"))

        remove_q = f"REMOVE({target.logical_id}, {boundary_label})"
        question = "Why was this question accepted?"
        result = service.explain(
            artifact_ref=artifact_ref,
            question=question,
            scripted_queries=[remove_q, "DONE"],
        )

        self.assertEqual(len(result.factual_records), 0)
        self.assertEqual(len(result.intervention_records), 1)
        ir = result.intervention_records[0]
        self.assertIsNone(ir.error)
        self.assertIsNotNone(ir.noop_result)
        self.assertIsNotNone(ir.counterfactual_result)
        self.assertEqual(ir.noop_result.branch_metadata.parent_branch_id, "factual")
        self.assertEqual(
            ir.counterfactual_result.branch_metadata.parent_branch_id, "factual"
        )
        self.assertEqual(
            ir.noop_result.branch_metadata.origin_checkpoint_id,
            ir.counterfactual_result.branch_metadata.origin_checkpoint_id,
        )
        self.assertTrue(result.replay_comparisons)
        status = result.replay_comparisons[0].evidence_status
        cmp0 = result.replay_comparisons[0]
        self.assertIsNone(cmp0.noop.error)
        self.assertTrue(cmp0.noop.outcome_available)
        self.assertTrue(getattr(cmp0.noop, "subject_present", True))
        self.assertNotEqual(status, "noop_error")
        self.assertIn(
            status,
            {
                "decision_effect",
                "criterion_effect",
                "downstream_effect",
                "no_demonstrated_effect",
                "replay_unstable",
            },
        )
        self.assertTrue(result.final_explanation)
        self.assertIn(str(explained.outcome), result.final_explanation.lower())
        if status == "no_demonstrated_effect":
            low = result.final_explanation.lower()
            for banned in ("caused", "necessary", "proven cause"):
                self.assertNotIn(banned, low)

        with tempfile.TemporaryDirectory() as td:
            path = dump_axis_explanation(
                result,
                str(Path(td) / "axis_e2e.json"),
                question=question,
                case_definition=case,
            )
            loaded = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema_version"], 1)
            self.assertEqual(loaded["run_id"], factual.run_id)

        # Factual immutability
        self.assertEqual(
            [r.checkpoint_id for r in factual.checkpoint_index.records],
            [r.checkpoint_id for r in idx_before],
        )
        reg_after = {
            k: str(v.ref)
            for k, v in (factual.artifact_registry.catalog if factual.artifact_registry else {}).items()
        }
        self.assertEqual(reg_before, reg_after)
        self.assertEqual(len(factual.produced_by), len(produced_before))
        self.assertEqual(len(factual.provenance_relations), len(relations_before))
        packs_after = list(
            getattr(factual.final_state.get("cooperative_state"), "final_quiz_packs", [])
            or []
        )
        self.assertEqual(len(packs_before), len(packs_after))


class TestCLISmoke(unittest.TestCase):
    def test_langgraph_stub_cli(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "mas_langgraph",
                "--orchestrate",
                "--stub",
                "--orchestrator",
                "langgraph",
            ],
            cwd=str(RAG_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])

    def test_classic_stub_cli(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "mas_langgraph",
                "--orchestrate",
                "--stub",
                "--orchestrator",
                "classic",
            ],
            cwd=str(RAG_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
