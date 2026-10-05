"""Regression: acceptance must persist OutcomeAssessment into the factual registry.

Production RAG blackboards historically lacked ``outcome_assessments`` /
``store_outcome_assessment``. Decisions were stored and CandidateMCQs accepted,
but ``MCQExplainedCaseResolver`` correctly required both assessment and decision
envelopes — so ``list_explainable_artifacts()`` returned empty.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

from mas_langgraph.artifacts.channel_state import (
    dump_artifact_channels,
    empty_artifact_channels,
    load_artifact_channels,
)
from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.cases.mcq.artifact_registration import register_mcq_artifacts_for_stage
from mas_langgraph.cases.mcq.explained_case import MCQExplainedCaseResolver
from mas_langgraph.core.entities import DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
from mas_langgraph.core.outcomes import OutcomeAssessment
from mas_langgraph.domains.quiz.blueprint_planner import create_mcq_blueprints_from_coverage
from mas_langgraph.domains.quiz.coverage import apply_acceptance_gate_to_all
from mas_langgraph.domains.quiz.cross_cove import run_cross_cove_for_candidate
from mas_langgraph.domains.quiz.entities import (
    AnswerOption,
    CandidateMCQ,
    SourceRef,
)
from mas_langgraph.domains.quiz.state_integration import ensure_step2_mcq_state
from mas_langgraph.domains.quiz.verification import verify_candidate_against_evidence
from mas_langgraph.input.orchestrator_config import DEFAULT_ORCHESTRATOR_CONFIG
from mas_langgraph.orchestrator.langgraph_workflow import node_apply_acceptance_gate
from mas_langgraph.orchestrator.runtime_helpers import workflow_runtime_context
from mas_langgraph.provenance.channel_state import empty_provenance_channels
from mas_langgraph.replay.executor import WorkflowRunResult


RAG_DIR = Path(__file__).resolve().parents[2]


class _RAGLikeBlackboard:
    """Mirrors rag_agents_cooperative.CooperativeState: Step-2 fields, no Prompt-8 assessments."""

    def __init__(self) -> None:
        self.doc_outline: list = []
        self.learning_objectives: list = []
        self.knowledge_atoms: dict = {}
        self.evidence_pool: dict = {}
        self.candidate_questions: dict = {}
        self.validated_questions: dict = {}
        self.rejected_questions: dict = {}
        self.final_quiz_packs: list = []
        self.quality_metrics = type(
            "QM",
            (),
            {
                "duplication_score": 0.0,
                "rejection_rate": 0.0,
                "avg_evidence_strength": 0.0,
            },
        )()
        # Step-2 fields appear only after ensure_step2_mcq_state (as in production).

    def add_section(self, section: Any) -> None:
        self.doc_outline.append(section)

    def add_learning_objective(self, lo: Any) -> None:
        self.learning_objectives.append(lo)

    def add_knowledge_atom(self, atom: Any) -> None:
        self.knowledge_atoms[atom.atom_id] = atom

    def add_evidence_chunk(self, chunk: Any) -> None:
        self.evidence_pool[chunk.chunk_id] = chunk

    def get_chunk(self, chunk_id: str) -> Any:
        return self.evidence_pool.get(chunk_id)

    def get_section(self, section_id: str) -> Any:
        for section in self.doc_outline:
            if getattr(section, "section_id", None) == section_id:
                return section
        return None

    def get_atom(self, atom_id: str) -> Any:
        return self.knowledge_atoms.get(atom_id)

    def generate_id(self, prefix: str) -> str:
        import uuid

        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def link_chunk_to_atom(self, chunk_id: str, atom_id: str) -> bool:
        chunk = self.evidence_pool.get(chunk_id)
        atom = self.knowledge_atoms.get(atom_id)
        if chunk is None or atom is None:
            return False
        used = getattr(chunk, "used_in_atoms", None)
        if used is None:
            chunk.used_in_atoms = []
            used = chunk.used_in_atoms
        if atom_id not in used:
            used.append(atom_id)
        return True


def _seed_rag_like() -> _RAGLikeBlackboard:
    state = _RAGLikeBlackboard()
    ensure_step2_mcq_state(state)
    state.add_section(
        DocSection(
            section_id="sec_kmeans",
            title="k-Means",
            slide_range=(6, 10),
            importance_weight=0.9,
            content_summary="k-Means clustering algorithm.",
        )
    )
    state.add_learning_objective(
        LearningObjective(
            lo_id="lo_kmeans_steps",
            section_id="sec_kmeans",
            objective_text="Explain k-means assignment and update steps.",
            difficulty_level="intermediate",
        )
    )
    state.add_evidence_chunk(
        EvidenceChunk(
            chunk_id="chunk_kmeans",
            content="k-means assigns each point to the nearest centroid, then updates centroids.",
            metadata={"page_number": 8, "pdf_id": "141", "source_file": "141.pdf"},
            quality_score=0.9,
        )
    )
    state.add_knowledge_atom(
        KnowledgeAtom(
            atom_id="atom_kmeans",
            content="k-means alternates assignment and centroid recomputation.",
            atom_type="procedure",
            citations=[{"chunk_id": "chunk_kmeans", "page": 8}],
            section_id="sec_kmeans",
            lo_id="lo_kmeans_steps",
        )
    )
    state.link_chunk_to_atom("chunk_kmeans", "atom_kmeans")
    return state


def _accepted_candidate(state: Any) -> CandidateMCQ:
    blueprint = create_mcq_blueprints_from_coverage(state)[0]
    state.store_mcq_blueprints([blueprint])
    mcq = CandidateMCQ(
        q_id="q_rag_explainable",
        blueprint_id=blueprint.blueprint_id,
        stem="What does k-means assign each point to during assignment?",
        options=[
            AnswerOption(option_id="A", text="Nearest centroid", is_correct=True),
            AnswerOption(option_id="B", text="Random cluster", is_correct=False),
            AnswerOption(option_id="C", text="Previous label", is_correct=False),
            AnswerOption(option_id="D", text="Outlier bucket", is_correct=False),
        ],
        correct_option_id="A",
        answer_explanation="k-means assigns each point to the nearest centroid before updating.",
        source_refs=[
            SourceRef(
                doc_id="141", page=8, chunk_id="chunk_kmeans", atom_id="atom_kmeans"
            ),
        ],
        status="evidence_verified",
    )
    state.store_candidate_mcq(mcq)
    state.store_evidence_report(verify_candidate_against_evidence(mcq, state, llm=None))
    state.store_cross_cove_report(run_cross_cove_for_candidate(mcq, verifier_llm=None))
    return mcq


def _strip_assessment_storage(state: Any) -> None:
    """Reproduce pre-fix RAG blackboard: decisions yes, assessments no."""
    if hasattr(state, "outcome_assessments"):
        delattr(state, "outcome_assessments")
    if "store_outcome_assessment" in getattr(state, "__dict__", {}):
        delattr(state, "store_outcome_assessment")


class TestExplainableAcceptancePath(unittest.TestCase):
    def test_rag_like_gate_registers_assessment_and_decision(self):
        state = _seed_rag_like()
        mcq = _accepted_candidate(state)
        _strip_assessment_storage(state)
        self.assertFalse(hasattr(state, "outcome_assessments"))
        self.assertFalse(
            hasattr(state, "store_outcome_assessment")
            and "store_outcome_assessment" in state.__dict__
        )

        graph_state: dict = {
            "cooperative_state": state,
            "run_id": "run_rag_explainable",
            "branch_id": "factual",
            "step_logs": [],
            "error": None,
            **empty_artifact_channels(),
            **empty_provenance_channels(),
        }
        # Seed candidate envelope as generate_candidate_mcqs would.
        reg, resolver = load_artifact_channels(graph_state)
        register_mcq_artifacts_for_stage(
            reg,
            resolver,
            state,
            stage="candidates",
            run_id="run_rag_explainable",
            branch_id="factual",
        )
        graph_state.update(dump_artifact_channels(reg, resolver))

        runtime = workflow_runtime_context(
            agent_runner=type("StubRunner", (), {})(),
            orchestrator_config=DEFAULT_ORCHESTRATOR_CONFIG,
            case_definition=__import__(
                "mas_langgraph.cases.mcq", fromlist=["MCQCaseDefinition"]
            ).MCQCaseDefinition(),
        )
        update = node_apply_acceptance_gate(graph_state, runtime)
        working = update["cooperative_state"]
        accepted = working.candidate_mcqs[mcq.q_id]
        self.assertEqual(accepted.status, "accepted")
        self.assertIn(mcq.q_id, working.outcome_assessments)
        assessment = working.outcome_assessments[mcq.q_id]
        self.assertIsInstance(assessment, OutcomeAssessment)
        self.assertEqual(assessment.target_ref, ArtifactRef(mcq.q_id, 1))

        catalog = update.get("artifact_catalog") or {}
        types = {getattr(e, "artifact_type", None) for e in catalog.values()}
        self.assertIn("CandidateMCQ", types)
        self.assertIn("OutcomeAssessment", types)
        self.assertIn("SupervisorDecision", types)

        fr = WorkflowRunResult(
            run_id="run_rag_explainable",
            branch_id="factual",
            final_state={
                "artifact_catalog": dict(catalog),
                "branch_id": "factual",
                "run_id": "run_rag_explainable",
            },
        )
        refs = MCQExplainedCaseResolver().list_explainable_artifacts(
            factual_run_result=fr
        )
        self.assertEqual(refs, (ArtifactRef(mcq.q_id, 1),))
        case = MCQExplainedCaseResolver().resolve(
            factual_run_result=fr, artifact_ref=refs[0]
        )
        self.assertEqual(case.assessment_ref, ArtifactRef(f"assessment:{mcq.q_id}", 1))
        self.assertEqual(case.decision_ref, ArtifactRef(f"decision:{mcq.q_id}", 1))
        self.assertEqual(case.outcome, "accept")

    def test_apply_acceptance_gate_to_all_stores_assessments_on_rag_like(self):
        state = _seed_rag_like()
        mcq = _accepted_candidate(state)
        _strip_assessment_storage(state)
        ensure_step2_mcq_state(state)
        decisions = apply_acceptance_gate_to_all(state, cross_cove_required=True)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].decision, "accept")
        self.assertIn(mcq.q_id, state.outcome_assessments)
        self.assertEqual(
            decisions[0].assessment_ref,
            state.outcome_assessments[mcq.q_id].assessment_ref,
        )


class TestRagMonolithGraphExplainable(unittest.TestCase):
    """Real compiled MCQ graph on the production RAG CooperativeState class."""

    def test_seeded_graph_lists_explainable_from_rag_blackboard(self):
        if str(RAG_DIR) not in sys.path:
            sys.path.insert(0, str(RAG_DIR))
        from rag_agents_cooperative import CooperativeState as RAGCooperativeState

        from mas_langgraph.axis.service import AxisExplanationService
        from mas_langgraph.integrations.rag_agents_bridge import attach_snapshot_counts
        from mas_langgraph.tests.test_prompt16_end_to_end import _build_seeded_factual_run

        cs = RAGCooperativeState()
        attach_snapshot_counts(cs)
        self.assertFalse(
            hasattr(RAGCooperativeState, "store_outcome_assessment"),
            "RAG monolith class must not define Prompt-8 assessment storage",
        )
        # Seed outline/evidence/atoms the same way as the Prompt-16 fixture.
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
            ("E17", "k-means assigns points to nearest centroids then updates them."),
            ("E18", "Objective minimizes within-cluster sum of squares."),
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

        case, ex, factual = _build_seeded_factual_run(
            "run_rag_blackboard_axis",
            cooperative_state=cs,
        )
        self.assertIsNone(factual.error)
        reg = factual.artifact_registry
        self.assertIsNotNone(reg)
        by_type: dict = {}
        for env in (reg.catalog if reg else {}).values():
            by_type.setdefault(env.artifact_type, []).append(env)
        self.assertGreaterEqual(len(by_type.get("CandidateMCQ", [])), 1)
        self.assertGreaterEqual(len(by_type.get("OutcomeAssessment", [])), 1)
        self.assertGreaterEqual(len(by_type.get("SupervisorDecision", [])), 1)

        service = AxisExplanationService(
            case_definition=case,
            factual_run_result=factual,
            replay_executor=ex.replay_executor,
            checkpoint_index=factual.checkpoint_index,
        )
        refs = service.list_explainable_artifacts()
        self.assertGreaterEqual(len(refs), 1)
        # Prefer an accepted version when revision produced v1=revise, v2=accept.
        explained = None
        for ref in reversed(refs):
            case_ex = service.resolve_case(ref)
            if case_ex.outcome == "accept":
                explained = case_ex
                break
        self.assertIsNotNone(explained)
        self.assertEqual(explained.assessment_ref.version, explained.artifact_ref.version)
        self.assertEqual(explained.decision_ref.version, explained.artifact_ref.version)
        self.assertEqual(explained.outcome, "accept")
        # Historical revise versions remain independently listable.
        self.assertTrue(all(isinstance(r, ArtifactRef) for r in refs))


if __name__ == "__main__":
    unittest.main()
