"""Smoke tests for Step 4 PDF evidence grounding verification."""

from __future__ import annotations

import unittest

from mas_langgraph.core.entities import DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
from mas_langgraph.core.state import CooperativeState
from mas_langgraph.domains.quiz.blueprint_planner import create_mcq_blueprints_from_coverage
from mas_langgraph.domains.quiz.entities import AnswerOption, CandidateMCQ, SourceRef
from mas_langgraph.domains.quiz.export import export_mcqs_for_external_iwf_check
from mas_langgraph.domains.quiz.generation import generate_candidate_from_blueprint, register_candidate_mcq
from mas_langgraph.domains.quiz.verification import (
    build_evidence_verification_context,
    verify_all_candidate_mcqs,
    verify_candidate_against_evidence,
)
from mas_langgraph.orchestrator.langgraph_workflow import (
    node_generate_candidate_mcqs_from_blueprints,
    node_verify_candidate_mcqs_against_evidence,
)
from mas_langgraph.orchestrator.runtime_helpers import workflow_runtime_context


def _fake_state() -> CooperativeState:
    state = CooperativeState()
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
    chunk = EvidenceChunk(
        chunk_id="chunk_kmeans",
        content="k-means assigns each point to the nearest centroid, then updates centroids.",
        metadata={"page_number": 8, "pdf_id": "141", "source_file": "141.pdf"},
        quality_score=0.9,
    )
    state.add_evidence_chunk(chunk)
    atom = KnowledgeAtom(
        atom_id="atom_kmeans",
        content="k-means alternates assignment and centroid recomputation.",
        atom_type="procedure",
        citations=[{"chunk_id": "chunk_kmeans", "page": 8}],
        section_id="sec_kmeans",
        lo_id="lo_kmeans_steps",
    )
    state.add_knowledge_atom(atom)
    state.link_chunk_to_atom("chunk_kmeans", "atom_kmeans")
    return state


def _graph_state(state: CooperativeState) -> dict:
    return {
        "cooperative_state": state,
        "step_logs": [],
        "error": None,
    }


def _test_runtime(state: CooperativeState):
    return workflow_runtime_context(
        agent_runner=type("StubRunner", (), {"_quiz_generation_rounds": 0})(),
    )


class TestMCQEvidenceVerificationStep4(unittest.TestCase):
    def test_build_evidence_verification_context(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        state.store_mcq_blueprints([blueprint])
        mcq = generate_candidate_from_blueprint(blueprint, state, llm=None)
        context = build_evidence_verification_context(mcq, state)
        self.assertEqual(context["q_id"], mcq.q_id)
        self.assertEqual(context["section_title"], "k-Means")
        self.assertEqual(context["learning_objective"], "Explain k-means assignment and update steps.")
        self.assertEqual(len(context["evidence_chunks"]), 1)
        self.assertEqual(len(context["knowledge_atoms"]), 1)
        self.assertTrue(context["source_refs"])
        self.assertTrue(context["stem"])
        self.assertEqual(len(context["options"]), 4)

    def test_missing_evidence_returns_not_supported(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        blueprint.evidence_chunk_ids = []
        state.store_mcq_blueprints([blueprint])
        mcq = generate_candidate_from_blueprint(blueprint, state, llm=None)
        report = verify_candidate_against_evidence(mcq, state, llm=None)
        self.assertFalse(report.supported_by_pdf)
        self.assertIn("missing_evidence", report.required_fixes)
        self.assertEqual(mcq.status, "needs_revision")

    def test_missing_source_refs_gets_required_fixes(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        state.store_mcq_blueprints([blueprint])
        mcq = CandidateMCQ(
            q_id="q_no_refs",
            blueprint_id=blueprint.blueprint_id,
            stem="What does k-means assign each point to?",
            options=[
                AnswerOption(option_id="A", text="Nearest centroid", is_correct=True),
                AnswerOption(option_id="B", text="Random cluster", is_correct=False),
                AnswerOption(option_id="C", text="Previous label", is_correct=False),
                AnswerOption(option_id="D", text="Outlier bucket", is_correct=False),
            ],
            correct_option_id="A",
            answer_explanation="k-means assigns each point to the nearest centroid.",
            source_refs=[],
        )
        report = verify_candidate_against_evidence(mcq, state, llm=None)
        self.assertFalse(report.supported_by_pdf)
        self.assertIn("missing_source_refs", report.required_fixes)
        self.assertIn("missing_source_refs", mcq.flags)

    def test_evidence_report_stored_in_state(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        state.store_mcq_blueprints([blueprint])
        mcq = generate_candidate_from_blueprint(blueprint, state, llm=None)
        state.store_candidate_mcq(mcq)
        reports = verify_all_candidate_mcqs(state, llm=None)
        self.assertEqual(len(reports), 1)
        self.assertIn(mcq.q_id, state.evidence_reports)
        self.assertEqual(state.evidence_reports[mcq.q_id].q_id, mcq.q_id)

    def test_langgraph_node_runs_after_generation(self):
        state = _fake_state()
        blueprints = create_mcq_blueprints_from_coverage(state)
        state.store_mcq_blueprints(blueprints)
        graph_state = _graph_state(state)
        runtime = _test_runtime(state)

        gen_update = node_generate_candidate_mcqs_from_blueprints(graph_state, runtime)
        self.assertIn("cooperative_state", gen_update)
        graph_state["cooperative_state"] = gen_update["cooperative_state"]
        self.assertGreaterEqual(len(graph_state["cooperative_state"].candidate_mcqs), 1)

        verify_update = node_verify_candidate_mcqs_against_evidence(graph_state, runtime)
        self.assertIn("cooperative_state", verify_update)
        graph_state["cooperative_state"] = verify_update["cooperative_state"]
        self.assertEqual(
            len(graph_state["cooperative_state"].evidence_reports),
            len(graph_state["cooperative_state"].candidate_mcqs),
        )

    def test_supported_candidate_status_evidence_verified(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        state.store_mcq_blueprints([blueprint])
        mcq = CandidateMCQ(
            q_id="q_supported",
            blueprint_id=blueprint.blueprint_id,
            stem="What does k-means assign each point to during assignment?",
            options=[
                AnswerOption(option_id="A", text="Nearest centroid", is_correct=True),
                AnswerOption(option_id="B", text="Random cluster", is_correct=False),
                AnswerOption(option_id="C", text="Previous label", is_correct=False),
                AnswerOption(option_id="D", text="Outlier bucket", is_correct=False),
            ],
            correct_option_id="A",
            answer_explanation="k-means assigns each point to the nearest centroid before updating centroids.",
            source_refs=[
                SourceRef(doc_id="141", page=8, chunk_id="chunk_kmeans", atom_id="atom_kmeans"),
            ],
        )
        report = verify_candidate_against_evidence(mcq, state, llm=None)
        self.assertTrue(report.supported_by_pdf)
        self.assertEqual(mcq.status, "evidence_verified")

    def test_unsupported_candidate_status_needs_revision(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        state.store_mcq_blueprints([blueprint])
        mcq = CandidateMCQ(
            q_id="q_unsupported",
            blueprint_id=blueprint.blueprint_id,
            stem="What is the capital of France?",
            options=[
                AnswerOption(option_id="A", text="Paris", is_correct=True),
                AnswerOption(option_id="B", text="Berlin", is_correct=False),
                AnswerOption(option_id="C", text="Madrid", is_correct=False),
                AnswerOption(option_id="D", text="Rome", is_correct=False),
            ],
            correct_option_id="A",
            answer_explanation="Paris is the capital of France.",
            source_refs=[
                SourceRef(doc_id="141", page=8, chunk_id="chunk_kmeans", atom_id="atom_kmeans"),
            ],
        )
        report = verify_candidate_against_evidence(mcq, state, llm=None)
        self.assertFalse(report.supported_by_pdf)
        self.assertEqual(mcq.status, "needs_revision")
        self.assertIn("evidence_not_supported", mcq.flags)

    def test_export_includes_evidence_verification_fields(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        mcq = generate_candidate_from_blueprint(blueprint, state, llm=None)
        register_candidate_mcq(state, mcq, blueprint)
        verify_all_candidate_mcqs(state, llm=None)
        exported = export_mcqs_for_external_iwf_check(state, accepted_only=False)
        self.assertEqual(len(exported), 1)
        payload = exported[0]
        self.assertIn("evidence_supported_by_pdf", payload)
        self.assertIn("evidence_confidence", payload)
        self.assertIn("unsupported_claims", payload)
        self.assertIn("required_fixes", payload)
        self.assertIn("candidate_flags", payload)
        self.assertIn("source_refs", payload)


if __name__ == "__main__":
    unittest.main()
