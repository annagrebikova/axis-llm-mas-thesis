"""Smoke tests for Step 7 MAS-internal revision loop."""

from __future__ import annotations

import unittest

from mas_langgraph.core.entities import DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
from mas_langgraph.core.state import CooperativeState
from mas_langgraph.domains.quiz.blueprint_planner import create_mcq_blueprints_from_coverage
from mas_langgraph.domains.quiz.coverage import apply_acceptance_gate
from mas_langgraph.domains.quiz.cross_cove import run_cross_cove_for_candidate
from mas_langgraph.domains.quiz.entities import (
    AnswerOption,
    CandidateMCQ,
    CrossCoVeReport,
    EvidenceCheckReport,
    SourceRef,
)
from mas_langgraph.domains.quiz.export import export_mcqs_for_external_iwf_check
from mas_langgraph.domains.quiz.packaging import package_final_quiz
from mas_langgraph.domains.quiz.revision import (
    reset_reports_after_revision,
    revise_all_needed_candidates,
    revise_candidate_mcq,
)
from mas_langgraph.domains.quiz.verification import verify_candidate_against_evidence
from mas_langgraph.orchestrator.langgraph_workflow import (
    node_apply_acceptance_gate,
    node_revise_candidate_mcqs,
)
from mas_langgraph.orchestrator.runtime_helpers import workflow_runtime_context
from mas_langgraph.input.orchestrator_config import DEFAULT_ORCHESTRATOR_CONFIG


def _fake_state() -> CooperativeState:
    state = CooperativeState()
    state.add_section(
        DocSection(
            section_id="sec_kmeans",
            title="k-Means",
            slide_range=(6, 10),
            importance_weight=0.9,
            content_summary="k-Means clustering.",
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
        metadata={"page_number": 8, "pdf_id": "141"},
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


def _failing_mcq(state: CooperativeState) -> CandidateMCQ:
    blueprint = create_mcq_blueprints_from_coverage(state)[0]
    state.store_mcq_blueprints([blueprint])
    mcq = CandidateMCQ(
        q_id="q_revise",
        blueprint_id=blueprint.blueprint_id,
        stem="What is the capital of France?",
        options=[
            AnswerOption(option_id="A", text="Paris", is_correct=True),
            AnswerOption(option_id="B", text="Berlin", is_correct=False),
            AnswerOption(option_id="C", text="Madrid", is_correct=False),
            AnswerOption(option_id="D", text="Rome", is_correct=False),
        ],
        correct_option_id="A",
        answer_explanation="Paris is the capital.",
        source_refs=[SourceRef(doc_id="141", page=8, chunk_id="chunk_kmeans", atom_id="atom_kmeans")],
        status="needs_revision",
        flags=["evidence_not_supported", "cross_cove_failed"],
    )
    state.store_candidate_mcq(mcq)
    state.store_evidence_report(
        EvidenceCheckReport(
            q_id=mcq.q_id,
            supported_by_pdf=False,
            unsupported_claims=["Stem not grounded."],
            source_refs=mcq.source_refs,
            confidence=0.2,
            stem_supported=False,
            answer_supported=False,
            distractors_valid=True,
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
            confidence=1.0,
        )
    )
    return mcq


class TestRevisionStep7(unittest.TestCase):
    def test_needs_revision_candidate_is_revised(self):
        state = _fake_state()
        mcq = _failing_mcq(state)
        original_stem = mcq.stem
        original_count = mcq.revision_count
        revised = revise_candidate_mcq(mcq, state, llm=None, max_revisions=2)
        self.assertIsNot(revised, mcq)
        self.assertEqual(mcq.revision_count, original_count)
        self.assertEqual(mcq.stem, original_stem)
        self.assertEqual(revised.revision_count, 1)
        self.assertEqual(revised.status, "draft")
        self.assertNotIn("evidence_not_supported", revised.flags)

    def test_revision_record_added_to_history(self):
        state = _fake_state()
        mcq = _failing_mcq(state)
        revise_candidate_mcq(mcq, state, llm=None, max_revisions=2)
        self.assertEqual(len(state.revision_history), 1)
        self.assertEqual(state.revision_history[0].q_id, mcq.q_id)

    def test_reports_invalidated_after_revision(self):
        state = _fake_state()
        mcq = _failing_mcq(state)
        self.assertIn(mcq.q_id, state.evidence_reports)
        self.assertIn(mcq.q_id, state.cross_cove_reports)
        revise_candidate_mcq(mcq, state, llm=None, max_revisions=2)
        self.assertNotIn(mcq.q_id, state.evidence_reports)
        self.assertNotIn(mcq.q_id, state.cross_cove_reports)
        self.assertNotIn(mcq.q_id, state.supervisor_decisions)

    def test_reset_reports_after_revision(self):
        state = _fake_state()
        mcq = _failing_mcq(state)
        reset_reports_after_revision(mcq, state)
        self.assertNotIn(mcq.q_id, state.evidence_reports)

    def test_max_revisions_rejects_candidate(self):
        state = _fake_state()
        mcq = _failing_mcq(state)
        mcq.revision_count = 2
        original_count = mcq.revision_count
        revised = revise_candidate_mcq(mcq, state, llm=None, max_revisions=2)
        self.assertEqual(mcq.revision_count, original_count)
        self.assertEqual(revised.status, "rejected")
        self.assertIn("max_revisions_reached", revised.flags)

    def test_revised_candidate_reverified_before_acceptance(self):
        state = _fake_state()
        mcq = _failing_mcq(state)
        revise_candidate_mcq(mcq, state, llm=None, max_revisions=2)
        current = state.candidate_mcqs[mcq.q_id]
        report = verify_candidate_against_evidence(current, state, llm=None)
        state.store_evidence_report(report)
        cove = run_cross_cove_for_candidate(current, verifier_llm=None)
        state.store_cross_cove_report(cove)
        decision = apply_acceptance_gate(current, state)
        self.assertEqual(decision.decision, "accept")
        self.assertEqual(current.status, "accepted")

    def test_accepted_candidate_can_have_revision_count(self):
        state = _fake_state()
        mcq = _failing_mcq(state)
        revise_candidate_mcq(mcq, state, llm=None, max_revisions=2)
        current = state.candidate_mcqs[mcq.q_id]
        report = verify_candidate_against_evidence(current, state, llm=None)
        state.store_evidence_report(report)
        cove = run_cross_cove_for_candidate(current, verifier_llm=None)
        state.store_cross_cove_report(cove)
        apply_acceptance_gate(current, state)
        self.assertEqual(current.status, "accepted")
        self.assertGreaterEqual(current.revision_count, 1)

    def test_packager_excludes_needs_revision(self):
        state = _fake_state()
        mcq = _failing_mcq(state)
        pack = package_final_quiz(state, title="Test")
        self.assertEqual(len(pack.questions), 0)
        exported = export_mcqs_for_external_iwf_check(state)
        self.assertEqual(len(exported), 0)

    def test_revise_all_needed_candidates(self):
        state = _fake_state()
        _failing_mcq(state)
        revised = revise_all_needed_candidates(state, llm=None, max_revisions=2)
        self.assertEqual(len(revised), 1)

    def test_langgraph_revision_node(self):
        state = _fake_state()
        _failing_mcq(state)
        graph_state = {
            "cooperative_state": state,
            "step_logs": [],
            "error": None,
            "revision_iterations": 0,
            "max_revision_iterations": 2,
        }
        runtime = workflow_runtime_context(orchestrator_config=DEFAULT_ORCHESTRATOR_CONFIG)
        update = node_revise_candidate_mcqs(graph_state, runtime)
        self.assertEqual(update["revision_iterations"], 1)
        self.assertGreaterEqual(len(update["cooperative_state"].revision_history), 1)


if __name__ == "__main__":
    unittest.main()
