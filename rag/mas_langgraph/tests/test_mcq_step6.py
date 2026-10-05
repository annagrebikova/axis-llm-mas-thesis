"""Smoke tests for Step 6 Cross-CoVe answer verification."""

from __future__ import annotations

import json
import unittest

from mas_langgraph.core.entities import DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
from mas_langgraph.core.state import CooperativeState
from mas_langgraph.domains.quiz.blueprint_planner import create_mcq_blueprints_from_coverage
from mas_langgraph.domains.quiz.coverage import apply_acceptance_gate
from mas_langgraph.domains.quiz.cross_cove import (
    build_cross_cove_prompt,
    majority_vote,
    parse_verifier_answer,
    run_cross_cove_for_candidate,
    run_cross_cove_for_all_candidates,
)
from mas_langgraph.domains.quiz.entities import (
    AnswerOption,
    CandidateMCQ,
    CrossCoVeReport,
    EvidenceCheckReport,
    SourceRef,
)
from mas_langgraph.domains.quiz.export import export_mcqs_for_external_iwf_check
from mas_langgraph.orchestrator.langgraph_workflow import node_cross_cove_verify_candidate_mcqs
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
            objective_text="Explain k-means steps.",
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


def _sample_mcq() -> CandidateMCQ:
    return CandidateMCQ(
        q_id="q_cross",
        blueprint_id="bp_cross",
        stem="What does k-means assign each point to?",
        options=[
            AnswerOption(option_id="A", text="Nearest centroid", is_correct=True),
            AnswerOption(option_id="B", text="Random cluster", is_correct=False),
            AnswerOption(option_id="C", text="Previous label", is_correct=False),
            AnswerOption(option_id="D", text="Outlier bucket", is_correct=False),
        ],
        correct_option_id="A",
        answer_explanation="k-means uses nearest centroid assignment.",
        source_refs=[SourceRef(doc_id="141", page=8, chunk_id="chunk_kmeans", atom_id="atom_kmeans")],
    )


def _passing_evidence(state: CooperativeState, mcq: CandidateMCQ) -> None:
    state.store_evidence_report(
        EvidenceCheckReport(
            q_id=mcq.q_id,
            supported_by_pdf=True,
            unsupported_claims=[],
            source_refs=mcq.source_refs,
            confidence=0.9,
            stem_supported=True,
            answer_supported=True,
            distractors_valid=True,
            explanation_supported=True,
        )
    )


class TestCrossCoVeStep6(unittest.TestCase):
    def test_majority_vote_two_of_three(self):
        majority, confidence = majority_vote(["A", "A", "B"])
        self.assertEqual(majority, "A")
        self.assertAlmostEqual(confidence, 0.67, places=2)

    def test_majority_vote_tie_returns_none(self):
        majority, confidence = majority_vote(["A", "B", "C"])
        self.assertIsNone(majority)
        self.assertEqual(confidence, 0.0)

    def test_parse_verifier_answer_from_json(self):
        raw = json.dumps({"selected_option_id": "B", "reason": "best fit"})
        label, notes = parse_verifier_answer(raw)
        self.assertEqual(label, "B")
        self.assertEqual(notes, [])

    def test_build_cross_cove_prompt_hides_correct_answer(self):
        mcq = _sample_mcq()
        prompt = build_cross_cove_prompt(mcq)
        self.assertIn(mcq.stem, prompt)
        self.assertNotIn(mcq.answer_explanation, prompt)
        self.assertNotIn("is_correct", prompt.lower())

    def test_cross_cove_passed_when_majority_matches(self):
        mcq = _sample_mcq()
        report = run_cross_cove_for_candidate(mcq, verifier_llm=None, n_runs=3)
        self.assertTrue(report.passed)
        self.assertEqual(report.generator_answer, "A")
        self.assertEqual(report.majority_answer, "A")
        self.assertEqual(len(report.verifier_answers), 3)

    def test_cross_cove_failed_when_majority_differs(self):
        mcq = _sample_mcq()

        class WrongStubLLM:
            def invoke(self, prompt: str):
                return json.dumps({"selected_option_id": "B", "reason": "wrong"})

        report = run_cross_cove_for_candidate(mcq, verifier_llm=WrongStubLLM(), n_runs=3)
        self.assertFalse(report.passed)
        self.assertEqual(report.majority_answer, "B")

    def test_cross_cove_failed_not_accepted(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        mcq = _sample_mcq()
        mcq.blueprint_id = blueprint.blueprint_id
        mcq.flags.append("cross_cove_failed")
        state.store_mcq_blueprints([blueprint])
        state.store_candidate_mcq(mcq)
        _passing_evidence(state, mcq)
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
        decision = apply_acceptance_gate(mcq, state)
        self.assertNotEqual(decision.decision, "accept")
        self.assertEqual(mcq.status, "needs_revision")

    def test_missing_cross_cove_report_not_accepted(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        mcq = _sample_mcq()
        mcq.blueprint_id = blueprint.blueprint_id
        state.store_mcq_blueprints([blueprint])
        state.store_candidate_mcq(mcq)
        _passing_evidence(state, mcq)
        decision = apply_acceptance_gate(mcq, state)
        self.assertEqual(decision.decision, "revise")
        self.assertIn("needs_cross_cove_verification", decision.required_fixes)

    def test_langgraph_node_stores_cross_cove_reports(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        mcq = _sample_mcq()
        mcq.blueprint_id = blueprint.blueprint_id
        state.store_mcq_blueprints([blueprint])
        state.store_candidate_mcq(mcq)
        _passing_evidence(state, mcq)

        graph_state = {
            "cooperative_state": state,
            "step_logs": [],
            "error": None,
        }
        runtime = workflow_runtime_context(orchestrator_config=DEFAULT_ORCHESTRATOR_CONFIG)
        update = node_cross_cove_verify_candidate_mcqs(graph_state, runtime)
        self.assertIn(mcq.q_id, update["cooperative_state"].cross_cove_reports)
        self.assertTrue(update["cooperative_state"].cross_cove_reports[mcq.q_id].passed)

    def test_export_includes_cross_cove_report(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        mcq = _sample_mcq()
        mcq.blueprint_id = blueprint.blueprint_id
        mcq.status = "accepted"
        state.store_mcq_blueprints([blueprint])
        state.store_candidate_mcq(mcq)
        _passing_evidence(state, mcq)
        run_cross_cove_for_all_candidates(state, verifier_llm=None)
        apply_acceptance_gate(mcq, state)
        mcq.status = "accepted"
        exported = export_mcqs_for_external_iwf_check(state)
        self.assertEqual(len(exported), 1)
        self.assertIn("cross_cove_report", exported[0])
        self.assertTrue(exported[0]["cross_cove_report"]["passed"])


if __name__ == "__main__":
    unittest.main()
