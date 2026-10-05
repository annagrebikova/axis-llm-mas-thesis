"""Smoke tests for Step 5 acceptance gating and coverage control."""

from __future__ import annotations

import unittest

from mas_langgraph.core.entities import DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
from mas_langgraph.core.state import CooperativeState
from mas_langgraph.domains.quiz.blueprint_planner import create_mcq_blueprints_from_coverage
from mas_langgraph.domains.quiz.coverage import (
    apply_acceptance_gate,
    compute_mcq_coverage,
    create_repair_blueprints,
    find_coverage_gaps,
)
from mas_langgraph.domains.quiz.cross_cove import run_cross_cove_for_candidate
from mas_langgraph.domains.quiz.entities import (
    AnswerOption,
    CandidateMCQ,
    EvidenceCheckReport,
    MCQBlueprint,
    SourceRef,
)
from mas_langgraph.domains.quiz.export import export_mcqs_for_external_iwf_check
from mas_langgraph.domains.quiz.generation import generate_candidate_from_blueprint, register_candidate_mcq
from mas_langgraph.domains.quiz.packaging import package_final_quiz
from mas_langgraph.domains.quiz.verification import verify_candidate_against_evidence
from mas_langgraph.orchestrator.langgraph_workflow import (
    node_apply_acceptance_gate,
    node_check_coverage,
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
            content_summary="k-Means clustering algorithm.",
        )
    )
    state.add_section(
        DocSection(
            section_id="sec_pca",
            title="PCA",
            slide_range=(20, 25),
            importance_weight=0.8,
            content_summary="Principal component analysis.",
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
    state.add_learning_objective(
        LearningObjective(
            lo_id="lo_pca_variance",
            section_id="sec_pca",
            objective_text="Explain PCA variance preservation.",
            difficulty_level="intermediate",
        )
    )
    for chunk_id, content, page in (
        ("chunk_kmeans", "k-means assigns each point to the nearest centroid, then updates centroids.", 8),
        ("chunk_pca", "PCA finds orthogonal directions that preserve maximum variance.", 22),
    ):
        state.add_evidence_chunk(
            EvidenceChunk(
                chunk_id=chunk_id,
                content=content,
                metadata={"page_number": page, "pdf_id": "141", "source_file": "141.pdf"},
                quality_score=0.9,
            )
        )
    for atom_id, content, section_id, lo_id, chunk_id in (
        (
            "atom_kmeans",
            "k-means alternates assignment and centroid recomputation.",
            "sec_kmeans",
            "lo_kmeans_steps",
            "chunk_kmeans",
        ),
        (
            "atom_pca",
            "PCA projects data onto principal components.",
            "sec_pca",
            "lo_pca_variance",
            "chunk_pca",
        ),
    ):
        state.add_knowledge_atom(
            KnowledgeAtom(
                atom_id=atom_id,
                content=content,
                atom_type="procedure",
                citations=[{"chunk_id": chunk_id, "page": 8}],
                section_id=section_id,
                lo_id=lo_id,
            )
        )
        state.link_chunk_to_atom(chunk_id, atom_id)
    return state


def _supported_mcq(state: CooperativeState, blueprint: MCQBlueprint) -> CandidateMCQ:
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
        status="evidence_verified",
    )
    state.store_mcq_blueprints([blueprint])
    state.store_candidate_mcq(mcq)
    report = verify_candidate_against_evidence(mcq, state, llm=None)
    state.store_evidence_report(report)
    cove = run_cross_cove_for_candidate(mcq, verifier_llm=None)
    state.store_cross_cove_report(cove)
    return mcq


class TestMCQCoverageStep5(unittest.TestCase):
    def test_insufficient_atoms_not_accepted(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        blueprint.target_atom_ids = []
        mcq = CandidateMCQ(
            q_id="q_no_atoms",
            blueprint_id=blueprint.blueprint_id,
            stem="k-means question",
            options=[
                AnswerOption(option_id="A", text="Nearest centroid", is_correct=True),
                AnswerOption(option_id="B", text="Random", is_correct=False),
                AnswerOption(option_id="C", text="Label", is_correct=False),
                AnswerOption(option_id="D", text="Outlier", is_correct=False),
            ],
            correct_option_id="A",
            answer_explanation="Uses centroid assignment.",
            source_refs=[SourceRef(doc_id="141", page=8, chunk_id="chunk_kmeans")],
            flags=["insufficient_atoms"],
        )
        state.store_mcq_blueprints([blueprint])
        state.store_candidate_mcq(mcq)
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
        decision = apply_acceptance_gate(mcq, state)
        self.assertNotEqual(decision.decision, "accept")
        self.assertEqual(mcq.status, "needs_revision")

    def test_supported_evidence_report_accepted(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        mcq = _supported_mcq(state, blueprint)
        decision = apply_acceptance_gate(mcq, state)
        self.assertEqual(decision.decision, "accept")
        self.assertEqual(mcq.status, "accepted")

    def test_missing_source_refs_needs_revision(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        mcq = CandidateMCQ(
            q_id="q_no_refs",
            blueprint_id=blueprint.blueprint_id,
            stem="k-means assigns points to what?",
            options=[
                AnswerOption(option_id="A", text="Nearest centroid", is_correct=True),
                AnswerOption(option_id="B", text="Random", is_correct=False),
                AnswerOption(option_id="C", text="Label", is_correct=False),
                AnswerOption(option_id="D", text="Outlier", is_correct=False),
            ],
            correct_option_id="A",
            answer_explanation="Nearest centroid assignment.",
            source_refs=[],
        )
        state.store_mcq_blueprints([blueprint])
        state.store_candidate_mcq(mcq)
        state.store_evidence_report(
            EvidenceCheckReport(
                q_id=mcq.q_id,
                supported_by_pdf=True,
                unsupported_claims=[],
                source_refs=[],
                confidence=0.8,
                stem_supported=True,
                answer_supported=True,
                distractors_valid=True,
                explanation_supported=True,
            )
        )
        decision = apply_acceptance_gate(mcq, state)
        self.assertEqual(decision.decision, "revise")
        self.assertEqual(mcq.status, "needs_revision")

    def test_compute_mcq_coverage_counts_only_accepted(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        accepted = _supported_mcq(state, blueprint)
        apply_acceptance_gate(accepted, state)

        rejected = CandidateMCQ(
            q_id="q_rejected",
            blueprint_id=blueprint.blueprint_id,
            stem="bad",
            options=[AnswerOption(option_id="A", text="x", is_correct=True)],
            correct_option_id="A",
            answer_explanation="",
            flags=["too_few_options"],
        )
        state.store_candidate_mcq(rejected)
        apply_acceptance_gate(rejected, state)

        coverage = compute_mcq_coverage(state)
        self.assertEqual(coverage["accepted_count"], 1)
        self.assertGreaterEqual(coverage["rejected_count"] + coverage["needs_revision_count"], 1)

    def test_find_coverage_gaps_detects_uncovered_sections(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        if blueprint.section_id != "sec_kmeans":
            blueprint = next(
                bp for bp in create_mcq_blueprints_from_coverage(state)
                if bp.section_id == "sec_kmeans"
            )
        mcq = _supported_mcq(state, blueprint)
        apply_acceptance_gate(mcq, state)

        gaps = find_coverage_gaps(state)
        uncovered_sections = {g["section_id"] for g in gaps}
        self.assertIn("sec_pca", uncovered_sections)

    def test_repair_blueprints_never_use_general(self):
        state = _fake_state()
        gaps = find_coverage_gaps(state)
        repair = create_repair_blueprints(state, gaps)
        self.assertGreaterEqual(len(repair), 1)
        for bp in repair:
            self.assertNotEqual(bp.section_id.lower(), "general")
            self.assertNotEqual(bp.lo_id.lower(), "general")
            self.assertNotIn("general", bp.generation_goal.lower())
            self.assertEqual(bp.status, "planned")

    def test_packager_exports_only_accepted(self):
        state = _fake_state()
        blueprints = create_mcq_blueprints_from_coverage(state)
        state.store_mcq_blueprints(blueprints)
        accepted_bp = blueprints[0]
        accepted = _supported_mcq(state, accepted_bp)
        apply_acceptance_gate(accepted, state)

        draft_bp = blueprints[1] if len(blueprints) > 1 else blueprints[0]
        draft = generate_candidate_from_blueprint(draft_bp, state, llm=None)
        draft.status = "needs_revision"
        state.store_candidate_mcq(draft)

        pack = package_final_quiz(state, title="Test Quiz")
        self.assertEqual(len(pack.questions), 1)
        exported = export_mcqs_for_external_iwf_check(state)
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0]["status"], "accepted")

    def test_langgraph_acceptance_and_coverage_nodes(self):
        state = _fake_state()
        blueprints = create_mcq_blueprints_from_coverage(state)
        state.store_mcq_blueprints(blueprints)
        for bp in blueprints[:1]:
            mcq = generate_candidate_from_blueprint(bp, state, llm=None)
            register_candidate_mcq(state, mcq, bp)
            report = verify_candidate_against_evidence(mcq, state, llm=None)
            state.store_evidence_report(report)

        graph_state = {
            "cooperative_state": state,
            "step_logs": [],
            "error": None,
            "repair_iterations": 0,
            "coverage_gaps": [],
        }
        runtime = workflow_runtime_context(
            agent_runner=type("StubRunner", (), {})(),
            orchestrator_config=DEFAULT_ORCHESTRATOR_CONFIG,
        )
        node_apply_acceptance_gate(graph_state, runtime)
        update = node_check_coverage(graph_state, runtime)
        self.assertIn("coverage_gaps", update)
        self.assertTrue(update["cooperative_state"].mcq_coverage_summary)


if __name__ == "__main__":
    unittest.main()
