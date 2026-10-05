"""Smoke tests for Step 8 full-outline evidence coverage and repair retrieval."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from mas_langgraph.core.entities import DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
from mas_langgraph.core.state import CooperativeState
from mas_langgraph.domains.quiz.blueprint_planner import create_mcq_blueprints_from_coverage
from mas_langgraph.domains.quiz.coverage import (
    build_coverage_requirements,
    compute_mcq_coverage,
    create_repair_blueprints,
    find_coverage_gaps,
)
from mas_langgraph.domains.quiz.evidence_repair import (
    build_section_retrieval_queries,
    repair_missing_evidence_for_sections,
)
from mas_langgraph.domains.quiz.packaging import package_final_quiz
from mas_langgraph.domains.quiz.section_evidence import (
    get_sections_without_evidence,
    update_section_evidence_status,
)
from mas_langgraph.orchestrator.langgraph_workflow import route_after_coverage
from mas_langgraph.orchestrator.runtime_helpers import workflow_runtime_context


def _outline_sections(n: int = 10) -> list[DocSection]:
    titles = [
        "Introduction",
        "Clustering Overview",
        "k-Means",
        "Density Estimation",
        "GMM",
        "EM Algorithm",
        "PCA",
        "ISOMAP",
        "Eigenfaces",
        "Summary",
    ]
    sections: list[DocSection] = []
    for i in range(n):
        sections.append(
            DocSection(
                section_id=f"sec_{i + 1}",
                title=titles[i] if i < len(titles) else f"Section {i + 1}",
                slide_range=(i * 5 + 1, i * 5 + 5),
                importance_weight=1.0 - i * 0.05,
                content_summary=f"Content for {titles[i] if i < len(titles) else i}.",
            )
        )
    return sections


def _state_with_partial_evidence() -> CooperativeState:
    state = CooperativeState()
    for section in _outline_sections(10):
        state.add_section(section)
        state.add_learning_objective(
            LearningObjective(
                lo_id=f"lo_{section.section_id}",
                section_id=section.section_id,
                objective_text=f"Explain {section.title}.",
                difficulty_level="intermediate",
            )
        )

    for section_id, chunk_id, content in (
        ("sec_1", "chunk_1", "Introduction to unsupervised learning."),
        ("sec_2", "chunk_2", "Clustering groups similar data points."),
        ("sec_3", "chunk_3", "k-means alternates assignment and centroid updates."),
    ):
        state.add_evidence_chunk(
            EvidenceChunk(
                chunk_id=chunk_id,
                content=content,
                metadata={"section_id": section_id, "page_number": 1},
                quality_score=0.9,
            )
        )
        state.add_knowledge_atom(
            KnowledgeAtom(
                atom_id=f"atom_{chunk_id}",
                content=content,
                atom_type="definition",
                citations=[{"chunk_id": chunk_id, "page": 1}],
                section_id=section_id,
                lo_id=f"lo_{section_id}",
            )
        )
        state.link_chunk_to_atom(chunk_id, f"atom_{chunk_id}")

    state.update_section_evidence_status()
    return state


class TestMCQStep8EvidenceCoverage(unittest.TestCase):
    def test_coverage_requirements_use_full_outline(self):
        state = _state_with_partial_evidence()
        requirements = build_coverage_requirements(state)
        self.assertEqual(requirements["outline_sections_total"], 10)
        self.assertEqual(len(requirements["sections"]), 10)
        self.assertLess(len(requirements["sections_requiring_mcq"]), 10)

    def test_coverage_sections_total_equals_outline_length(self):
        state = _state_with_partial_evidence()
        coverage = compute_mcq_coverage(state)
        self.assertEqual(coverage["coverage_sections_total"], 10)
        self.assertGreater(len(coverage["uncovered_due_to_missing_evidence"]), 0)

    def test_section_without_evidence_is_marked_missing(self):
        state = _state_with_partial_evidence()
        missing = get_sections_without_evidence(state)
        missing_ids = {s.section_id for s in missing}
        self.assertIn("sec_4", missing_ids)
        self.assertIn("sec_7", missing_ids)
        self.assertNotIn("sec_3", missing_ids)

    def test_retrieval_repair_creates_queries_for_missing_sections(self):
        queries = build_section_retrieval_queries("k-Means")
        self.assertTrue(any("algorithm steps" in q for q in queries))
        self.assertTrue(any("k-Means" in q for q in queries))
        self.assertEqual(build_section_retrieval_queries("general"), [])

    def test_repair_retrieval_tags_new_chunks(self):
        state = _state_with_partial_evidence()
        rag = MagicMock()
        rag.state = state

        def fake_search(query, k=5):
            return [
                (
                    MagicMock(
                        page_content=f"Retrieved content for {query}",
                        metadata={"page_number": 99},
                    ),
                    0.95,
                )
            ]

        rag.rag_pipeline.search.side_effect = fake_search
        rag.log_step = MagicMock()
        rag.detect_duplicate_chunks.return_value = ""

        result = repair_missing_evidence_for_sections(state, rag, set())
        self.assertGreater(result["chunks_added"], 0)
        status = update_section_evidence_status(state).get("sec_4")
        self.assertIsNotNone(status)
        self.assertGreater(len(status.evidence_chunk_ids), 0)

    def test_no_blueprint_with_empty_evidence(self):
        state = _state_with_partial_evidence()
        blueprints = create_mcq_blueprints_from_coverage(state)
        self.assertLessEqual(len(blueprints), 3)
        for bp in blueprints:
            self.assertTrue(bp.evidence_chunk_ids)
            self.assertTrue(bp.target_atom_ids)
            self.assertNotEqual(bp.section_id, "general")

    def test_repair_blueprints_created_when_section_has_evidence_and_atoms(self):
        state = _state_with_partial_evidence()
        state.add_evidence_chunk(
            EvidenceChunk(
                chunk_id="chunk_kmeans_extra",
                content="k-means uses Euclidean distance for assignment.",
                metadata={"section_id": "sec_3", "page_number": 12},
                quality_score=0.88,
            )
        )
        state.add_knowledge_atom(
            KnowledgeAtom(
                atom_id="atom_kmeans_extra",
                content="k-means minimizes within-cluster variance.",
                atom_type="procedure",
                citations=[{"chunk_id": "chunk_kmeans_extra", "page": 12}],
                section_id="sec_3",
                lo_id="lo_sec_3",
            )
        )
        state.link_chunk_to_atom("chunk_kmeans_extra", "atom_kmeans_extra")
        state.update_section_evidence_status()

        gaps = [
            {
                "section_id": "sec_3",
                "lo_id": "lo_sec_3",
                "section_title": "k-Means",
                "reason": "no accepted MCQ for section",
                "suggested_question_type": "basic_definition",
                "suggested_difficulty": "intermediate",
            }
        ]
        repair_bps = create_repair_blueprints(state, gaps)
        self.assertGreaterEqual(len(repair_bps), 1)
        self.assertEqual(repair_bps[0].section_id, "sec_3")

    def test_repair_blueprints_skipped_without_evidence(self):
        state = _state_with_partial_evidence()
        gaps = [
            {
                "section_id": "sec_7",
                "lo_id": "lo_sec_7",
                "section_title": "PCA",
                "reason": "no accepted MCQ for section",
            }
        ]
        repair_bps = create_repair_blueprints(state, gaps)
        self.assertEqual(len(repair_bps), 0)

    def test_route_after_coverage_prefers_evidence_repair(self):
        state_obj = _state_with_partial_evidence()
        graph_state = {
            "cooperative_state": state_obj,
            "evidence_repair_iterations": 0,
            "max_evidence_repair_iterations": 2,
            "blueprint_repair_iterations": 0,
            "max_blueprint_repair_iterations": 2,
            "coverage_gaps": [{"section_id": "sec_3"}],
        }
        self.assertEqual(
            route_after_coverage(graph_state, workflow_runtime_context()),
            "repair_missing_evidence",
        )

    def test_final_package_includes_only_accepted_mcqs(self):
        state = _state_with_partial_evidence()
        blueprints = create_mcq_blueprints_from_coverage(state)
        self.assertTrue(blueprints)
        bp = blueprints[0]
        from mas_langgraph.domains.quiz.entities import AnswerOption, CandidateMCQ, SourceRef

        accepted = CandidateMCQ(
            q_id="q_accepted",
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
            source_refs=[SourceRef(doc_id="141", page=1, chunk_id=bp.evidence_chunk_ids[0])],
            status="accepted",
        )
        rejected = CandidateMCQ(
            q_id="q_rejected",
            blueprint_id=bp.blueprint_id,
            stem="Rejected question",
            options=[
                AnswerOption(option_id="A", text="A", is_correct=True),
                AnswerOption(option_id="B", text="B", is_correct=False),
                AnswerOption(option_id="C", text="C", is_correct=False),
                AnswerOption(option_id="D", text="D", is_correct=False),
            ],
            correct_option_id="A",
            answer_explanation="Rejected.",
            source_refs=[],
            status="rejected",
        )
        state.store_mcq_blueprints([bp])
        state.store_candidate_mcq(accepted)
        state.store_candidate_mcq(rejected)

        pack = package_final_quiz(state, title="Test Quiz")
        self.assertEqual(len(pack.questions), 1)
        self.assertEqual(pack.coverage_summary["outline_sections_total"], 10)
        self.assertEqual(pack.coverage_summary["accepted"], 1)
        self.assertEqual(pack.coverage_summary["rejected"], 1)


if __name__ == "__main__":
    unittest.main()
