"""Smoke tests for Step 2 MCQ blueprint model."""

from __future__ import annotations

import json
import unittest

from mas_langgraph.core.entities import CandidateQuestion, DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
from mas_langgraph.core.state import CooperativeState
from mas_langgraph.domains.quiz.adapters import (
    candidate_mcq_to_candidate_question,
    candidate_question_to_candidate_mcq,
)
from mas_langgraph.domains.quiz.blueprint_planner import create_mcq_blueprints_from_coverage
from mas_langgraph.domains.quiz.entities import MCQBlueprint
from mas_langgraph.domains.quiz.export import export_mcqs_for_external_iwf_check
from mas_langgraph.domains.quiz.state_integration import ensure_step2_mcq_state
from mas_langgraph.domains.quiz.generation import (
    generate_candidate_from_blueprint,
    register_candidate_mcq,
)


def _fake_state() -> CooperativeState:
    state = CooperativeState()
    state.add_section(
        DocSection(
            section_id="sec_clustering",
            title="Clustering",
            slide_range=(3, 8),
            importance_weight=0.9,
            content_summary="Clustering groups unlabeled data.",
        )
    )
    state.add_learning_objective(
        LearningObjective(
            lo_id="lo_kmeans",
            section_id="sec_clustering",
            objective_text="Explain k-means clustering steps.",
            difficulty_level="intermediate",
        )
    )
    chunk = EvidenceChunk(
        chunk_id="chunk_kmeans_1",
        content="k-means assigns points to nearest cluster centroids.",
        metadata={"page_number": 6, "pdf_id": "141", "source_file": "141.pdf"},
        quality_score=0.85,
    )
    state.add_evidence_chunk(chunk)
    atom = KnowledgeAtom(
        atom_id="atom_kmeans",
        content="k-means iterates assignment and centroid updates.",
        atom_type="procedure",
        citations=[{"chunk_id": "chunk_kmeans_1", "page": 6}],
        section_id="sec_clustering",
        lo_id="lo_kmeans",
    )
    state.add_knowledge_atom(atom)
    state.link_chunk_to_atom("chunk_kmeans_1", "atom_kmeans")
    return state


class TestMCQBlueprintStep2(unittest.TestCase):
    def test_blueprint_creation_from_fake_state(self):
        state = _fake_state()
        blueprints = create_mcq_blueprints_from_coverage(state)
        self.assertGreaterEqual(len(blueprints), 1)
        bp = blueprints[0]
        self.assertEqual(bp.section_id, "sec_clustering")
        self.assertEqual(bp.lo_id, "lo_kmeans")
        self.assertIn("chunk_kmeans_1", bp.evidence_chunk_ids)
        self.assertIn("atom_kmeans", bp.target_atom_ids)

    def test_no_blueprint_uses_general_topic(self):
        state = _fake_state()
        blueprints = create_mcq_blueprints_from_coverage(state)
        for bp in blueprints:
            self.assertNotEqual(bp.section_id.lower(), "general")
            self.assertNotEqual(bp.lo_id.lower(), "general")
            self.assertNotIn("general", bp.generation_goal.lower())

    def test_candidate_mcq_from_blueprint_has_one_correct_option(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        mcq = generate_candidate_from_blueprint(blueprint, state, llm=None)
        correct = [o for o in mcq.options if o.is_correct]
        self.assertEqual(len(mcq.options), 4)
        self.assertEqual(len(correct), 1)
        self.assertEqual(mcq.correct_option_id, correct[0].option_id)
        self.assertTrue(mcq.source_refs)

    def test_structural_validation_flags_missing_evidence(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        blueprint.evidence_chunk_ids = []
        mcq = generate_candidate_from_blueprint(blueprint, state, llm=None)
        self.assertIn("insufficient_evidence", mcq.flags)

    def test_external_iwf_export_serialization(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        mcq = generate_candidate_from_blueprint(blueprint, state, llm=None)
        register_candidate_mcq(state, mcq, blueprint)
        from mas_langgraph.domains.quiz.verification import verify_candidate_against_evidence
        from mas_langgraph.domains.quiz.cross_cove import run_cross_cove_for_candidate
        from mas_langgraph.domains.quiz.coverage import apply_acceptance_gate

        report = verify_candidate_against_evidence(mcq, state, llm=None)
        state.store_evidence_report(report)
        cove = run_cross_cove_for_candidate(mcq, verifier_llm=None)
        state.store_cross_cove_report(cove)
        apply_acceptance_gate(mcq, state)
        exported = export_mcqs_for_external_iwf_check(state, accepted_only=False)
        self.assertEqual(len(exported), 1)
        payload = exported[0]
        self.assertEqual(payload["q_id"], mcq.q_id)
        self.assertEqual(payload["section_id"], "sec_clustering")
        self.assertEqual(payload["lo_id"], "lo_kmeans")
        self.assertIn("distractor_explanations", payload)
        json.dumps(exported)

    def test_store_mcq_blueprints_on_state(self):
        state = CooperativeState()
        blueprints = [
            MCQBlueprint(
                blueprint_id="bp_one",
                section_id="sec_a",
                lo_id="lo_a",
                target_atom_ids=["atom_a"],
                evidence_chunk_ids=["chunk_a"],
                question_type="basic_definition",
                difficulty="beginner",
                bloom_level="Remembering",
                generation_goal="Test goal A",
            ),
            MCQBlueprint(
                blueprint_id="bp_two",
                section_id="sec_b",
                lo_id="lo_b",
                target_atom_ids=["atom_b"],
                evidence_chunk_ids=["chunk_b"],
                question_type="example_scenario",
                difficulty="intermediate",
                bloom_level="Applying",
                generation_goal="Test goal B",
            ),
        ]
        state.store_mcq_blueprints(blueprints)
        self.assertEqual(len(state.mcq_blueprints), 2)
        self.assertIn("bp_one", state.mcq_blueprints)
        self.assertIn("bp_two", state.mcq_blueprints)

    def test_store_mcq_blueprints_on_monolith_like_state(self):
        class MonolithLikeState:
            pass

        state = MonolithLikeState()
        ensure_step2_mcq_state(state)
        blueprints = [
            MCQBlueprint(
                blueprint_id="bp_x",
                section_id="sec_x",
                lo_id="lo_x",
                target_atom_ids=[],
                evidence_chunk_ids=["chunk_x"],
                question_type="key_characteristic",
                difficulty="intermediate",
                bloom_level="Understanding",
                generation_goal="Monolith integration",
            ),
            MCQBlueprint(
                blueprint_id="bp_y",
                section_id="sec_y",
                lo_id="lo_y",
                target_atom_ids=[],
                evidence_chunk_ids=["chunk_y"],
                question_type="difference_identification",
                difficulty="advanced",
                bloom_level="Analyzing",
                generation_goal="Monolith integration",
            ),
        ]
        state.store_mcq_blueprints(blueprints)
        self.assertEqual(len(state.mcq_blueprints), 2)

    def test_legacy_candidate_question_adapter_roundtrip(self):
        legacy = CandidateQuestion(
            q_id="q_legacy",
            question_text="What is k-means?",
            question_type="multiple_choice",
            options=["A) Clustering", "B) Regression", "C) Classification", "D) Ranking"],
            correct_answer="A) Clustering",
            difficulty="intermediate",
            topic_tags=["sec_clustering"],
            related_atoms=["atom_kmeans"],
            evidence_chunks=["chunk_kmeans_1"],
        )
        mcq = candidate_question_to_candidate_mcq(
            legacy,
            blueprint_id="bp_test",
            section_id="sec_clustering",
            lo_id="lo_kmeans",
        )
        roundtrip = candidate_mcq_to_candidate_question(
            mcq,
            section_id="sec_clustering",
            lo_id="lo_kmeans",
        )
        self.assertEqual(roundtrip.q_id, legacy.q_id)
        self.assertEqual(roundtrip.question_text, legacy.question_text)
        self.assertEqual(len(roundtrip.options or []), 4)


if __name__ == "__main__":
    unittest.main()
