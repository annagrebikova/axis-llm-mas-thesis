"""Smoke tests for Step 3 blueprint-driven MCQ generation."""

from __future__ import annotations

import json
import unittest

from mas_langgraph.core.entities import DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
from mas_langgraph.core.state import CooperativeState
from mas_langgraph.domains.quiz.blueprint_planner import create_mcq_blueprints_from_coverage
from mas_langgraph.domains.quiz.generation import (
    build_generation_context_from_blueprint,
    generate_candidate_from_blueprint,
    validate_candidate_mcq_structure,
)
from mas_langgraph.orchestrator.langgraph_workflow import node_generate_candidate_mcqs_from_blueprints
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


class TestMCQBlueprintStep3(unittest.TestCase):
    def test_build_generation_context_from_blueprint(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        context = build_generation_context_from_blueprint(blueprint, state)
        self.assertEqual(context["section_title"], "k-Means")
        self.assertEqual(context["lo_id"], "lo_kmeans_steps")
        self.assertEqual(len(context["atoms"]), 1)
        self.assertEqual(len(context["evidence_chunks"]), 1)
        self.assertTrue(context["source_refs"])
        blob = f"{context['section_title']} {context['learning_objective']} {context['question_type']}"
        self.assertNotIn("general", blob.lower())

    def test_generate_candidate_does_not_use_general(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        mcq = generate_candidate_from_blueprint(blueprint, state, llm=None)
        payload = json.dumps({
            "stem": mcq.stem,
            "options": [o.text for o in mcq.options],
            "flags": mcq.flags,
        }).lower()
        self.assertNotIn("general", payload)

    def test_generated_mcq_has_four_options_and_one_correct(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        mcq = generate_candidate_from_blueprint(blueprint, state, llm=None)
        self.assertEqual(len(mcq.options), 4)
        self.assertEqual(sum(1 for o in mcq.options if o.is_correct), 1)

    def test_generated_mcq_has_source_refs(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        mcq = generate_candidate_from_blueprint(blueprint, state, llm=None)
        self.assertTrue(mcq.source_refs)
        self.assertTrue(any(ref.chunk_id for ref in mcq.source_refs))

    def test_no_evidence_gets_insufficient_evidence_flag(self):
        state = _fake_state()
        blueprint = create_mcq_blueprints_from_coverage(state)[0]
        blueprint.evidence_chunk_ids = []
        mcq = generate_candidate_from_blueprint(blueprint, state, llm=None)
        self.assertIn("insufficient_evidence", mcq.flags)

    def test_langgraph_node_stores_candidates_in_state(self):
        state = _fake_state()
        blueprints = create_mcq_blueprints_from_coverage(state)
        state.store_mcq_blueprints(blueprints)

        class StubRunner:
            _quiz_generation_rounds = 0

        graph_state = {
            "cooperative_state": state,
            "step_logs": [],
            "error": None,
        }
        runtime = workflow_runtime_context(agent_runner=StubRunner())
        update = node_generate_candidate_mcqs_from_blueprints(graph_state, runtime)
        self.assertIn("cooperative_state", update)
        self.assertIsNot(update["cooperative_state"], state)
        self.assertGreaterEqual(len(update["cooperative_state"].candidate_mcqs), 1)
        self.assertGreaterEqual(len(update["cooperative_state"].candidate_questions), 1)

    def test_validate_candidate_mcq_structure_flags(self):
        from mas_langgraph.domains.quiz.entities import AnswerOption, CandidateMCQ

        candidate = CandidateMCQ(
            q_id="q_bad",
            blueprint_id="bp_bad",
            stem="",
            options=[
                AnswerOption(option_id="A", text="", is_correct=True),
            ],
            correct_option_id="A",
            answer_explanation="",
            source_refs=[],
        )
        validated = validate_candidate_mcq_structure(candidate)
        self.assertIn("empty_stem", validated.flags)
        self.assertIn("too_few_options", validated.flags)
        self.assertIn("missing_source_refs", validated.flags)


if __name__ == "__main__":
    unittest.main()
