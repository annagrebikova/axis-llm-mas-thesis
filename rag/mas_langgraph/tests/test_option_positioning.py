"""Deterministic correct-answer option positioning (A–D cycle by q_id)."""

from __future__ import annotations

import unittest

from mas_langgraph.core.entities import DocSection, EvidenceChunk, KnowledgeAtom, LearningObjective
from mas_langgraph.core.state import CooperativeState
from mas_langgraph.core.state_cloning import clone_cooperative_state
from mas_langgraph.domains.quiz.blueprint_planner import create_mcq_blueprints_from_coverage
from mas_langgraph.domains.quiz.entities import AnswerOption, CandidateMCQ, SourceRef
from mas_langgraph.domains.quiz.generation import (
    assemble_candidate_mcq,
    build_generation_context_from_blueprint,
    generate_candidate_from_blueprint,
)
from mas_langgraph.domains.quiz.option_positioning import (
    assign_deterministic_option_positions,
    target_option_label_for_qid,
)
from mas_langgraph.domains.quiz.revision import revise_candidate_mcq


def _options_with_correct_at(label: str, correct_text: str = "CORRECT") -> list[AnswerOption]:
    texts = {
        "A": "distractor-A",
        "B": "distractor-B",
        "C": "distractor-C",
        "D": "distractor-D",
    }
    texts[label] = correct_text
    return [
        AnswerOption(
            option_id=lid,
            text=texts[lid],
            is_correct=(lid == label),
            explanation=f"expl-{lid}",
            misconception=None if lid == label else f"misc-{lid}",
            source_refs=[SourceRef(doc_id="141", page=1, chunk_id="c1")] if lid == label else [],
        )
        for lid in ("A", "B", "C", "D")
    ]


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
        metadata={"page_number": 6, "pdf_id": "141"},
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


class TestTargetLabelMapping(unittest.TestCase):
    def test_sequential_qids_cycle_abcd(self):
        expected = {
            "q_00000001": "A",
            "q_00000002": "B",
            "q_00000003": "C",
            "q_00000004": "D",
            "q_00000005": "A",
        }
        for qid, label in expected.items():
            self.assertEqual(target_option_label_for_qid(qid), label)

    def test_same_qid_stable_across_calls(self):
        qid = "q_00000007"
        self.assertEqual(target_option_label_for_qid(qid), target_option_label_for_qid(qid))
        self.assertEqual(target_option_label_for_qid(qid), "C")


class TestAssignDeterministicPositions(unittest.TestCase):
    def test_preserves_correct_text_and_semantics(self):
        opts = _options_with_correct_at("A", correct_text="SEMANTIC KEY")
        new_opts, cid = assign_deterministic_option_positions(
            opts, stable_key="q_00000002", correct_option_id="A"
        )
        self.assertEqual(cid, "B")
        correct = [o for o in new_opts if o.is_correct]
        self.assertEqual(len(correct), 1)
        self.assertEqual(correct[0].text, "SEMANTIC KEY")
        self.assertEqual(correct[0].option_id, "B")
        self.assertEqual(correct[0].explanation, "expl-A")
        self.assertTrue(correct[0].source_refs)

    def test_correct_option_id_matches_is_correct(self):
        for qid, label in (
            ("q_00000001", "A"),
            ("q_00000002", "B"),
            ("q_00000003", "C"),
            ("q_00000004", "D"),
        ):
            opts = _options_with_correct_at("A")
            new_opts, cid = assign_deterministic_option_positions(
                opts, stable_key=qid, correct_option_id="A"
            )
            self.assertEqual(cid, label)
            self.assertEqual(sum(1 for o in new_opts if o.is_correct), 1)
            self.assertEqual(
                next(o.option_id for o in new_opts if o.is_correct),
                cid,
            )

    def test_llm_a_correct_moves_by_qid(self):
        opts = _options_with_correct_at("A", correct_text="RIGHT")
        _, cid = assign_deterministic_option_positions(
            opts, stable_key="q_00000003", correct_option_id="A"
        )
        self.assertEqual(cid, "C")

    def test_llm_correct_originally_c_not_semantically_changed(self):
        opts = _options_with_correct_at("C", correct_text="RIGHT-C")
        new_opts, cid = assign_deterministic_option_positions(
            opts, stable_key="q_00000004", correct_option_id="C"
        )
        self.assertEqual(cid, "D")
        self.assertEqual(next(o.text for o in new_opts if o.is_correct), "RIGHT-C")
        distractor_texts = {o.text for o in new_opts if not o.is_correct}
        self.assertEqual(distractor_texts, {"distractor-A", "distractor-B", "distractor-D"})

    def test_repeated_execution_same_position(self):
        opts = _options_with_correct_at("B", correct_text="KEY")
        a1, c1 = assign_deterministic_option_positions(
            opts, stable_key="q_00000002", correct_option_id="B"
        )
        a2, c2 = assign_deterministic_option_positions(
            opts, stable_key="q_00000002", correct_option_id="B"
        )
        self.assertEqual(c1, c2)
        self.assertEqual([o.text for o in a1], [o.text for o in a2])
        self.assertEqual([o.is_correct for o in a1], [o.is_correct for o in a2])


class TestGenerationIntegration(unittest.TestCase):
    def test_stub_generation_cycles_positions(self):
        state = _fake_state()
        blueprints = create_mcq_blueprints_from_coverage(state)
        # Force enough blueprints by reusing the first.
        bp = blueprints[0]
        positions = []
        for _ in range(5):
            mcq = generate_candidate_from_blueprint(bp, state, llm=None)
            positions.append(mcq.correct_option_id)
            correct = [o for o in mcq.options if o.is_correct]
            self.assertEqual(len(correct), 1)
            self.assertEqual(mcq.correct_option_id, correct[0].option_id)
        self.assertEqual(positions, ["A", "B", "C", "D", "A"])

    def test_parsed_llm_a_bias_normalized(self):
        state = _fake_state()
        bp = create_mcq_blueprints_from_coverage(state)[0]
        context = build_generation_context_from_blueprint(bp, state)
        # Consume q_00000001 so next assemble gets q_00000002 -> B
        state.generate_id("q")
        parsed = {
            "stem": "Which statement is supported?",
            "answer_explanation": "Because evidence.",
            "correct_option_id": "A",
            "options": [
                {
                    "option_id": "A",
                    "text": "RIGHT ANSWER",
                    "is_correct": True,
                    "explanation": "ok",
                    "misconception": None,
                },
                {
                    "option_id": "B",
                    "text": "wrong1",
                    "is_correct": False,
                    "explanation": "no",
                    "misconception": "m1",
                },
                {
                    "option_id": "C",
                    "text": "wrong2",
                    "is_correct": False,
                    "explanation": "no",
                    "misconception": "m2",
                },
                {
                    "option_id": "D",
                    "text": "wrong3",
                    "is_correct": False,
                    "explanation": "no",
                    "misconception": "m3",
                },
            ],
        }
        mcq = assemble_candidate_mcq(
            blueprint=bp, state=state, context=context, parsed=parsed
        )
        self.assertEqual(mcq.q_id, "q_00000002")
        self.assertEqual(mcq.correct_option_id, "B")
        self.assertEqual(next(o.text for o in mcq.options if o.is_correct), "RIGHT ANSWER")

    def test_parsed_correct_originally_d(self):
        state = _fake_state()
        bp = create_mcq_blueprints_from_coverage(state)[0]
        context = build_generation_context_from_blueprint(bp, state)
        parsed = {
            "stem": "Stem?",
            "answer_explanation": "Because.",
            "correct_option_id": "D",
            "options": [
                {"option_id": "A", "text": "w1", "is_correct": False, "explanation": "n", "misconception": "m"},
                {"option_id": "B", "text": "w2", "is_correct": False, "explanation": "n", "misconception": "m"},
                {"option_id": "C", "text": "w3", "is_correct": False, "explanation": "n", "misconception": "m"},
                {
                    "option_id": "D",
                    "text": "RIGHT-D",
                    "is_correct": True,
                    "explanation": "yes",
                    "misconception": None,
                },
            ],
        }
        mcq = assemble_candidate_mcq(
            blueprint=bp, state=state, context=context, parsed=parsed
        )
        self.assertEqual(mcq.q_id, "q_00000001")
        self.assertEqual(mcq.correct_option_id, "A")
        self.assertEqual(next(o.text for o in mcq.options if o.is_correct), "RIGHT-D")

    def test_no_silent_default_to_a_when_ambiguous(self):
        state = _fake_state()
        bp = create_mcq_blueprints_from_coverage(state)[0]
        context = build_generation_context_from_blueprint(bp, state)
        parsed = {
            "stem": "Stem?",
            "answer_explanation": "Because.",
            "options": [
                {"option_id": "A", "text": "w1", "is_correct": False},
                {"option_id": "B", "text": "w2", "is_correct": False},
                {"option_id": "C", "text": "w3", "is_correct": False},
                {"option_id": "D", "text": "w4", "is_correct": False},
            ],
        }
        mcq = assemble_candidate_mcq(
            blueprint=bp, state=state, context=context, parsed=parsed
        )
        self.assertIn("no_correct_option", mcq.flags)
        self.assertNotEqual(mcq.correct_option_id, "A")


class TestRevisionRetainsPosition(unittest.TestCase):
    def test_revision_keeps_same_correct_position_for_qid(self):
        state = _fake_state()
        bp = create_mcq_blueprints_from_coverage(state)[0]
        state.store_mcq_blueprints([bp])
        q_id = "q_00000002"  # -> B
        mcq = CandidateMCQ(
            q_id=q_id,
            blueprint_id=bp.blueprint_id,
            stem="Old stem about France?",
            options=_options_with_correct_at("B", correct_text="OLD CORRECT"),
            correct_option_id="B",
            answer_explanation="old",
            source_refs=[SourceRef(doc_id="141", page=6, chunk_id="chunk_kmeans_1")],
            status="needs_revision",
            flags=["evidence_not_supported"],
        )
        state.store_candidate_mcq(mcq)
        revised = revise_candidate_mcq(mcq, state, llm=None, max_revisions=2)
        self.assertEqual(revised.q_id, q_id)
        self.assertEqual(revised.correct_option_id, "B")
        self.assertEqual(sum(1 for o in revised.options if o.is_correct), 1)
        self.assertEqual(
            next(o.option_id for o in revised.options if o.is_correct),
            "B",
        )

        # Second revision still B
        revised.status = "needs_revision"
        revised.flags = list(revised.flags) + ["evidence_not_supported"]
        state.store_candidate_mcq(revised)
        revised2 = revise_candidate_mcq(revised, state, llm=None, max_revisions=2)
        self.assertEqual(revised2.correct_option_id, "B")


class TestFactualNoopSamePosition(unittest.TestCase):
    def test_cloned_state_generates_same_position(self):
        """Same checkpointed id counter → same q_id → same answer position (NOOP-like)."""
        state = _fake_state()
        bp = create_mcq_blueprints_from_coverage(state)[0]
        # Advance unrelated counters so q starts at a known value after clone.
        state._id_counters["q"] = 1  # next q id will be q_00000002 -> B

        factual_state = clone_cooperative_state(state)
        noop_state = clone_cooperative_state(state)

        factual_mcq = generate_candidate_from_blueprint(bp, factual_state, llm=None)
        noop_mcq = generate_candidate_from_blueprint(bp, noop_state, llm=None)

        self.assertEqual(factual_mcq.q_id, noop_mcq.q_id)
        self.assertEqual(factual_mcq.q_id, "q_00000002")
        self.assertEqual(factual_mcq.correct_option_id, noop_mcq.correct_option_id)
        self.assertEqual(factual_mcq.correct_option_id, "B")
        self.assertEqual(
            next(o.text for o in factual_mcq.options if o.is_correct),
            next(o.text for o in noop_mcq.options if o.is_correct),
        )


if __name__ == "__main__":
    unittest.main()
