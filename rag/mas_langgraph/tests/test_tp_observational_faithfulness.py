"""Trace+Provenance observational faithfulness + repair-flow regression tests."""

from __future__ import annotations

import unittest
from typing import List

from mas_langgraph.experiment.faithfulness import (
    provenance_only_role_boundary,
    tp_has_unsupported_causal_role_claim,
    validate_explanation_faithfulness,
)
from mas_langgraph.experiment.lay_facts import LayExplanationFacts, build_lay_explanation_facts
from mas_langgraph.experiment.service import _synthesize_condition
from mas_langgraph.experiment.verbalizer import ExplanationEvidenceView, build_evidence_view


def _tp_facts() -> LayExplanationFacts:
    return build_lay_explanation_facts(
        condition="trace_provenance",
        raw_evidence={
            "factual_decision": "accept",
            "question_stem": (
                "What is the primary goal of unsupervised learning in data analysis?"
            ),
            "correct_answer": (
                "To understand the structure of the data without labeled outputs."
            ),
            "execution_trace": [
                "build_outline#1",
                "generate_learning_objectives#1",
                "plan_mcq_blueprints#1",
                "generate_candidate_mcqs_from_blueprints#1",
                "verify_candidate_mcqs_against_evidence#1",
                "apply_acceptance_gate#1",
            ],
            "relevant_artifacts": [
                {
                    "ref": "lo_1:v1",
                    "artifact_type": "LearningObjective",
                    "objective_text": (
                        "Define unsupervised learning and describe its purpose"
                    ),
                },
                {
                    "ref": "atom_1:v1",
                    "artifact_type": "KnowledgeAtom",
                    "content_preview": (
                        "Unsupervised learning tries to understand the structure "
                        "of data without labeled outputs."
                    ),
                },
            ],
            "provenance_relations": [
                {"relation": "planning_atom", "artifact_ref": "atom_1:v1"},
                {"relation": "knowledge_atom", "artifact_ref": "atom_1:v1"},
                {"relation": "grounding_atom", "artifact_ref": "atom_1:v1"},
            ],
        },
    )


def _tp_view() -> ExplanationEvidenceView:
    facts = _tp_facts()
    return ExplanationEvidenceView(
        factual_decision=facts.decision,
        observed_process=facts.educational_purpose,
        recorded_dependencies=facts.source_basis,
        limitations=facts.limitations,
        lay_facts=facts,
    )


def _axis_role_view() -> ExplanationEvidenceView:
    facts = LayExplanationFacts(
        decision="ACCEPT",
        educational_purpose=("The system generated and checked the question.",),
        trace_narrative=("The system generated and checked the question.",),
        relevant_provenance_facts=(
            "The records link the question to unsupervised learning.",
        ),
        counterfactual_component_findings=(
            "This information helped keep the question focused on the general "
            "purpose of unsupervised learning.",
        ),
        question_content_findings=(
            "This information helped keep the question focused on the general "
            "purpose of unsupervised learning.",
        ),
        specific_source_known=True,
        recorded_use_known=True,
        information_role_known=True,
        counterfactual_effect_known=True,
        decision_effect_known=True,
        information_role="focus_shaping",
        information_role_supported=True,
        primary_information_description=(
            "Unsupervised learning tries to understand the structure of data "
            "without labeled outputs."
        ),
        primary_information_role_target="atom_1:v1",
        question_identity_effect=True,
        question_identity_effect_known=True,
        acceptance_effect=False,
        acceptance_effect_known=True,
    )
    return ExplanationEvidenceView(
        factual_decision="ACCEPT",
        tested_interventions=facts.question_content_findings,
        lay_facts=facts,
    )


class TestTPObservationalFaithfulness(unittest.TestCase):
    def test_a_observational_provenance_passes(self):
        text = (
            "The records link the question to the teaching fact that unsupervised "
            "learning tries to understand the structure of data without labeled "
            "outputs. This information was available during planning, generation, "
            "and checking."
        )
        ok, warns = validate_explanation_faithfulness(
            condition="trace_provenance",
            explanation=text,
            evidence_view=_tp_view(),
        )
        self.assertTrue(ok, warns)
        self.assertFalse(warns)

    def test_b_explicit_limitation_passes(self):
        text = (
            "The records show that this information was used, but they do not "
            "establish whether it influenced the question."
        )
        ok, warns = validate_explanation_faithfulness(
            condition="trace_provenance",
            explanation=text,
            evidence_view=_tp_view(),
        )
        self.assertTrue(ok, warns)

    def test_c_influenced_positive_claim_fails(self):
        text = "This information influenced which question was produced."
        ok, warns = validate_explanation_faithfulness(
            condition="trace_provenance",
            explanation=text,
            evidence_view=_tp_view(),
        )
        self.assertFalse(ok)
        self.assertTrue(
            any("unsupported causal information-role" in w for w in warns),
            warns,
        )

    def test_d_shaped_fails(self):
        text = "The teaching facts shaped the incorrect answer options."
        ok, warns = validate_explanation_faithfulness(
            condition="trace_provenance",
            explanation=text,
            evidence_view=_tp_view(),
        )
        self.assertFalse(ok, warns)

    def test_e_helped_determine_fails(self):
        text = "This fact helped determine the correct answer."
        ok, warns = validate_explanation_faithfulness(
            condition="trace_provenance",
            explanation=text,
            evidence_view=_tp_view(),
        )
        self.assertFalse(ok, warns)

    def test_f_anchored_fails(self):
        text = "This information anchored the expected answer."
        ok, warns = validate_explanation_faithfulness(
            condition="trace_provenance",
            explanation=text,
            evidence_view=_tp_view(),
        )
        self.assertFalse(ok, warns)

    def test_g_clause_negation_does_not_hide_later_overclaim(self):
        text = (
            "The records do not show whether the information influenced planning, "
            "but it shaped the final answer options."
        )
        self.assertTrue(tp_has_unsupported_causal_role_claim(text))
        ok, warns = validate_explanation_faithfulness(
            condition="trace_provenance",
            explanation=text,
            evidence_view=_tp_view(),
        )
        self.assertFalse(ok, warns)

    def test_h_legitimate_verification_wording_passes(self):
        text = (
            "The question was checked against the teaching material and was "
            "recorded as supported by it."
        )
        ok, warns = validate_explanation_faithfulness(
            condition="trace_provenance",
            explanation=text,
            evidence_view=_tp_view(),
        )
        self.assertTrue(ok, warns)

    def test_i_axis_may_use_supported_causal_language(self):
        text = (
            "This information helped keep the question focused on the general "
            "purpose of unsupervised learning."
        )
        ok, warns = validate_explanation_faithfulness(
            condition="axis",
            explanation=text,
            evidence_view=_axis_role_view(),
        )
        self.assertTrue(ok, warns)
        self.assertFalse(
            any("unsupported causal information-role" in w for w in warns),
            warns,
        )

    def test_capability_boundary_helper(self):
        facts = _tp_facts()
        self.assertTrue(facts.recorded_use_known)
        self.assertFalse(facts.information_role_known)
        self.assertTrue(provenance_only_role_boundary(facts))
        axis_facts = _axis_role_view().lay_facts
        self.assertFalse(provenance_only_role_boundary(axis_facts))


class TestSynthesisRepairFlow(unittest.TestCase):
    def test_repair_valid_avoids_fallback(self):
        calls: List[str] = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if "Rewrite the explanation" in prompt or "failed faithfulness" in prompt:
                return (
                    "The records show that this information was used when the "
                    "question was planned and generated, but they do not establish "
                    "which part of the MCQ depended on it. The teaching fact that "
                    "unsupervised learning tries to understand the structure of "
                    "data without labeled outputs was available during checking."
                )
            return "This information influenced which question was produced."

        text, view, warns, diag = _synthesize_condition(
            condition="trace_provenance",
            raw_evidence={
                "factual_decision": "accept",
                "question_stem": (
                    "What is the primary goal of unsupervised learning?"
                ),
                "correct_answer": (
                    "To understand the structure of the data without labeled outputs."
                ),
                "execution_trace": [
                    "plan_mcq_blueprints#1",
                    "generate_candidate_mcqs_from_blueprints#1",
                ],
                "relevant_artifacts": [
                    {
                        "ref": "atom_1:v1",
                        "artifact_type": "KnowledgeAtom",
                        "content_preview": (
                            "Unsupervised learning tries to understand the "
                            "structure of data without labeled outputs."
                        ),
                    }
                ],
                "provenance_relations": [
                    {"relation": "planning_atom", "artifact_ref": "atom_1:v1"},
                    {"relation": "knowledge_atom", "artifact_ref": "atom_1:v1"},
                ],
            },
            question="Why was this question selected?",
            factual_decision="accept",
            question_stem="What is the primary goal of unsupervised learning?",
            llm_generate=llm,
        )
        self.assertEqual(len(calls), 2)
        self.assertTrue(diag["synthesis_repair_attempted"])
        self.assertTrue(diag["synthesis_repair_valid"])
        self.assertFalse(diag["synthesis_used_fallback"])
        self.assertFalse(diag["synthesis_initial_valid"])
        self.assertIn("structure", text.lower())
        self.assertNotIn("influenced which question", text.lower())
        self.assertTrue(view.lay_facts.recorded_use_known)
        self.assertFalse(any("mechanical_fallback" in w for w in warns))

    def test_invalid_repair_uses_mechanical_fallback(self):
        def llm(prompt: str) -> str:
            # Always return causal overclaim (initial + repair).
            return "This information influenced which question was produced."

        text, _view, warns, diag = _synthesize_condition(
            condition="trace_provenance",
            raw_evidence={
                "factual_decision": "accept",
                "question_stem": "What is unsupervised learning?",
                "correct_answer": "Understanding unlabeled structure.",
                "execution_trace": ["generate_candidate_mcqs_from_blueprints#1"],
                "relevant_artifacts": [
                    {
                        "ref": "atom_1:v1",
                        "artifact_type": "KnowledgeAtom",
                        "content_preview": (
                            "Unsupervised learning tries to understand the "
                            "structure of data without labeled outputs."
                        ),
                    }
                ],
                "provenance_relations": [
                    {"relation": "knowledge_atom", "artifact_ref": "atom_1:v1"},
                ],
            },
            question="Why was this question selected?",
            factual_decision="accept",
            question_stem="What is unsupervised learning?",
            llm_generate=llm,
        )
        self.assertTrue(diag["synthesis_repair_attempted"])
        self.assertFalse(diag["synthesis_repair_valid"])
        self.assertTrue(diag["synthesis_used_fallback"])
        self.assertTrue(
            any("mechanical_fallback" in w for w in warns),
            warns,
        )
        self.assertNotIn("influenced which question", text.lower())
        self.assertIn("structure", text.lower())


if __name__ == "__main__":
    unittest.main()
