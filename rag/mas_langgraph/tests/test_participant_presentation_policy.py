"""Participant presentation-policy lint and concise synthesis regressions."""

from __future__ import annotations

import unittest

from mas_langgraph.experiment.faithfulness import validate_explanation_faithfulness
from mas_langgraph.experiment.lay_facts import LayExplanationFacts
from mas_langgraph.experiment.presentation_policy import (
    count_presentation_sentences,
    count_presentation_words,
    has_detailed_workflow_chronology,
    mechanical_provenance_explanation,
    mechanical_trace_explanation,
    validate_participant_presentation_policy,
)
from mas_langgraph.experiment.service import _synthesize_condition
from mas_langgraph.experiment.synthesis import mechanical_synthesize_from_lay_facts
from mas_langgraph.experiment.verbalizer import ExplanationEvidenceView


LONG_TRACE_CHRONOLOGY = (
    "The system organized the teaching material into sections, created "
    "learning objectives, retrieved candidate supporting material, extracted "
    "concise factual statements from the material, and created a plan for the "
    "question; the MCQ and its answer options were then generated, checked "
    "against the teaching material and for coverage of the learning objectives, "
    "and retained for the final quiz."
)

COMPACT_TRACE = (
    "The question was generated, checked against the teaching material, and "
    "retained after the recorded checks, but the record does not identify which "
    "specific teaching fact determined the wording, correct answer, or distractors."
)

PREFERRED_TRACE = (
    "The record shows that the question was generated from the teaching "
    "material, but it does not identify which specific teaching fact determined "
    "the wording, correct answer, or distractors."
)


def _trace_facts() -> LayExplanationFacts:
    return LayExplanationFacts(
        decision="ACCEPT",
        trace_narrative=(LONG_TRACE_CHRONOLOGY,),
        limitations=(
            "The record does not identify which specific teaching fact determined "
            "the wording, correct answer, or distractors.",
        ),
        specific_source_known=False,
        recorded_use_known=False,
        information_role_known=False,
        counterfactual_effect_known=False,
        source_basis_unavailable=True,
    )


def _tp_facts(fact1: str, fact2: str = "") -> LayExplanationFacts:
    basis = (fact1,) + ((fact2,) if fact2 else ())
    return LayExplanationFacts(
        decision="ACCEPT",
        source_basis=basis,
        relevant_provenance_facts=tuple(
            f"The records link the question to the teaching fact that {f}."
            for f in basis
        ),
        specific_source_known=True,
        recorded_use_known=True,
        information_role_known=False,
        counterfactual_effect_known=False,
        selected_source_fact_count=len(basis),
    )


def _axis_facts(
    *,
    description: str,
    role_line: str,
    alt_line: str = "",
    role: str = "focus_shaping",
    stem_changed: bool = True,
    answer_changed: bool = False,
    distractors_changed: bool = False,
    decision_effect: bool = False,
) -> LayExplanationFacts:
    findings = (role_line,) + ((alt_line,) if alt_line else ())
    return LayExplanationFacts(
        decision="ACCEPT",
        counterfactual_component_findings=findings,
        question_content_findings=findings,
        specific_source_known=True,
        recorded_use_known=True,
        information_role_known=True,
        counterfactual_effect_known=True,
        decision_effect_known=True,
        decision_effect=decision_effect,
        information_role=role,
        information_role_supported=True,
        primary_information_description=description,
        stem_changed=stem_changed,
        correct_answer_changed=answer_changed,
        distractors_changed=distractors_changed,
    )


def _view(facts: LayExplanationFacts) -> ExplanationEvidenceView:
    return ExplanationEvidenceView(
        factual_decision=facts.decision,
        tested_interventions=facts.question_content_findings,
        lay_facts=facts,
        limitations=facts.limitations,
    )


class TestChronologyDetection(unittest.TestCase):
    def test_a_long_trace_chronology_rejected(self):
        self.assertTrue(has_detailed_workflow_chronology(LONG_TRACE_CHRONOLOGY))
        result = validate_participant_presentation_policy(
            LONG_TRACE_CHRONOLOGY, facts=_trace_facts()
        )
        self.assertFalse(result.valid)
        self.assertIn("workflow_chronology", result.violations)

    def test_b_compact_trace_summary_accepted(self):
        self.assertFalse(has_detailed_workflow_chronology(COMPACT_TRACE))
        result = validate_participant_presentation_policy(
            COMPACT_TRACE, facts=_trace_facts()
        )
        self.assertTrue(result.valid, result.violations)

    def test_preferred_limitation_sentence_accepted(self):
        result = validate_participant_presentation_policy(
            PREFERRED_TRACE, facts=_trace_facts()
        )
        self.assertTrue(result.valid, result.violations)
        self.assertLessEqual(result.word_count, 45)
        self.assertEqual(result.sentence_count, 1)


class TestProvenanceRedundancy(unittest.TestCase):
    def test_c_concrete_plus_triad_plus_disclaimer_rejected(self):
        text = (
            "The question was linked to teaching information stating that "
            "unsupervised learning tries to understand the structure of data "
            "without labeled outputs. This information was available during "
            "planning, generation, and checking. The records show what "
            "information was used, but not by themselves which parts of the "
            "question depended on each fact."
        )
        result = validate_participant_presentation_policy(
            text, facts=_tp_facts(
                "unsupervised learning tries to understand the structure of data "
                "without labeled outputs"
            )
        )
        self.assertFalse(result.valid)
        self.assertIn("redundant_provenance_boilerplate", result.violations)

    def test_observational_source_only_accepted(self):
        text = (
            "The question was linked to teaching information stating that "
            "unsupervised learning tries to understand the structure of data "
            "without labeled outputs."
        )
        result = validate_participant_presentation_policy(
            text, facts=_tp_facts(
                "unsupervised learning tries to understand the structure of data "
                "without labeled outputs"
            )
        )
        self.assertTrue(result.valid, result.violations)


class TestAxisChronologyTail(unittest.TestCase):
    def test_d_axis_chronology_tail_rejected(self):
        text = (
            "The question was linked to teaching information stating that "
            "Gaussian mixture models fit weighted Gaussians, and this information "
            "helped keep the question focused on mixture components. The workflow "
            "organized the material, generated the MCQ and answer options, checked "
            "it against the teaching material, and then evaluated whether it "
            "satisfied the required criteria."
        )
        facts = _axis_facts(
            description="Gaussian mixture models fit weighted Gaussians",
            role_line=(
                "The information stating Gaussian mixture models fit weighted "
                "Gaussians helped keep the selected question focused on mixture "
                "components."
            ),
        )
        result = validate_participant_presentation_policy(text, facts=facts)
        self.assertFalse(result.valid)
        self.assertIn("workflow_chronology", result.violations)


class TestNecessaryForReproductionWording(unittest.TestCase):
    def test_e_visible_content_unchanged_does_not_imply_mcq_change(self):
        facts = _axis_facts(
            description="Novelty detection identifies unusual user behavior",
            role_line=(
                'The information stating "Novelty detection identifies unusual '
                'user behavior" is important for reproducing this particular '
                "selected question. Without it, the remaining support did not "
                "yield a corresponding question with the same educational focus."
            ),
            role="necessary_for_reproduction",
            stem_changed=False,
            answer_changed=False,
            distractors_changed=False,
        )
        text = mechanical_synthesize_from_lay_facts(facts=facts)
        low = text.lower()
        self.assertIn("reproduce the recorded result", low)
        self.assertIn("remained unchanged", low)
        self.assertNotIn("same educational focus", low)
        self.assertNotIn("organized the material", low)
        ok, warns = validate_explanation_faithfulness(
            condition="axis", explanation=text, evidence_view=_view(facts)
        )
        self.assertTrue(ok, warns)
        pres = validate_participant_presentation_policy(text, facts=facts)
        self.assertTrue(pres.valid, pres.violations)


class TestMechanicalFallbackPolicy(unittest.TestCase):
    def test_trace_fallback_is_limitation_centered(self):
        text = mechanical_synthesize_from_lay_facts(facts=_trace_facts())
        self.assertEqual(text, mechanical_trace_explanation())
        self.assertFalse(has_detailed_workflow_chronology(text))
        self.assertEqual(count_presentation_sentences(text), 1)
        ok, warns = validate_explanation_faithfulness(
            condition="trace", explanation=text, evidence_view=_view(_trace_facts())
        )
        self.assertTrue(ok, warns)

    def test_tp_unsupervised_and_gmm_have_no_causal_or_chronology(self):
        cases = (
            (
                "unsupervised learning tries to understand the structure of data "
                "without labeled outputs",
                "clusters can be found without class labels",
            ),
            (
                "a Gaussian mixture model represents density as weighted Gaussians",
                "each component has its own mean and covariance",
            ),
        )
        for fact1, fact2 in cases:
            with self.subTest(fact1=fact1[:24]):
                facts = _tp_facts(fact1, fact2)
                text = mechanical_synthesize_from_lay_facts(facts=facts)
                self.assertIn(fact1.split()[0].lower(), text.lower())
                self.assertFalse(has_detailed_workflow_chronology(text))
                self.assertNotRegex(text.lower(), r"\binfluenced\b")
                self.assertNotRegex(text.lower(), r"\bshaped\b")
                self.assertNotIn(
                    "available during planning, generation, and checking",
                    text.lower(),
                )
                self.assertLessEqual(count_presentation_sentences(text), 2)
                self.assertLessEqual(count_presentation_words(text), 85)
                ok, warns = validate_explanation_faithfulness(
                    condition="trace_provenance",
                    explanation=text,
                    evidence_view=_view(facts),
                )
                self.assertTrue(ok, warns)
                pres = validate_participant_presentation_policy(text, facts=facts)
                self.assertTrue(pres.valid, pres.violations)

    def test_axis_pca_keeps_role_and_drops_chronology(self):
        facts = _axis_facts(
            description="PCA finds directions of maximum variance",
            role_line=(
                "The information stating PCA finds directions of maximum variance "
                "helped keep the selected question focused on linear projections."
            ),
            alt_line=(
                "With this support the question asks \"What does PCA maximize?\"; "
                "with the remaining support the available content instead favors "
                "\"Which method preserves geodesic distances, such as ISOMAP?\""
            ),
        )
        padded = facts.question_content_findings + (
            "The material was organized, objectives were created, supporting "
            "facts were retrieved and extracted, the MCQ and options were "
            "generated and checked against the teaching material, and the "
            "question was accepted.",
        )
        facts = LayExplanationFacts(
            **{
                **facts.__dict__,
                "question_content_findings": padded,
                "counterfactual_component_findings": padded,
            }
        )
        text = mechanical_synthesize_from_lay_facts(facts=facts)
        self.assertIn("pca", text.lower())
        self.assertIn("helped keep", text.lower())
        self.assertIn("isomap", text.lower())
        self.assertNotIn("organized", text.lower())
        self.assertNotIn("objectives were created", text.lower())
        self.assertGreaterEqual(count_presentation_sentences(text), 2)
        self.assertLessEqual(count_presentation_sentences(text), 4)
        ok, warns = validate_explanation_faithfulness(
            condition="axis", explanation=text, evidence_view=_view(facts)
        )
        self.assertTrue(ok, warns)
        pres = validate_participant_presentation_policy(text, facts=facts)
        self.assertTrue(pres.valid, pres.violations)


class TestRepairUsesExistingRound(unittest.TestCase):
    def test_faithful_but_verbose_trace_is_repaired_once(self):
        calls: list[str] = []
        facts = _trace_facts()

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if "Rewrite the explanation" in prompt:
                return PREFERRED_TRACE
            return LONG_TRACE_CHRONOLOGY

        text, _view_out, warns, diag = _synthesize_condition(
            condition="trace",
            raw_evidence={
                "factual_decision": "accept",
                "execution_trace": [
                    "build_outline#1",
                    "generate_learning_objectives#1",
                    "initial_retrieve_evidence_for_all_sections#1",
                    "plan_mcq_blueprints#1",
                    "generate_candidate_mcqs_from_blueprints#1",
                    "verify_candidate_mcqs_against_evidence#1",
                    "apply_acceptance_gate#1",
                ],
            },
            question="Why was this question selected?",
            factual_decision="accept",
            question_stem="What is unsupervised learning?",
            llm_generate=llm,
            evidence_view=_view(facts),
        )
        self.assertEqual(len(calls), 2)
        self.assertTrue(diag["synthesis_repair_attempted"])
        self.assertTrue(diag["participant_presentation_valid"])
        self.assertFalse(has_detailed_workflow_chronology(text))
        self.assertIn("does not identify", text.lower())
        ok, faith_warns = validate_explanation_faithfulness(
            condition="trace", explanation=text, evidence_view=_view(facts)
        )
        self.assertTrue(ok, faith_warns)
        self.assertFalse(any("mechanical_fallback" in w for w in warns))

    def test_no_llm_uses_mechanical_density(self):
        text, _v, _w, diag = _synthesize_condition(
            condition="trace",
            raw_evidence={"factual_decision": "accept", "execution_trace": ["gen#1"]},
            question="Why?",
            factual_decision="accept",
            question_stem="stem",
            llm_generate=None,
            evidence_view=_view(_trace_facts()),
        )
        self.assertTrue(diag["synthesis_used_mechanical_fallback"])
        self.assertEqual(count_presentation_sentences(text), 1)
        self.assertIn("participant_presentation_violations", diag)
        self.assertTrue(diag["participant_presentation_valid"])


class TestLiveLikeAcceptance(unittest.TestCase):
    def test_three_topics_mechanical_suite(self):
        trace = mechanical_synthesize_from_lay_facts(facts=_trace_facts())
        tp_ul = mechanical_provenance_explanation(
            _tp_facts(
                "unsupervised learning tries to understand the structure of data "
                "without labeled outputs"
            )
        )
        tp_gmm = mechanical_provenance_explanation(
            _tp_facts(
                "a Gaussian mixture model represents density as weighted Gaussians"
            )
        )
        axis = mechanical_synthesize_from_lay_facts(
            facts=_axis_facts(
                description="PCA finds directions of maximum variance",
                role_line=(
                    "The information stating PCA finds directions of maximum "
                    "variance helped keep the selected question focused on linear "
                    "projections."
                ),
                alt_line=(
                    "Without that information, the available content instead "
                    "favored geodesic methods such as ISOMAP."
                ),
            )
        )
        for label, text, facts, cond in (
            ("trace", trace, _trace_facts(), "trace"),
            (
                "tp_ul",
                tp_ul,
                _tp_facts(
                    "unsupervised learning tries to understand the structure of data "
                    "without labeled outputs"
                ),
                "trace_provenance",
            ),
            (
                "tp_gmm",
                tp_gmm,
                _tp_facts(
                    "a Gaussian mixture model represents density as weighted Gaussians"
                ),
                "trace_provenance",
            ),
            (
                "axis",
                axis,
                _axis_facts(
                    description="PCA finds directions of maximum variance",
                    role_line=(
                        "The information stating PCA finds directions of maximum "
                        "variance helped keep the selected question focused on linear "
                        "projections."
                    ),
                    alt_line=(
                        "Without that information, the available content instead "
                        "favored geodesic methods such as ISOMAP."
                    ),
                ),
                "axis",
            ),
        ):
            with self.subTest(label=label):
                self.assertFalse(has_detailed_workflow_chronology(text), text)
                self.assertNotIn(
                    "available during planning, generation, and checking",
                    text.lower(),
                )
                ok, warns = validate_explanation_faithfulness(
                    condition=cond, explanation=text, evidence_view=_view(facts)
                )
                self.assertTrue(ok, warns)
                pres = validate_participant_presentation_policy(text, facts=facts)
                self.assertTrue(pres.valid, pres.violations)
                if cond == "trace_provenance":
                    self.assertNotRegex(text.lower(), r"\binfluenced\b")
                    self.assertNotRegex(text.lower(), r"\bshaped\b")
                if cond == "axis":
                    self.assertIn("helped keep", text.lower())
                    self.assertTrue(
                        "isomap" in text.lower() or "favored" in text.lower(),
                        text,
                    )


if __name__ == "__main__":
    unittest.main()
