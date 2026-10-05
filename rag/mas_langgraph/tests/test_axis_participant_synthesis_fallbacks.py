"""Q7/Q8-like AXIS participant synthesis: keep the validator, fix verbalization."""

from __future__ import annotations

import unittest
from typing import List

from mas_langgraph.experiment.faithfulness import validate_explanation_faithfulness
from mas_langgraph.experiment.lay_facts import LayExplanationFacts
from mas_langgraph.experiment.service import _synthesize_condition
from mas_langgraph.experiment.synthesis import build_faithfulness_repair_prompt
from mas_langgraph.experiment.verbalizer import ExplanationEvidenceView, build_evidence_view


def _q7_facts() -> LayExplanationFacts:
    return LayExplanationFacts(
        decision="ACCEPT",
        educational_purpose=("The system generated and checked the question.",),
        trace_narrative=("The system generated and checked the question.",),
        relevant_provenance_facts=(
            "The records link the question to teaching material about PCA.",
        ),
        counterfactual_component_findings=(
            "The information stating PCA helped keep the selected question "
            "focused on its particular conceptual target.",
        ),
        question_content_findings=(
            "The information stating PCA helped keep the selected question "
            "focused on its particular conceptual target.",
        ),
        specific_source_known=True,
        recorded_use_known=True,
        information_role_known=True,
        counterfactual_effect_known=True,
        decision_effect_known=False,
        information_role="focus_shaping",
        information_role_supported=True,
        primary_information_description="PCA",
        question_identity_effect=False,
        question_identity_effect_known=True,
        acceptance_effect=False,
        acceptance_effect_known=True,
        stem_changed=False,
        correct_answer_changed=False,
        distractors_changed=False,
        primary_question_content_status="question_content_unchanged",
        intervention_before_question_generation=True,
    )


def _q8_facts() -> LayExplanationFacts:
    factual = "What is the key characteristic of eigenfaces in facial recognition?"
    alternative = (
        "You have a dataset where local distances between points are known "
        "and want an ISOMAP embedding."
    )
    return LayExplanationFacts(
        decision="ACCEPT",
        educational_purpose=("The system generated and checked the question.",),
        trace_narrative=("The system generated and checked the question.",),
        counterfactual_component_findings=(
            "The information stating Eigenfaces (courtesy ORL database) helped "
            "keep the selected question focused on its particular conceptual target.",
        ),
        question_content_findings=(
            "The information stating Eigenfaces (courtesy ORL database) helped "
            "keep the selected question focused on its particular conceptual target.",
        ),
        specific_source_known=True,
        recorded_use_known=True,
        information_role_known=True,
        counterfactual_effect_known=True,
        decision_effect_known=False,
        information_role="focus_shaping",
        information_role_supported=True,
        primary_information_description="Eigenfaces (courtesy ORL database)",
        question_identity_effect=True,
        question_identity_effect_known=True,
        acceptance_effect=False,
        acceptance_effect_known=True,
        stem_changed=True,
        correct_answer_changed=False,
        distractors_changed=False,
        primary_question_content_status="question_content_changed",
        intervention_before_question_generation=True,
        factual_question_stem=factual,
        counterfactual_question_stem=alternative,
    )


def _view(facts: LayExplanationFacts) -> ExplanationEvidenceView:
    return ExplanationEvidenceView(
        factual_decision=facts.decision,
        tested_interventions=facts.question_content_findings,
        lay_facts=facts,
    )


class TestPromptProjectsTestedInformation(unittest.TestCase):
    def test_axis_prompt_names_tested_information_and_stems(self):
        facts = _q8_facts()
        text = facts.to_prompt_text()
        self.assertIn("Tested teaching information:", text)
        self.assertIn("description = Eigenfaces (courtesy ORL database)", text)
        self.assertIn("role = focus_shaping", text)
        self.assertIn("role_supported = true", text)
        self.assertIn("stem_changed = true", text)
        self.assertIn("factual_question =", text)
        self.assertIn("alternative_question =", text)
        self.assertIn("ISOMAP", text)

    def test_trace_prompt_omits_tested_information_block(self):
        view = build_evidence_view(
            condition="trace",
            raw_evidence={
                "factual_decision": "accept",
                "execution_trace": ["generate_candidate_mcqs_from_blueprints#1"],
            },
        )
        text = view.to_prompt_text()
        self.assertNotIn("Tested teaching information:", text)
        self.assertNotIn("alternative_question =", text)
        self.assertIn("information_role_known = false", text)


class TestQ7LikeSupportingMaterial(unittest.TestCase):
    def test_vague_supporting_material_is_rejected(self):
        text = "The supporting material influenced how the question was written."
        ok, warns = validate_explanation_faithfulness(
            condition="axis",
            explanation=text,
            evidence_view=_view(_q7_facts()),
        )
        self.assertFalse(ok)
        self.assertTrue(
            any("supporting material without the tested information" in w for w in warns),
            warns,
        )

    def test_named_tested_information_passes(self):
        text = (
            "The information stating PCA helped keep the selected question "
            "focused on its particular conceptual target. The resulting "
            "question still passed the recorded checks."
        )
        ok, warns = validate_explanation_faithfulness(
            condition="axis",
            explanation=text,
            evidence_view=_view(_q7_facts()),
        )
        self.assertTrue(ok, warns)

    def test_scripted_repair_avoids_mechanical_fallback(self):
        calls: List[str] = []

        def llm(prompt: str) -> str:
            calls.append(prompt)
            if "Rewrite the explanation" in prompt or "failed faithfulness" in prompt:
                self.assertIn(
                    "Name the concrete tested teaching information",
                    prompt,
                )
                self.assertIn(
                    "Preserve at least one supported information-role finding",
                    prompt,
                )
                return (
                    "The information stating PCA helped keep the selected "
                    "question focused on its particular conceptual target. "
                    "The resulting question still passed the recorded checks."
                )
            return "The supporting material influenced how the question was written."

        text, view, warns, diag = _synthesize_condition_with_view(
            condition="axis",
            view=_view(_q7_facts()),
            question="Why was this question selected?",
            factual_decision="accept",
            question_stem="What is an advantage of using PCA?",
            llm_generate=llm,
        )
        self.assertFalse(diag["synthesis_used_fallback"], msg=str(diag))
        self.assertFalse(diag["synthesis_used_mechanical_fallback"])
        self.assertTrue(diag["synthesis_repair_valid"])
        self.assertIn("PCA", text)
        self.assertNotIn("supporting material", text.lower())


def _synthesize_condition_with_view(
    *,
    condition: str,
    view: ExplanationEvidenceView,
    question: str,
    factual_decision: str,
    question_stem: str,
    llm_generate,
):
    return _synthesize_condition(
        condition=condition,
        raw_evidence={},
        question=question,
        factual_decision=factual_decision,
        question_stem=question_stem,
        llm_generate=llm_generate,
        evidence_view=view,
    )


class TestQ8LikeStemChange(unittest.TestCase):
    def test_topic_changed_without_stems_is_rejected(self):
        text = "The topic changed because of this information."
        ok, warns = validate_explanation_faithfulness(
            condition="axis",
            explanation=text,
            evidence_view=_view(_q8_facts()),
        )
        self.assertFalse(ok)
        self.assertTrue(
            any("stem change not stated" in w for w in warns),
            warns,
        )

    def test_stating_verified_stems_passes(self):
        facts = _q8_facts()
        text = (
            "Eigenfaces (courtesy ORL database) helped keep the selected question "
            "focused on its particular conceptual target. With this information, "
            f"the question asked '{facts.factual_question_stem}'. When this "
            "information was absent, the available content instead produced "
            f"'{facts.counterfactual_question_stem}'. This information influenced "
            "which question was produced, while the workflow could still produce "
            "an acceptable MCQ."
        )
        ok, warns = validate_explanation_faithfulness(
            condition="axis",
            explanation=text,
            evidence_view=_view(facts),
        )
        self.assertTrue(ok, warns)

    def test_scripted_repair_states_stems_without_fallback(self):
        facts = _q8_facts()

        def llm(prompt: str) -> str:
            if "Rewrite the explanation" in prompt or "failed faithfulness" in prompt:
                self.assertIn(
                    "factual-vs-alternative question-stem difference",
                    prompt,
                )
                return (
                    "Eigenfaces (courtesy ORL database) helped keep the selected "
                    "question focused on its particular conceptual target. With "
                    f"this information, the question asked '{facts.factual_question_stem}'. "
                    "When this information was absent, the available content instead "
                    f"produced '{facts.counterfactual_question_stem}'. This information "
                    "influenced which question was produced, while the workflow could "
                    "still produce an acceptable MCQ."
                )
            return "The topic changed because of this information."

        text, _ev, warns, diag = _synthesize_condition_with_view(
            condition="axis",
            view=_view(facts),
            question="Why was this question selected?",
            factual_decision="accept",
            question_stem=facts.factual_question_stem or "",
            llm_generate=llm,
        )
        self.assertFalse(diag["synthesis_used_fallback"], msg=str(warns))
        self.assertTrue(diag["synthesis_repair_valid"])
        self.assertIn("eigenfaces", text.lower())
        self.assertIn("isomap", text.lower())


class TestRepairPromptIsWarningDriven(unittest.TestCase):
    def test_repair_prompt_contains_generic_warning_instructions(self):
        prompt = build_faithfulness_repair_prompt(
            previous_explanation="x",
            warnings=(
                "AXIS refers to supporting material without the tested information",
                "AXIS explanation omits valid tested-intervention evidence",
                "AXIS omits question-content effect: stem change not stated",
            ),
            verbalized_evidence="facts",
        )
        self.assertIn("Name the concrete tested teaching information", prompt)
        self.assertIn("supplied tested-information description exactly as", prompt)
        self.assertIn("Preserve at least one supported information-role finding", prompt)
        self.assertIn("factual-vs-alternative question-stem difference", prompt)
        self.assertNotIn("q_00000007", prompt)
        self.assertNotIn("q_00000008", prompt)


class TestAcronymTestedInformationMatcher(unittest.TestCase):
    def test_pca_named_with_teaching_material_passes(self):
        text = (
            "The information 'PCA' helped keep the question focused on its "
            "particular conceptual target. The records also mention teaching "
            "material used during checking. The resulting question still passed "
            "the recorded checks."
        )
        ok, warns = validate_explanation_faithfulness(
            condition="axis",
            explanation=text,
            evidence_view=_view(_q7_facts()),
        )
        self.assertTrue(ok, warns)
        self.assertFalse(
            any("supporting material without the tested information" in w for w in warns)
        )

    def test_gmm_acronym_required(self):
        facts = LayExplanationFacts(
            decision="ACCEPT",
            educational_purpose=("The system generated and checked the question.",),
            counterfactual_component_findings=(
                "The information stating GMM helped keep the selected question "
                "focused on its particular conceptual target.",
            ),
            question_content_findings=(
                "The information stating GMM helped keep the selected question "
                "focused on its particular conceptual target.",
            ),
            information_role_known=True,
            recorded_use_known=True,
            counterfactual_effect_known=True,
            information_role="focus_shaping",
            information_role_supported=True,
            primary_information_description="GMM",
            intervention_before_question_generation=True,
            stem_changed=False,
        )
        ok_pos, warns_pos = validate_explanation_faithfulness(
            condition="axis",
            explanation=(
                "The information 'GMM' helped keep the question focused on its "
                "particular conceptual target."
            ),
            evidence_view=_view(facts),
        )
        self.assertTrue(ok_pos, warns_pos)
        ok_neg, warns_neg = validate_explanation_faithfulness(
            condition="axis",
            explanation="The supporting material influenced the question.",
            evidence_view=_view(facts),
        )
        self.assertFalse(ok_neg)
        self.assertTrue(
            any("supporting material without the tested information" in w for w in warns_neg),
            warns_neg,
        )


def _q5_facts() -> LayExplanationFacts:
    return LayExplanationFacts(
        decision="ACCEPT",
        educational_purpose=("The system generated and checked the question.",),
        trace_narrative=("The system generated and checked the question.",),
        counterfactual_component_findings=(
            "The information stating Update the mean and variance parameters "
            "to match the assigned data points in each cluster during the "
            "Gaussian Mixture Model algorithm helped keep the selected question "
            "focused on its particular conceptual target.",
            "The information stating Place k means at random to initialize the "
            "Gaussian Mixture Model algorithm helped determine what counted as "
            "the correct answer.",
        ),
        question_content_findings=(
            "The information stating Update the mean and variance parameters "
            "to match the assigned data points in each cluster during the "
            "Gaussian Mixture Model algorithm helped keep the selected question "
            "focused on its particular conceptual target.",
        ),
        specific_source_known=True,
        recorded_use_known=True,
        information_role_known=True,
        counterfactual_effect_known=True,
        decision_effect_known=False,
        information_role="focus_shaping",
        information_role_supported=True,
        primary_information_description=(
            "Update the mean and variance parameters to match the assigned data "
            "points in each cluster during the Gaussian Mixture Model algorithm."
        ),
        question_identity_effect=True,
        question_identity_effect_known=True,
        acceptance_effect=False,
        acceptance_effect_known=True,
        stem_changed=True,
        correct_answer_changed=False,
        distractors_changed=False,
        primary_question_content_status="question_content_changed",
        intervention_before_question_generation=True,
        factual_question_stem="Which of the following correctly describes a GMM step?",
        counterfactual_question_stem="What is the first step of k-means clustering?",
    )


class TestQ5SupportedRoleRepair(unittest.TestCase):
    def test_associated_with_downgrade_fails(self):
        facts = _q5_facts()
        text = (
            "Update the mean and variance parameters to match the assigned data "
            "points in each cluster during the Gaussian Mixture Model algorithm "
            "was associated with the selected question's focus. Place k means at "
            "random to initialize the Gaussian Mixture Model algorithm was "
            "associated with what counted as the correct answer. With this "
            "information, the question asked "
            f"'{facts.factual_question_stem}'. When this information was absent, "
            "the available content instead produced "
            f"'{facts.counterfactual_question_stem}'."
        )
        ok, warns = validate_explanation_faithfulness(
            condition="axis",
            explanation=text,
            evidence_view=_view(facts),
        )
        self.assertFalse(ok)
        self.assertTrue(
            any("omits valid tested-intervention evidence" in w for w in warns),
            warns,
        )

    def test_concise_supported_role_repair_passes(self):
        facts = _q5_facts()
        text = (
            "Update the mean and variance parameters to match the assigned data "
            "points in each cluster during the Gaussian Mixture Model algorithm "
            "helped keep the question focused on that GMM update step. Place k "
            "means at random to initialize the Gaussian Mixture Model algorithm "
            "helped determine the correct answer. With this information, the "
            f"question asked '{facts.factual_question_stem}'. When this "
            "information was absent, the available content instead produced "
            f"'{facts.counterfactual_question_stem}'. The workflow could still "
            "produce an acceptable MCQ."
        )
        from mas_langgraph.experiment.synthesis import MAX_PARTICIPANT_WORDS, count_words

        self.assertLessEqual(count_words(text), MAX_PARTICIPANT_WORDS)
        ok, warns = validate_explanation_faithfulness(
            condition="axis",
            explanation=text,
            evidence_view=_view(facts),
        )
        self.assertTrue(ok, warns)


class TestQuoteAwareWordLimit(unittest.TestCase):
    def test_internal_period_in_quoted_stem_is_atomic(self):
        from mas_langgraph.experiment.synthesis import (
            MAX_PARTICIPANT_WORDS,
            count_words,
            enforce_participant_word_limit,
            participant_quotes_are_balanced,
        )

        stem = (
            "You have a dataset where you want to reduce dimensionality while "
            "preserving the intrinsic geometry of the data. Which of the "
            "following methods is most appropriate?"
        )
        generic = (
            "The system organized the teaching material into sections, created "
            "learning objectives, retrieved candidate supporting material, "
            "extracted concise factual statements, and created a plan for the "
            "question. "
        )
        body = (
            "Eigenfaces (courtesy ORL database) helped keep the selected question "
            "focused on its particular conceptual target. With this information, "
            'the question asked "What is the key characteristic of eigenfaces '
            'in facial recognition?". When this information was absent, the '
            f'available content instead produced "{stem}" This information '
            "influenced which question was produced, while the workflow could "
            "still produce an acceptable MCQ."
        )
        long = (generic * 20) + body
        self.assertGreater(count_words(long), MAX_PARTICIPANT_WORDS)
        out, used = enforce_participant_word_limit(long)
        self.assertTrue(used)
        self.assertLessEqual(count_words(out), MAX_PARTICIPANT_WORDS)
        self.assertTrue(participant_quotes_are_balanced(out))
        self.assertFalse(out.rstrip().endswith("data."))
        self.assertNotRegex(out, r'geometry of the data\.\s*$')
        if "intrinsic geometry of the data" in out:
            self.assertIn("Which of the following", out)

    def test_valid_overlong_repair_is_safe_compression_not_mechanical(self):
        facts = _q8_facts()
        stem = facts.counterfactual_question_stem
        generic = (
            "The system organized the teaching material into sections, created "
            "learning objectives, and retrieved candidate supporting material. "
        )
        repair = (
            generic * 18
            + "Eigenfaces (courtesy ORL database) helped keep the selected "
            "question focused on its particular conceptual target. With this "
            f"information, the question asked '{facts.factual_question_stem}'. "
            "When this information was absent, the available content instead "
            f"produced '{stem}'. This information influenced which question "
            "was produced, while the workflow could still produce an acceptable MCQ."
        )
        from mas_langgraph.experiment.synthesis import count_words

        self.assertGreater(count_words(repair), 200)

        def llm(prompt: str) -> str:
            if "Rewrite the explanation" in prompt or "failed faithfulness" in prompt:
                return repair
            return "The topic changed because of this information."

        text, _ev, warns, diag = _synthesize_condition_with_view(
            condition="axis",
            view=_view(facts),
            question="Why was this question selected?",
            factual_decision="accept",
            question_stem=facts.factual_question_stem or "",
            llm_generate=llm,
        )
        self.assertTrue(diag["synthesis_repair_attempted"])
        self.assertTrue(diag["synthesis_used_mechanical_fallback"])
        self.assertLessEqual(diag["participant_word_count"], 200)
        from mas_langgraph.experiment.synthesis import participant_quotes_are_balanced

        self.assertTrue(participant_quotes_are_balanced(text))
        self.assertIn("eigenfaces", text.lower())
        self.assertNotIn("organized the teaching material", text.lower())
        self.assertTrue(diag["participant_presentation_valid"])

    def test_compression_that_drops_required_evidence_uses_mechanical(self):
        facts = _q8_facts()
        from mas_langgraph.experiment.synthesis import count_words

        pad = " ".join(["geometry"] * 180)
        repair = (
            "Eigenfaces (courtesy ORL database) helped keep the selected "
            "question focused on its particular conceptual target. "
            f'When this information was absent, the available content instead produced "{facts.counterfactual_question_stem} {pad}?"'
        )
        self.assertGreater(count_words(repair), 200)

        def llm(prompt: str) -> str:
            if "Rewrite the explanation" in prompt or "failed faithfulness" in prompt:
                return repair
            return "The topic changed because of this information."

        text, _ev, warns, diag = _synthesize_condition_with_view(
            condition="axis",
            view=_view(facts),
            question="Why was this question selected?",
            factual_decision="accept",
            question_stem=facts.factual_question_stem or "",
            llm_generate=llm,
        )
        self.assertLessEqual(diag["participant_word_count"], 200)
        from mas_langgraph.experiment.synthesis import participant_quotes_are_balanced

        self.assertTrue(participant_quotes_are_balanced(text))
        ok, _ = validate_explanation_faithfulness(
            condition="axis", explanation=text, evidence_view=_view(facts)
        )
        if diag["synthesis_word_limit_strategy"] == "safe_compression":
            self.assertTrue(ok)
            self.assertFalse(diag["synthesis_used_mechanical_fallback"])
        else:
            self.assertTrue(diag["synthesis_used_mechanical_fallback"])
            self.assertEqual(diag["synthesis_word_limit_strategy"], "mechanical_fallback")


if __name__ == "__main__":
    unittest.main()
