"""MCQ question-content comparison for AXIS participant explanations."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import List

from mas_langgraph.artifacts.models import ArtifactRef
from mas_langgraph.axis.comparison import CaseRunObservation, ReplayComparator
from mas_langgraph.axis.models import ExplainedCase, QuerySignature
from mas_langgraph.cases.mcq.comparison import MCQRunObservationAdapter
from mas_langgraph.cases.mcq.content_comparison import (
    compare_mcq_subject_content,
    correct_answer_text,
)
from mas_langgraph.domains.quiz.entities import AnswerOption, CandidateMCQ
from mas_langgraph.experiment.lay_facts import build_lay_explanation_facts
from mas_langgraph.experiment.verbalizer import (
    build_evidence_view,
    participant_section_headings,
)
from mas_langgraph.replay.executor import WorkflowRunResult


def _cand(
    *,
    stem: str,
    correct: str,
    distractors: List[str],
    explanation: str = "because unlabeled",
    q_id: str = "q1",
    blueprint_id: str = "bp1",
) -> CandidateMCQ:
    opts = [AnswerOption(option_id="A", text=correct, is_correct=True)]
    for i, d in enumerate(distractors):
        opts.append(
            AnswerOption(option_id=chr(ord("B") + i), text=d, is_correct=False)
        )
    return CandidateMCQ(
        q_id=q_id,
        blueprint_id=blueprint_id,
        stem=stem,
        options=opts,
        correct_option_id="A",
        answer_explanation=explanation,
    )


def _obs(payload, *, decision="accept", present=True) -> CaseRunObservation:
    return CaseRunObservation(
        run_id="r",
        branch_id="b",
        subject_ref=ArtifactRef("q1", 1),
        decision=decision,
        subject_payload=payload,
        subject_present=present and payload is not None,
        outcome_available=True,
    )


class TestFactualSubjectPayload(unittest.TestCase):
    def test_observe_factual_loads_exact_version(self):
        class _Env:
            def __init__(self, payload):
                self.payload = payload

        v1 = _cand(stem="Q v1?", correct="a", distractors=["b", "c", "d"])
        v2 = _cand(stem="Q v2?", correct="a", distractors=["b", "c", "d"])

        class _Reg:
            def get_by_ref(self, ref, branch=None):
                if ref == ArtifactRef("q12", 1):
                    return _Env(v1)
                if ref == ArtifactRef("q12", 2):
                    return _Env(v2)
                return None

        explained = ExplainedCase(
            run_id="r",
            artifact_ref=ArtifactRef("q12", 1),
            branch_id="factual",
            decision_ref=ArtifactRef("d", 1),
            outcome="accept",
        )
        obs = MCQRunObservationAdapter().observe_factual(
            explained_case=explained,
            case_slice=type("S", (), {"decision": "accept", "criterion_values": (), "decision_reasons": (), "required_fixes": ()})(),
            registry=_Reg(),
        )
        self.assertEqual(obs.subject_payload.stem, "Q v1?")
        self.assertNotEqual(obs.subject_payload.stem, "Q v2?")


class TestContentControlStability(unittest.TestCase):
    def test_identical_noop_stable(self):
        c = _cand(
            stem="What is unsupervised learning?",
            correct="unlabeled",
            distractors=["labeled", "rules", "feedback"],
        )
        delta = compare_mcq_subject_content(
            factual=_obs(c),
            noop=_obs(c),
            counterfactual=_obs(c),
            intervention_before_subject_generation=True,
            subject_generation_relation="before_generation",
        )
        self.assertTrue(delta.control_stable)
        self.assertEqual(delta.effect_status, "question_content_unchanged")

    def test_unstable_noop(self):
        f = _cand(
            stem="What is unsupervised learning?",
            correct="unlabeled",
            distractors=["a", "b", "c"],
        )
        n = _cand(
            stem="Which statement describes unsupervised learning?",
            correct="unlabeled",
            distractors=["a", "b", "c"],
        )
        delta = compare_mcq_subject_content(
            factual=_obs(f),
            noop=_obs(n),
            counterfactual=_obs(n),
            intervention_before_subject_generation=True,
        )
        self.assertFalse(delta.control_stable)
        self.assertEqual(delta.effect_status, "question_content_inconclusive")


class TestContentEffects(unittest.TestCase):
    def test_stem_change(self):
        base = _cand(
            stem="What is unsupervised learning?",
            correct="unlabeled",
            distractors=["a", "b", "c"],
        )
        cf = _cand(
            stem="Which task is associated with unsupervised learning?",
            correct="clustering",
            distractors=["a", "b", "c"],
        )
        delta = compare_mcq_subject_content(
            factual=_obs(base),
            noop=_obs(base),
            counterfactual=_obs(cf),
            intervention_before_subject_generation=True,
        )
        self.assertTrue(delta.details["stem_changed"])
        self.assertTrue(delta.details["content_changed"])
        raw = {
            "factual_decision": "accept",
            "relevant_artifacts": [],
            "replay_comparisons": [
                {
                    "evidence_status": "criterion_effect",
                    "intervention_before_subject_generation": True,
                    "question_content_comparison": delta.to_dict(),
                    "factual": {"decision": "accept"},
                    "noop": {"decision": "accept"},
                    "counterfactual": {"decision": "accept"},
                    "criterion_deltas": [
                        {
                            "name": "target_atom_count",
                            "factual": 2,
                            "noop": 2,
                            "counterfactual": 9,
                            "decision_relevant": True,
                            "counterfactual_differs_from_noop": True,
                        }
                    ],
                    "intervention_signature": {
                        "operation": "REMOVE",
                        "target_ref": {"key": "atom_1:v1"},
                        "checkpoint_id": "before plan_mcq_blueprints#1",
                    },
                }
            ],
        }
        facts = build_lay_explanation_facts(condition="axis", raw_evidence=raw)
        joined = " ".join(facts.question_content_findings).lower()
        self.assertTrue(
            "changed from" in joined
            or "conceptual focus" in joined
            or "question asks" in joined
            or "phrased" in joined
            or "influenced how" in joined,
            joined,
        )
        # Stem wording may still appear for focus-level shifts; formulation may omit quotes.
        if "phrased" not in joined and "influenced how" not in joined:
            self.assertIn("what is unsupervised learning?", joined)
            self.assertIn("which task is associated", joined)
        self.assertNotIn("two to nine", joined)
        self.assertNotIn("supporting knowledge", joined)

    def test_correct_answer_change(self):
        base = _cand(
            stem="What is unsupervised learning?",
            correct="unlabeled data",
            distractors=["a", "b", "c"],
        )
        cf = _cand(
            stem="What is unsupervised learning?",
            correct="clustering only",
            distractors=["a", "b", "c"],
        )
        delta = compare_mcq_subject_content(
            factual=_obs(base), noop=_obs(base), counterfactual=_obs(cf),
            intervention_before_subject_generation=True,
        )
        self.assertTrue(delta.details["correct_answer_changed"])
        self.assertFalse(delta.details["stem_changed"])

    def test_distractor_only(self):
        base = _cand(
            stem="What is unsupervised learning?",
            correct="unlabeled",
            distractors=["labeled", "rules", "teacher"],
        )
        cf = _cand(
            stem="What is unsupervised learning?",
            correct="unlabeled",
            distractors=["labeled", "rules", "reward"],
        )
        delta = compare_mcq_subject_content(
            factual=_obs(base), noop=_obs(base), counterfactual=_obs(cf),
            intervention_before_subject_generation=True,
        )
        self.assertTrue(delta.details["distractors_changed"])
        self.assertFalse(delta.details["stem_changed"])
        self.assertFalse(delta.details["correct_answer_changed"])
        raw = {
            "factual_decision": "accept",
            "relevant_artifacts": [],
            "replay_comparisons": [
                {
                    "evidence_status": "no_demonstrated_effect",
                    "intervention_before_subject_generation": True,
                    "question_content_comparison": delta.to_dict(),
                    "factual": {"decision": "accept"},
                    "noop": {"decision": "accept"},
                    "counterfactual": {"decision": "accept"},
                    "intervention_signature": {
                        "operation": "REMOVE",
                        "target_ref": {"key": "atom_1:v1"},
                        "checkpoint_id": "before plan_mcq_blueprints#1",
                    },
                }
            ],
        }
        facts = build_lay_explanation_facts(condition="axis", raw_evidence=raw)
        joined = " ".join(facts.question_content_findings).lower()
        self.assertTrue(
            "incorrect" in joined
            and ("alternative" in joined or "option" in joined or "shape" in joined),
            joined,
        )
        self.assertNotIn("different question", joined)
        self.assertNotIn("the question stayed the same", joined)
        self.assertEqual(facts.changed_distractor_count, 1)
        self.assertEqual(facts.information_role, "distractor_shaping")
        self.assertEqual(facts.causal_classification, "influential_non_decisive")

    def test_unchanged_pre_generation(self):
        c = _cand(
            stem="What is unsupervised learning?",
            correct="unlabeled",
            distractors=["a", "b", "c"],
        )
        delta = compare_mcq_subject_content(
            factual=_obs(c), noop=_obs(c), counterfactual=_obs(c),
            intervention_before_subject_generation=True,
        )
        raw = {
            "factual_decision": "accept",
            "relevant_artifacts": [],
            "replay_comparisons": [
                {
                    "evidence_status": "no_demonstrated_effect",
                    "intervention_before_subject_generation": True,
                    "question_content_comparison": delta.to_dict(),
                    "factual": {"decision": "accept"},
                    "noop": {"decision": "accept"},
                    "counterfactual": {"decision": "accept"},
                    "intervention_signature": {
                        "operation": "REMOVE",
                        "target_ref": {"key": "atom_c:v1"},
                        "checkpoint_id": "before plan_mcq_blueprints#1",
                    },
                }
            ],
        }
        facts = build_lay_explanation_facts(condition="axis", raw_evidence=raw)
        joined = " ".join(facts.question_content_findings).lower()
        self.assertTrue(
            "robust" in joined
            or "remained the same" in joined
            or "unchanged" in joined
            or "stable" in joined
            or "left the question wording" in joined,
            joined,
        )
        self.assertNotIn("necessary", joined)

    def test_post_generation_no_generation_claim(self):
        c = _cand(
            stem="What is unsupervised learning?",
            correct="unlabeled",
            distractors=["a", "b", "c"],
        )
        delta = compare_mcq_subject_content(
            factual=_obs(c), noop=_obs(c), counterfactual=_obs(c),
            intervention_before_subject_generation=False,
            subject_generation_relation="after_generation",
        )
        raw = {
            "factual_decision": "accept",
            "relevant_artifacts": [],
            "replay_comparisons": [
                {
                    "evidence_status": "criterion_effect",
                    "intervention_before_subject_generation": False,
                    "question_content_comparison": delta.to_dict(),
                    "factual": {"decision": "accept"},
                    "noop": {"decision": "accept"},
                    "counterfactual": {"decision": "accept"},
                    "intervention_signature": {
                        "operation": "REMOVE",
                        "target_ref": {"key": "c:v1"},
                    },
                }
            ],
        }
        facts = build_lay_explanation_facts(condition="axis", raw_evidence=raw)
        joined = " ".join(facts.question_content_findings).lower()
        self.assertIn("after the question had already been created", joined)
        self.assertNotIn("would have been generated", joined)

    def test_question_disappears(self):
        c = _cand(
            stem="What is unsupervised learning?",
            correct="unlabeled",
            distractors=["a", "b", "c"],
        )
        delta = compare_mcq_subject_content(
            factual=_obs(c),
            noop=_obs(c),
            counterfactual=_obs(None, present=False),
            intervention_before_subject_generation=True,
        )
        self.assertEqual(delta.effect_status, "question_not_reproduced")
        raw = {
            "factual_decision": "accept",
            "relevant_artifacts": [],
            "replay_comparisons": [
                {
                    "evidence_status": "downstream_effect",
                    "intervention_before_subject_generation": True,
                    "question_content_comparison": delta.to_dict(),
                    "factual": {"decision": "accept"},
                    "noop": {"decision": "accept"},
                    "counterfactual": {"decision": None, "subject_present": False},
                    "intervention_signature": {
                        "operation": "REMOVE",
                        "target_ref": {"key": "atom_c:v1"},
                        "checkpoint_id": "before plan_mcq_blueprints#1",
                    },
                }
            ],
        }
        facts = build_lay_explanation_facts(condition="axis", raw_evidence=raw)
        joined = " ".join(facts.question_content_findings).lower()
        self.assertTrue(
            "reproducing this particular question" in joined
            or "did not generate a corresponding question" in joined
            or "did not yield a corresponding question" in joined,
            joined,
        )


class TestConditionWording(unittest.TestCase):
    def test_shared_prompt_no_forced_sections(self):
        from mas_langgraph.experiment.synthesis import SHARED_FINAL_SYNTHESIS_PROMPT

        self.assertNotIn("USE EXACTLY THESE SECTIONS", SHARED_FINAL_SYNTHESIS_PROMPT)

    def test_trace_limitation(self):
        facts = build_lay_explanation_facts(
            condition="trace",
            raw_evidence={"factual_decision": "accept", "execution_trace": ["g#1"]},
        )
        joined = " ".join(facts.limitations).lower()
        self.assertTrue(
            "does not identify" in joined
            or "cannot determine" in joined
            or "not which specific teaching" in joined
            or "responsible for the exact" in joined,
            joined,
        )

    def test_trace_provenance_no_causal_content(self):
        facts = build_lay_explanation_facts(
            condition="trace_provenance",
            raw_evidence={
                "factual_decision": "accept",
                "relevant_artifacts": [
                    {
                        "ref": "a:v1",
                        "artifact_type": "KnowledgeAtom",
                        "content_preview": (
                            "Unsupervised learning learns patterns without labels."
                        ),
                    }
                ],
            },
        )
        joined = " ".join(facts.limitations).lower() + " " + " ".join(
            facts.relevant_provenance_facts
        ).lower()
        self.assertTrue(
            "linked" in joined
            or "stating that" in joined
            or "patterns without labels" in joined
            or "records link" in joined,
            joined,
        )
        self.assertNotIn("would have been different", joined)
        self.assertNotIn("available during planning, generation, and checking", joined)


class TestComponentLevelStability(unittest.TestCase):
    def test_stable_stem_unstable_distractors(self):
        """PDF-141 failure mode: stem control OK, options not."""
        f = _cand(
            stem="What is unsupervised learning?",
            correct="unlabeled",
            distractors=["a", "b", "c"],
        )
        n = _cand(
            stem="What is unsupervised learning?",
            correct="patterns",
            distractors=["x", "y", "z"],
        )
        cf = _cand(
            stem="What is the primary goal of unsupervised learning algorithms?",
            correct="patterns",
            distractors=["x", "y", "z"],
        )
        delta = compare_mcq_subject_content(
            factual=_obs(f),
            noop=_obs(n),
            counterfactual=_obs(cf),
            intervention_before_subject_generation=True,
        )
        d = delta.details
        self.assertTrue(d["stem_control_stable"])
        self.assertEqual(d["stem_effect"], "changed")
        self.assertFalse(d["distractors_control_stable"])
        self.assertEqual(d["distractor_effect"], "inconclusive")
        self.assertFalse(d["correct_answer_control_stable"])
        self.assertEqual(d["correct_answer_effect"], "inconclusive")
        self.assertEqual(delta.effect_status, "question_content_partially_supported")
        self.assertFalse(delta.control_stable)  # core not fully stable

        raw = {
            "factual_decision": "accept",
            "relevant_artifacts": [],
            "replay_comparisons": [
                {
                    "evidence_status": "no_demonstrated_effect",
                    "intervention_before_subject_generation": True,
                    "question_content_comparison": delta.to_dict(),
                    "factual": {"decision": "accept"},
                    "noop": {"decision": "accept"},
                    "counterfactual": {"decision": "accept"},
                    "intervention_signature": {
                        "operation": "REMOVE",
                        "target_ref": {"key": "atom_00000001:v1"},
                    },
                }
            ],
        }
        facts = build_lay_explanation_facts(condition="axis", raw_evidence=raw)
        joined = " ".join(facts.question_content_findings).lower()
        self.assertTrue(
            "changed from" in joined
            or "conceptual focus" in joined
            or "question asks" in joined
            or "phrased" in joined
            or "influenced how" in joined,
            joined,
        )
        self.assertNotIn("incorrect option", joined)
        self.assertNotIn("distractor", joined)
        self.assertEqual(facts.question_changed_parts, ("stem",))
        self.assertEqual(facts.stem_effect, "changed")
        self.assertEqual(facts.distractor_effect, "inconclusive")
        self.assertEqual(facts.correct_answer_effect, "inconclusive")
        self.assertNotIn("teaching material", joined)

    def test_unstable_stem(self):
        f = _cand(
            stem="What is unsupervised learning?",
            correct="unlabeled",
            distractors=["a", "b", "c"],
        )
        n = _cand(
            stem="What is an advantage of the k-Means algorithm?",
            correct="speed",
            distractors=["a", "b", "c"],
        )
        cf = _cand(
            stem="What is an advantage of the k-Means algorithm?",
            correct="speed",
            distractors=["a", "b", "c"],
        )
        delta = compare_mcq_subject_content(
            factual=_obs(f),
            noop=_obs(n),
            counterfactual=_obs(cf),
            intervention_before_subject_generation=True,
        )
        self.assertFalse(delta.details["stem_control_stable"])
        self.assertEqual(delta.details["stem_effect"], "inconclusive")
        self.assertEqual(delta.effect_status, "question_content_inconclusive")
        raw = {
            "factual_decision": "accept",
            "relevant_artifacts": [],
            "replay_comparisons": [
                {
                    "evidence_status": "no_demonstrated_effect",
                    "intervention_before_subject_generation": True,
                    "question_content_comparison": delta.to_dict(),
                    "factual": {"decision": "accept"},
                    "noop": {"decision": "accept"},
                    "counterfactual": {"decision": "accept"},
                    "intervention_signature": {
                        "operation": "REMOVE",
                        "target_ref": {"key": "atom:v1"},
                    },
                }
            ],
        }
        facts = build_lay_explanation_facts(condition="axis", raw_evidence=raw)
        joined = " ".join(facts.question_content_findings).lower()
        self.assertTrue(
            "cannot reliably determine" in joined
            or "could not reliably show" in joined,
            joined,
        )
        self.assertNotIn("changed from \"what is unsupervised learning?\"", joined)

    def test_answer_explanation_only_instability(self):
        f = _cand(
            stem="What is unsupervised learning?",
            correct="unlabeled",
            distractors=["a", "b", "c"],
            explanation="because A",
        )
        n = _cand(
            stem="What is unsupervised learning?",
            correct="unlabeled",
            distractors=["a", "b", "c"],
            explanation="because B",
        )
        delta = compare_mcq_subject_content(
            factual=_obs(f),
            noop=_obs(n),
            counterfactual=_obs(n),
            intervention_before_subject_generation=True,
        )
        self.assertTrue(delta.control_stable)
        self.assertTrue(delta.details["core_question_control_stable"])
        self.assertFalse(delta.details["answer_explanation_control_stable"])
        self.assertEqual(delta.effect_status, "question_content_unchanged")

    def test_prefer_stem_over_early_inconclusive(self):
        from mas_langgraph.experiment.question_content_facts import (
            select_primary_question_content_finding,
        )

        f = _cand(stem="X", correct="a", distractors=["b", "c", "d"])
        n_bad = _cand(stem="Z", correct="a", distractors=["b", "c", "d"])
        cf_bad = _cand(stem="Y", correct="a", distractors=["b", "c", "d"])
        bad = compare_mcq_subject_content(
            factual=_obs(f), noop=_obs(n_bad), counterfactual=_obs(cf_bad),
            intervention_before_subject_generation=True,
        )
        n_ok = _cand(stem="X", correct="a", distractors=["b", "c", "d"])
        cf_ok = _cand(stem="Y", correct="a", distractors=["b", "c", "d"])
        good = compare_mcq_subject_content(
            factual=_obs(f), noop=_obs(n_ok), counterfactual=_obs(cf_ok),
            intervention_before_subject_generation=True,
        )
        lines, meta, _ = select_primary_question_content_finding(
            [
                {
                    "intervention_before_subject_generation": True,
                    "question_content_comparison": bad.to_dict(),
                    "noop": {"decision": "accept"},
                    "counterfactual": {"decision": "accept"},
                },
                {
                    "intervention_before_subject_generation": True,
                    "question_content_comparison": good.to_dict(),
                    "noop": {"decision": "accept"},
                    "counterfactual": {"decision": "accept"},
                },
            ]
        )
        joined = " ".join(lines).lower()
        self.assertTrue(
            "changed from" in joined
            or "conceptual focus" in joined
            or "question asks" in joined,
            joined,
        )
        self.assertTrue(meta.get("stem_changed") or meta.get("stem_effect") == "changed")
        self.assertEqual(meta.get("stem_effect"), "changed")
        # Must not leak secondary intervention effects into primary meta.
        self.assertNotEqual(meta.get("correct_answer_effect"), "changed")
        self.assertNotEqual(meta.get("distractor_effect"), "changed")
        self.assertIn("stem", meta.get("question_changed_parts") or [])


class TestAggregationSingleSource(unittest.TestCase):
    def test_primary_fields_from_one_comparison_only(self):
        from mas_langgraph.experiment.question_content_facts import (
            select_primary_question_content_finding,
        )

        # A: stem-only change
        f = _cand(stem="X", correct="a", distractors=["b", "c", "d"])
        n = _cand(stem="X", correct="a", distractors=["b", "c", "d"])
        cf_a = _cand(stem="Y", correct="a", distractors=["b", "c", "d"])
        a = compare_mcq_subject_content(
            factual=_obs(f), noop=_obs(n), counterfactual=_obs(cf_a),
            intervention_before_subject_generation=True,
        )
        # B: same stem, correct+distractors changed
        cf_b = _cand(stem="X", correct="z", distractors=["p", "q", "r"])
        b = compare_mcq_subject_content(
            factual=_obs(f), noop=_obs(n), counterfactual=_obs(cf_b),
            intervention_before_subject_generation=True,
        )
        _, meta, primary = select_primary_question_content_finding(
            [
                {
                    "intervention_before_subject_generation": True,
                    "question_content_comparison": a.to_dict(),
                    "intervention_signature": {
                        "operation": "REMOVE",
                        "target_ref": {"key": "atom_a:v1"},
                    },
                    "noop": {"decision": "accept"},
                    "counterfactual": {"decision": "accept"},
                },
                {
                    "intervention_before_subject_generation": True,
                    "question_content_comparison": b.to_dict(),
                    "intervention_signature": {
                        "operation": "REMOVE",
                        "target_ref": {"key": "atom_b:v1"},
                    },
                    "noop": {"decision": "accept"},
                    "counterfactual": {"decision": "accept"},
                },
            ]
        )
        self.assertEqual(meta["stem_effect"], "changed")
        self.assertEqual(meta["correct_answer_effect"], "unchanged")
        self.assertEqual(meta["distractor_effect"], "unchanged")
        self.assertEqual(list(meta["question_changed_parts"]), ["stem"])
        self.assertTrue(meta["stem_changed"])
        self.assertFalse(meta["correct_answer_changed"])
        self.assertFalse(meta["distractors_changed"])
        self.assertEqual(
            (meta.get("primary_question_content_intervention") or {}).get("target_ref"),
            "atom_a:v1",
        )
        self.assertIsNotNone(primary)


if __name__ == "__main__":
    unittest.main()
